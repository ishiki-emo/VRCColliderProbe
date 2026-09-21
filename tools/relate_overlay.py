"""RelateAnything の検出枠と状況説明を、VRChat の画面にリアルタイムに重ねる。

RelateAnything の環境（torch 入り）で動かす。walker.py --relate-overlay から別プロセスで起動されるほか、
単体でも使える:

    E:\\AIProject\\RelateAnything\\.venv\\Scripts\\python.exe tools/relate_overlay.py
    （Ctrl+C で終了）

- VRChat のクライアント領域を撮って推論し、同じ大きさの透明ウィンドウに検出枠を描く
  （黒をカラーキーにして透過、クリックスルー、最前面）
- 左下に「アバター は ledge の上にいる」のような状況説明を出す（関係の文は report.py と同じ）
- OBS や Win+Shift+R の録画に写るよう、キャプチャからは除外しない。推論用の画面はウィンドウの中身だけを撮る
  （capture.WindowCapture の PrintWindow）ので、自分の描いた枠が推論に入り込むことはない
  （--hide-from-recording で以前の動作に戻せる）
- VRChat と GPU を取り合うので、推論の回数は --fps で抑える（既定 4 回/秒）
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture  # noqa: E402
import report  # noqa: E402
import scene_annotate  # noqa: E402
import semantic  # noqa: E402

WINDOW = "VRCColliderProbe relate"
WDA_EXCLUDEFROMCAPTURE = 0x11
FONT = r"C:\Windows\Fonts\meiryo.ttc"
AVATAR_COLOR = (0, 220, 255)       # BGR
OTHER_COLOR = (255, 200, 80)


def parent_alive(pid: int) -> bool:
    """walker.py から起動されたとき、親が終わったらこちらも終わる。"""
    SYNCHRONIZE = 0x00100000
    h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not h:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102   # WAIT_TIMEOUT = まだ動いている
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


class RelateOverlay:
    def __init__(self, vrchat_hwnd: int, exclude_from_capture: bool = False):
        import cv2
        from PIL import ImageFont
        self.cv2 = cv2
        self.vrchat = vrchat_hwnd
        self.exclude_from_capture = exclude_from_capture
        self.hwnd: int | None = None
        self.font = ImageFont.truetype(FONT, 18)
        self.font_small = ImageFont.truetype(FONT, 14)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    def _attach(self) -> bool:
        import win32con
        import win32gui
        if self.hwnd:
            return True
        hwnd = win32gui.FindWindow(None, WINDOW)
        if not hwnd:
            return False
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, win32con.WS_POPUP | win32con.WS_VISIBLE)
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd, win32con.GWL_EXSTYLE,
            ex | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE)
        # 黒（0,0,0）を透明にする。枠と文字だけが見える
        win32gui.SetLayeredWindowAttributes(hwnd, 0x000000, 0, win32con.LWA_COLORKEY)
        if self.exclude_from_capture:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        self.hwnd = hwnd
        return True

    def draw(self, shape, ann: dict, fps: float):
        import numpy as np
        from PIL import Image, ImageDraw
        cv2 = self.cv2
        h, w = shape[:2]
        canvas = np.zeros((h, w, 3), np.uint8)
        me = report.avatar_index(ann)
        for i, o in enumerate(ann["objects"]):
            x1, y1, x2, y2 = (int(v) for v in o["box"])
            avatar = i == me
            color = AVATAR_COLOR if avatar else OTHER_COLOR
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if avatar else 2, cv2.LINE_AA)
            text = ("avatar" if avatar else o["label"]) + f" {o['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1, th + 6)
            cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
            cv2.putText(canvas, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                        cv2.LINE_AA)
        # 左下の状況説明（日本語は PIL で描く）
        lines = report.describe_relations(ann, limit=4) or ["（関係は見つからない）"]
        pad, lh = 10, 26
        box_h = pad * 2 + lh * len(lines) + 20
        box_w = min(w - 32, 460)
        y0 = h - box_h - 16
        img = Image.fromarray(canvas[:, :, ::-1])
        d = ImageDraw.Draw(img)
        # 背景は真っ黒にすると透明になるので、ほぼ黒の濃い灰色にする
        d.rounded_rectangle((16, y0, 16 + box_w, y0 + box_h), radius=8, fill=(24, 24, 24))
        d.text((16 + pad, y0 + pad - 2), f"RelateAnything  {fps:.1f} 回/秒", font=self.font_small,
               fill=(170, 170, 165))
        for i, line in enumerate(lines):
            d.text((16 + pad, y0 + pad + 18 + lh * i), line, font=self.font, fill=(240, 240, 235))
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])

    def show(self, canvas) -> None:
        import win32con
        import win32gui
        self.cv2.imshow(WINDOW, canvas)
        self.cv2.waitKey(1)
        if self._attach():
            r = capture.client_rect(self.vrchat)
            # 最前面にはするが、ミニマップ（overlay.py）はこちらより頻繁に最前面へ出し直すので上に来る
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, r.x, r.y, r.w, r.h,
                                  win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW)

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(WINDOW)
            self.cv2.waitKey(1)
        except Exception:
            pass


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=4.0, help="1 秒あたりの推論回数の上限")
    ap.add_argument("--title", default="VRChat", help="重ねる先のウィンドウのタイトル（部分一致）")
    ap.add_argument("--parent-pid", type=int, default=0, help="この PID が終わったら終了する")
    ap.add_argument("--hide-from-recording", action="store_true",
                    help="オーバーレイを OBS などの録画・スクリーンショットに写さない")
    ap.add_argument("--obs-out", default="",
                    help="見つけた「足元／近くのもの」を JSON Lines で追記するファイル（walker.py がミニマップに使う）")
    args = ap.parse_args()
    obs_f = open(args.obs_out, "a", encoding="utf-8") if args.obs_out else None

    capture.enable_dpi_awareness()
    w = capture.find_window(args.title)
    if w is None:
        print(f"ウィンドウが見つからない: {args.title}")
        return 1
    pipe = scene_annotate.load_pipeline()
    cap = capture.WindowCapture(w[0])
    ov = RelateOverlay(w[0], exclude_from_capture=args.hide_from_recording)
    print("RelateAnything のオーバーレイを開始（Ctrl+C で終了）", flush=True)
    interval = 1.0 / args.fps
    fps = 0.0
    try:
        while True:
            if args.parent_pid and not parent_alive(args.parent_pid):
                break
            t0 = time.perf_counter()
            captured = time.time()   # walker とは別プロセスなので、時刻はエポック秒で渡す
            frame = cap.read()
            if frame is None:        # 最小化中など
                time.sleep(0.5)
                continue
            res = pipe(frame, top_k=12, score_thr=0.30)
            ann = scene_annotate.to_annotation(res, frame)
            if obs_f is not None:
                hits = semantic.relation_hits(ann)
                if hits:
                    obs_f.write(json.dumps({"epoch": round(captured, 3), "hits": hits}, ensure_ascii=False) + "\n")
                    obs_f.flush()
            ov.show(ov.draw(frame.shape, ann, fps))
            spent = time.perf_counter() - t0
            time.sleep(max(0.0, interval - spent))
            fps = 0.8 * fps + 0.2 / max(time.perf_counter() - t0, 1e-3) if fps else 1 / max(spent, 1e-3)
    except KeyboardInterrupt:
        pass
    finally:
        ov.close()
        if obs_f is not None:
            obs_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
