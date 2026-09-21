"""VRChat の画面の右上に、歩いているルートと作りかけの地図をリアルタイムに重ねる。

walker.py --overlay から使う。最前面・クリックスルー・半透明のウィンドウを VRChat の
クライアント領域の右上に置き、毎回 VRChat の位置に追従する。
RelateAnything の TransparentOverlay（tools/game_scene.py）と同じ Win32 の仕組み。

OBS や Win+Shift+R の録画に写るよう、キャプチャからは除外しない（以前は除外していて録画に写らなかった）。
証拠用の画面や推論用の画面には、capture.WindowCapture がウィンドウの中身だけを撮る（PrintWindow）ので写らない。
exclude_from_capture=True で以前の動作（録画にも写らない）に戻せる。

表示: 北（スポーン時の正面）が上。今の位置を中心に半径 RADIUS_M の範囲。
OpenCV は日本語を描けないので、画面上の文字は英数字だけにしている。
"""
from __future__ import annotations

import ctypes
import math

import cv2
import numpy as np

import capture
import explore

WINDOW = "VRCColliderProbe map"
SIZE = 520                 # オーバーレイの一辺の既定値 [px]（300 → 400 → 520 と、見てもらいながら大きくした）
MARGIN = 16                # VRChat の画面の端からの距離 [px]
RADIUS_M = 15.0            # 表示する範囲（今の位置からの半径）[m]
ALPHA = 215                # 不透明度（0〜255）
WDA_EXCLUDEFROMCAPTURE = 0x11
LANDMARK_COLORS = {"surface": (180, 200, 60), "nearby": (200, 120, 255)}   # BGR。報告と同じ（足元 = 青緑、近く = ピンク）


class MapOverlay:
    def __init__(self, vrchat_hwnd: int, size: int = SIZE, exclude_from_capture: bool = False):
        self.vrchat = vrchat_hwnd
        self.size = size
        self.exclude_from_capture = exclude_from_capture
        self.hwnd: int | None = None
        self.capture_excluded = False
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

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
            | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOPMOST)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, ALPHA, win32con.LWA_ALPHA)
        if self.exclude_from_capture:
            self.capture_excluded = bool(
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
        self.hwnd = hwnd
        return True

    def update(self, pose, explorer: explore.Explorer, trail: list[tuple[float, float]],
               falls: list[tuple[float, float, str]], lines: list[str],
               landmarks: list[dict] | None = None, goal: dict | None = None) -> None:
        size = self.size
        img = self.render(pose, explorer, trail, falls, lines, size, landmarks or [], goal)
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)
        if self._attach():
            import win32con
            import win32gui
            r = capture.client_rect(self.vrchat)
            if r.w > size + MARGIN and r.h > size + MARGIN:
                win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST,
                                      r.x + r.w - size - MARGIN, r.y + MARGIN, size, size,
                                      win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW)

    @staticmethod
    def render(pose, explorer: explore.Explorer, trail: list[tuple[float, float]],
               falls: list[tuple[float, float, str]], lines: list[str], size: int = SIZE,
               landmarks: list[dict] | None = None, goal: dict | None = None) -> np.ndarray:
        k = size / 300                                 # 文字や印の大きさの倍率（300px を基準）
        img = np.full((size, size, 3), (30, 30, 28), np.uint8)
        s = (size / 2) / RADIUS_M                      # px/m
        cx, cy = pose.x, pose.y

        def P(x: float, y: float) -> tuple[int, int]:
            return int(size / 2 + (x - cx) * s), int(size / 2 - (y - cy) * s)

        def cell_rect(c: tuple[int, int], color, fill=-1) -> None:
            x0, y0 = c[0] * explore.CELL, c[1] * explore.CELL
            if abs(x0 - cx) > RADIUS_M + 1 or abs(y0 - cy) > RADIUS_M + 1:
                return
            cv2.rectangle(img, P(x0, y0 + explore.CELL), P(x0 + explore.CELL, y0), color, fill)

        # 5m 格子
        for g in range(math.floor((cx - RADIUS_M) / 5), math.ceil((cx + RADIUS_M) / 5) + 1):
            x = P(g * 5, 0)[0]
            cv2.line(img, (x, 0), (x, size), (48, 48, 45), 1)
        for g in range(math.floor((cy - RADIUS_M) / 5), math.ceil((cy + RADIUS_M) / 5) + 1):
            y = P(0, g * 5)[1]
            cv2.line(img, (0, y), (size, y), (48, 48, 45), 1)
        for c in explorer.covered:
            cell_rect(c, (62, 84, 60))
        for c in explorer.cliffs:
            cell_rect(c, (30, 110, 200))
        for c in explorer.walls:
            cell_rect(c, (225, 225, 220))
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, P(*a), P(*b), (230, 190, 90), 1, cv2.LINE_AA)
        # RelateAnything で見つけたもの（確度の高いものだけ。古いほど薄く）
        for lm in landmarks or []:
            p = P(lm["x"], lm["y"])
            if not (0 <= p[0] < size and 0 <= p[1] < size):
                continue
            base = LANDMARK_COLORS[lm["kind"]]
            color = tuple(int(30 + (c - 30) * lm["fade"]) for c in base)   # 背景色へ寄せて薄くする
            if lm["kind"] == "surface":
                cv2.circle(img, p, max(3, int(4 * k)), color, -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(img, p, color, cv2.MARKER_DIAMOND, max(7, int(9 * k)), 2, cv2.LINE_AA)
            cv2.putText(img, lm["label"], (p[0] + int(6 * k), p[1] + int(4 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36 * k, color, 1, cv2.LINE_AA)
        for x, y, label in falls:
            p = P(x, y)
            cv2.drawMarker(img, p, (40, 140, 255), cv2.MARKER_TILTED_CROSS, int(12 * k), 2)
            cv2.putText(img, label, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (40, 140, 255), 1)
        sp = P(0, 0)
        cv2.drawMarker(img, sp, (240, 240, 240), cv2.MARKER_STAR, int(12 * k), 1)
        # 遠くの目標（未踏エリア）。範囲外なら縁に矢印で方角だけ示す
        if goal is not None:
            gp = P(goal["x"], goal["y"])
            color = (255, 255, 120)
            if 0 <= gp[0] < size and 0 <= gp[1] < size:
                cv2.circle(img, gp, int(9 * k), color, 2, cv2.LINE_AA)
                cv2.drawMarker(img, gp, color, cv2.MARKER_CROSS, int(8 * k), 1)
            else:
                ang = math.atan2(goal["x"] - pose.x, goal["y"] - pose.y)
                r0, r1 = size / 2 - 26 * k, size / 2 - 8 * k
                a = (int(size / 2 + r0 * math.sin(ang)), int(size / 2 - r0 * math.cos(ang)))
                b = (int(size / 2 + r1 * math.sin(ang)), int(size / 2 - r1 * math.cos(ang)))
                cv2.arrowedLine(img, a, b, color, 2, cv2.LINE_AA, tipLength=0.5)
            dist = math.hypot(goal["x"] - pose.x, goal["y"] - pose.y)
            cv2.putText(img, f"goal {dist:.0f}m", (7, size - int(10 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42 * k, color, 1, cv2.LINE_AA)
        # 今の位置と向き
        h = math.radians(pose.heading)
        c = (size // 2, size // 2)
        tip = (int(c[0] + 14 * k * math.sin(h)), int(c[1] - 14 * k * math.cos(h)))
        left = (int(c[0] + 7 * k * math.sin(h - 2.5)), int(c[1] - 7 * k * math.cos(h - 2.5)))
        right = (int(c[0] + 7 * k * math.sin(h + 2.5)), int(c[1] - 7 * k * math.cos(h + 2.5)))
        cv2.fillPoly(img, [np.array([tip, left, right], np.int32)], (0, 230, 255), cv2.LINE_AA)
        # 文字（英数字のみ）
        for i, line in enumerate(lines):
            y = int((16 + 15 * i) * k)
            cv2.putText(img, line, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * k, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(img, line, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * k, (235, 235, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "N", (size - int(18 * k), int(16 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (90, 90, 85), 1)
        return img

    def close(self) -> None:
        try:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass
