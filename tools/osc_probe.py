"""段階 1: VRChat との OSC 疎通を確かめる。

受信（9001）した値をすべて CSV に記録しつつ主要パラメータを画面に表示し、
キー操作で移動入力（9000）を送る。テストシーケンスで
「VelocityX/Z がワールド基準かアバター基準か」を判定する材料を取る。

    uv run tools/osc_probe.py
    uv run tools/osc_probe.py --raw        # 受信メッセージを 1 件ずつ全部表示
    uv run tools/osc_probe.py --sequence   # テストシーケンスを 1 回流して終了（キー操作不要）

PowerShell / Windows Terminal で実行すること（Git Bash の mintty ではキー入力が取れない）。

キー操作（入力は次のキーを押すまで保持される）
    w / s    前進 / 後退           a / d    左 / 右へ平行移動
    q / e    左 / 右へ旋回         j        ジャンプ
    r        走り 切替             space    全入力を止める
    1        テストシーケンス（前進→後退→横移動→旋回→前進）
    m        ログに目印を入れる    ESC / Ctrl+C  終了（全入力を 0 に戻す）
"""
from __future__ import annotations

import argparse
import csv
import msvcrt
import os
import sys
import threading
import time
from collections import defaultdict

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 画面に常時表示する組み込みパラメータ（届くかどうか自体が段階 1 の確認項目）
KEY_PARAMS = [
    "Grounded", "VelocityX", "VelocityY", "VelocityZ",
    "VelocityMagnitude", "AngularY", "Upright", "IsLocal", "AFK",
]

# 終了時に 0 に戻す入力。ボタン式は押しっぱなしになるので漏れなく並べる
AXES = ["/input/Vertical", "/input/Horizontal", "/input/LookHorizontal"]
BUTTONS = [
    "/input/MoveForward", "/input/MoveBackward", "/input/MoveLeft", "/input/MoveRight",
    "/input/LookLeft", "/input/LookRight", "/input/Jump", "/input/Run",
]


class Recorder:
    """受信値の最新状態と CSV ログ、区間ごとの集計を持つ。"""

    def __init__(self, path: str):
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.latest: dict[str, tuple[float, tuple]] = {}
        self.count = 0
        self.segment: str | None = None
        # 区間名 -> パラメータ名 -> 値のリスト
        self.seg_values: dict[str, dict[str, list[float]]] = {}
        self.seg_order: list[str] = []
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["t", "kind", "address", "values"])

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def write(self, kind: str, address: str, values) -> None:
        with self.lock:
            self._w.writerow([f"{self.now():.4f}", kind, address,
                              " ".join(str(v) for v in values)])

    def on_message(self, address: str, *args) -> None:
        t = self.now()
        with self.lock:
            self.latest[address] = (t, args)
            self.count += 1
            if self.segment is not None and args and isinstance(args[0], (int, float)):
                name = address.rsplit("/", 1)[-1]
                self.seg_values[self.segment].setdefault(name, []).append(float(args[0]))
        self.write("rx", address, args)

    def begin_segment(self, name: str) -> None:
        with self.lock:
            self.segment = name
            # 値は変化したときしか届かないので、区間開始時点の値を最初のサンプルにする
            self.seg_values[name] = {
                addr.rsplit("/", 1)[-1]: [float(args[0])]
                for addr, (_, args) in self.latest.items()
                if args and isinstance(args[0], (int, float))
            }
            self.seg_order.append(name)
        self.write("mark", "segment_begin", [name])

    def end_segment(self) -> None:
        with self.lock:
            name, self.segment = self.segment, None
        self.write("mark", "segment_end", [name])

    def summary(self) -> str:
        lines = ["区間ごとの平均（届いたサンプルのみ。n はサンプル数）"]
        with self.lock:
            for name in self.seg_order:
                vals = self.seg_values[name]
                cols = []
                for p in ("VelocityX", "VelocityZ", "VelocityY", "AngularY"):
                    v = vals.get(p, [])
                    cols.append(f"{p}={sum(v) / len(v):+.2f}(n={len(v)})" if v else f"{p}=--")
                lines.append(f"  {name:<14} " + "  ".join(cols))
        return "\n".join(lines)

    def close(self) -> None:
        with self.lock:
            self._f.close()


class Controller:
    """VRChat への入力。現在値を保持して定期的に送り直す。"""

    def __init__(self, client: SimpleUDPClient, rec: Recorder, turn: float):
        self.client = client
        self.rec = rec
        self.turn = turn
        self.lock = threading.Lock()
        self.axes = {a: 0.0 for a in AXES}
        self.run = 0

    def send(self, address: str, value) -> None:
        self.client.send_message(address, value)
        self.rec.write("tx", address, [value])

    def set(self, vertical=0.0, horizontal=0.0, look=0.0) -> None:
        with self.lock:
            self.axes["/input/Vertical"] = vertical
            self.axes["/input/Horizontal"] = horizontal
            self.axes["/input/LookHorizontal"] = look
        self.resend()

    def resend(self) -> None:
        with self.lock:
            items = list(self.axes.items())
        for a, v in items:
            self.send(a, float(v))

    def toggle_run(self) -> None:
        self.run ^= 1
        self.send("/input/Run", self.run)

    def jump(self) -> None:
        # ボタン式は 1 のあと必ず 0 を送る
        self.send("/input/Jump", 1)
        time.sleep(0.1)
        self.send("/input/Jump", 0)

    def release_all(self) -> None:
        with self.lock:
            for a in self.axes:
                self.axes[a] = 0.0
        self.run = 0
        for a in AXES:
            self.send(a, 0.0)
        for b in BUTTONS:
            self.send(b, 0)

    def describe(self) -> str:
        with self.lock:
            v = self.axes["/input/Vertical"]
            h = self.axes["/input/Horizontal"]
            l = self.axes["/input/LookHorizontal"]
        return f"Vertical={v:+.1f} Horizontal={h:+.1f} LookHorizontal={l:+.1f} Run={self.run}"


def run_sequence(ctl: Controller, rec: Recorder, stop: threading.Event,
                 move_sec: float, turn_sec: float) -> None:
    """速度の基準判定用。旋回の前後で同じ「前進」の VelocityX/Z が変わるかを見る。

    アバター基準なら前進は旋回前後とも VelocityZ>0・VelocityX≈0 になる。
    ワールド基準なら旋回後の前進で X/Z の比率が変わる。
    """
    steps = [
        ("forward_1", dict(vertical=1.0), move_sec),
        ("backward", dict(vertical=-1.0), move_sec),
        ("strafe_right", dict(horizontal=1.0), move_sec),
        ("turn_right", dict(look=ctl.turn), turn_sec),
        ("forward_2", dict(vertical=1.0), move_sec),
    ]
    rec.write("mark", "sequence_begin", [])
    try:
        for name, inputs, sec in steps:
            if stop.is_set():
                break
            ctl.set(**inputs)
            # 動き出しの加速を区間に含めない
            if stop.wait(0.4):
                break
            rec.begin_segment(name)
            stop.wait(max(0.0, sec - 0.4))
            rec.end_segment()
            ctl.set()
            if stop.wait(1.0):
                break
    finally:
        ctl.set()
        rec.write("mark", "sequence_end", [])


def render(rec: Recorder, ctl: Controller, status: str) -> str:
    t = rec.now()
    with rec.lock:
        latest = dict(rec.latest)
        count = rec.count
    by_name = {addr.rsplit("/", 1)[-1]: (addr, ts, args) for addr, (ts, args) in latest.items()}
    last_rx = max((ts for ts, _ in latest.values()), default=None)
    lines = [
        "VRCColliderProbe 段階 1: OSC 疎通確認   (ESC で終了)",
        f"受信 {count} 件 / アドレス {len(latest)} 種 / 最終受信 "
        + (f"{t - last_rx:.1f}s 前" if last_rx is not None else "なし（OSC が有効か確認）"),
        "",
    ]
    for name in KEY_PARAMS:
        if name in by_name:
            addr, ts, args = by_name[name]
            val = ", ".join(f"{a:+.3f}" if isinstance(a, float) else str(a) for a in args)
            lines.append(f"  {name:<18} {val:<16} ({t - ts:5.1f}s 前)  {addr}")
        else:
            lines.append(f"  {name:<18} 未受信")
    others = sorted(a for a in latest if a.rsplit("/", 1)[-1] not in KEY_PARAMS)
    lines += ["", f"その他のアドレス {len(others)} 種:"]
    lines += [f"  {a}" for a in others[:15]]
    if len(others) > 15:
        lines.append(f"  … ほか {len(others) - 15} 種（全部は CSV に記録）")
    lines += ["", f"入力: {ctl.describe()}", f"状態: {status}"]
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="VRChat の動いているホスト")
    ap.add_argument("--send-port", type=int, default=9000)
    ap.add_argument("--recv-port", type=int, default=9001)
    # 0.5 では旋回しなかった（実機確認）。1.0 で約 200°/s
    ap.add_argument("--turn", type=float, default=1.0, help="旋回時の LookHorizontal の大きさ")
    ap.add_argument("--move-sec", type=float, default=2.0, help="シーケンスの 1 区間の長さ")
    ap.add_argument("--turn-sec", type=float, default=1.0, help="シーケンスの旋回時間")
    ap.add_argument("--raw", action="store_true", help="受信メッセージを 1 件ずつ表示する")
    ap.add_argument("--sequence", action="store_true",
                    help="キー操作なしでテストシーケンスを 1 回流して終了する")
    args = ap.parse_args()

    os.makedirs(os.path.join(_ROOT, "logs"), exist_ok=True)
    log_path = os.path.join(_ROOT, "logs", time.strftime("osc_%Y%m%d_%H%M%S.csv"))
    rec = Recorder(log_path)

    disp = Dispatcher()
    if args.raw:
        def raw_handler(address, *a):
            rec.on_message(address, *a)
            print(f"{rec.now():8.3f}  {address}  {a}")
        disp.set_default_handler(raw_handler)
    else:
        disp.set_default_handler(rec.on_message)
    try:
        server = ThreadingOSCUDPServer(("127.0.0.1", args.recv_port), disp)
    except OSError as e:
        print(f"ポート {args.recv_port} を開けない: {e}\n"
              "（VRCOSC など他の OSC ツールが同じポートを使っていないか確認）")
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ctl = Controller(SimpleUDPClient(args.host, args.send_port), rec, args.turn)
    stop_seq = threading.Event()
    seq_thread: threading.Thread | None = None
    status = f"ログ: {log_path}"
    os.system("")  # conhost で ANSI エスケープを有効にする

    try:
        next_draw = next_resend = 0.0
        if args.sequence:
            run_sequence(ctl, rec, stop_seq, args.move_sec, args.turn_sec)
        while not args.sequence:
            now = time.perf_counter()
            # 軸入力は値が保持される想定だが、取りこぼし対策で定期的に送り直す
            if now >= next_resend:
                ctl.resend()
                next_resend = now + 0.1
            if not args.raw and now >= next_draw:
                sys.stdout.write("\x1b[H\x1b[J" + render(rec, ctl, status) + "\n")
                sys.stdout.flush()
                next_draw = now + 0.2

            if not msvcrt.kbhit():
                time.sleep(0.02)
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # 矢印キーなどの 2 バイト目を捨てる
                msvcrt.getwch()
                continue
            if ch in ("\x1b", "\x03"):
                break
            k = ch.lower()
            busy = seq_thread is not None and seq_thread.is_alive()
            if busy and k != " ":
                status = "シーケンス実行中（space で中断）"
                continue
            if k == "w":
                ctl.set(vertical=1.0)
            elif k == "s":
                ctl.set(vertical=-1.0)
            elif k == "a":
                ctl.set(horizontal=-1.0)
            elif k == "d":
                ctl.set(horizontal=1.0)
            elif k == "q":
                ctl.set(look=-args.turn)
            elif k == "e":
                ctl.set(look=args.turn)
            elif k == "j":
                ctl.jump()
            elif k == "r":
                ctl.toggle_run()
            elif k == " ":
                stop_seq.set()
                ctl.set()
                status = "停止"
            elif k == "m":
                rec.write("mark", "user", [])
                status = f"目印を記録 t={rec.now():.2f}"
            elif k == "1":
                stop_seq.clear()
                seq_thread = threading.Thread(
                    target=run_sequence,
                    args=(ctl, rec, stop_seq, args.move_sec, args.turn_sec), daemon=True)
                seq_thread.start()
                status = "シーケンス実行中（space で中断）"
            if args.raw:
                print(f"--- 入力: {ctl.describe()}")
    except KeyboardInterrupt:
        pass
    finally:
        # 止め忘れると歩き続けるので、何があっても全入力を 0 に戻す
        stop_seq.set()
        if seq_thread is not None:
            seq_thread.join(timeout=2.0)
        ctl.release_all()
        server.shutdown()
        rec.close()

    print("\n全入力を 0 に戻した。")
    if rec.seg_order:
        print(rec.summary())
        print("判定: forward_1 と forward_2 がどちらも VelocityZ>0・VelocityX≈0 ならアバター基準、"
              "X/Z の比率が変わっていればワールド基準")
    print(f"ログ: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
