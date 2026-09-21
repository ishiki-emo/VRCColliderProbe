"""自動歩行。前進し、壁に詰まったら旋回し、落下・リスポーン・挟まりを記録する。

段階 2（反射行動）に、段階 4 の探索（explore.py。未踏の多い向きを選ぶ）を載せたもの。

    uv run tools/walker.py                    # 5 分歩いて終了
    uv run tools/walker.py --strategy random  # 段階 2 のランダム旋回（比較用）
    uv run tools/walker.py --duration 60      # 秒数を指定
    uv run tools/walker.py --no-capture       # 画面キャプチャなし
    uv run tools/walker.py --not-at-spawn     # スポーン地点以外から歩き始める
    uv run tools/walker.py --overlay          # VRChat の画面の右上にミニマップを重ねる
    uv run tools/walker.py --relate-overlay   # RelateAnything の検出枠と状況説明を重ねる（GPU を使う）

ESC / Ctrl+C で止まる（全入力を 0 に戻す）。結果は runs/<日時>/ に出る。
終了時に mapper.py で地図（map.png）を、report.py で報告（report.html）を作る。地図はスポーン地点が原点なので、
起動前に VRChat のメニューからリスポーンしておくこと。

判定は SPEC.md 3 章の実測に基づく:
- 詰まり: 前進中に VelocityMagnitude が歩行速度より大きく落ちたまま続く
- リスポーン: 空中（Grounded=False）のまま VelocityY が大きな負から一気にほぼ 0 に戻る。
  着地も VelocityY が 0 になるが、その場合は直後に Grounded=True が届く
- 挟まり: 落下状態（VelocityY < -1.5）が 3 秒以上続く。コライダーの隙間に引っかかると
  落下状態のまま位置が変わらない。ジャンプと移動の組み合わせで抜け出しを試す
- 異常停止: /avatar/change（ワールド移動など）や、旋回を指示しても AngularY が届かない、
  挟まりから抜け出せない
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

try:
    import msvcrt
except ImportError:  # Windows 以外
    msvcrt = None

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture  # noqa: E402
import explore  # noqa: E402
import mapper  # noqa: E402
import overlay  # noqa: E402
import report  # noqa: E402
import semantic  # noqa: E402
import knowledge  # noqa: E402
import world  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AXES = ["/input/Vertical", "/input/Horizontal", "/input/LookHorizontal"]
BUTTONS = [
    "/input/MoveForward", "/input/MoveBackward", "/input/MoveLeft", "/input/MoveRight",
    "/input/LookLeft", "/input/LookRight", "/input/Jump", "/input/Run",
]

TURN_RATE = 200.0          # LookHorizontal=1 のときの旋回速度 [deg/s]（実測）
GRAVITY_STEP = 0.5         # 重力だけならこれ以上 1 サンプルで VelocityY は増えない
TELEPORT_JUMP = 3.0        # 空中でこれ以上 VelocityY が戻ったらテレポート候補
LANDING_WINDOW = 0.05      # VelocityY の急変から Grounded=True までの猶予 [s]
FALLING_VY = -1.5          # これより VelocityY が小さければ落下中とみなす
REPLAN_SEC = 3.0           # 直進中に向きを見直す間隔 [s]
LOOP_WINDOW = 60.0         # この秒数以内に
LOOP_REPEATS = 3           # 同じ場所（1.5m 以内）でこの回数詰まったら、ランダムに曲がってループを断つ
CLIMB_SEC = 1.0            # ジャンプしてから判定するまでの時間 [s]（滞空は約 0.8 秒）
CLIMB_PROGRESS_M = 0.8     # ジャンプ前の位置から向きの方向にこれ以上進めたら乗り越えた
GOAL_GIVEUP_SEC = 40.0     # 遠くの目標にこの秒数近づけなければあきらめる
PATROL_GIVEUP_SEC = 20.0   # 巡回の行き先にこの秒数近づけなければあきらめる（塗った範囲が壁にかかっていることがある）
GOAL_MAX_BUMPS = 3        # 目標を追う途中でこの回数壁に詰まったらあきらめる
UNWEDGE_GROUNDED_SEC = 0.5  # 挟まりから抜けたとみなすのに必要な、続けて接地している時間 [s]
TRAP_WINDOW = 40.0         # この秒数以内に
TRAP_REPEATS = 3           # 同じ場所でループの断ち切りがこの回数起きたら閉じ込め
TRAP_RADIUS_M = 2.0        # 「同じ場所」の半径 [m]。抜け出しの成功もこの距離で判定する
REPLAN_GAIN = 3.0          # 今の向きよりこれ以上点数が高く、今の向きが最良の半分未満なら曲がる

# 挟まり（落下状態のまま位置が変わらない）から抜け出す手順: (名前, 軸入力, 秒数)
# どれもジャンプと組み合わせる。本当に長く落下している最中でも空中なので害はない
ESCAPES = [
    ("back", dict(vertical=-1.0), 0.8),
    ("left", dict(horizontal=-1.0), 0.8),
    ("right", dict(horizontal=1.0), 0.8),
    ("forward", dict(vertical=1.0), 0.8),
    ("turn180", dict(look=1.0), 180.0 / 200.0),
    ("forward", dict(vertical=1.0), 0.8),
]


class Log:
    """CSV（全受信・送信）と events.jsonl（出来事）を書く。"""

    def __init__(self, run_dir: str):
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self._csv_f = open(os.path.join(run_dir, "osc.csv"), "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["t", "kind", "address", "values"])
        self._ev = open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8")
        self.events: list[dict] = []

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def row(self, kind: str, address: str, values) -> None:
        with self.lock:
            self._csv.writerow([f"{self.now():.4f}", kind, address,
                                " ".join(str(v) for v in values)])

    def event(self, kind: str, **fields) -> dict:
        ev = {"t": round(self.now(), 3), "kind": kind, **fields}
        with self.lock:
            self.events.append(ev)
            self._ev.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._ev.flush()
        print(f"[{ev['t']:8.2f}] {kind} " + " ".join(f"{k}={v}" for k, v in fields.items()))
        return ev

    def close(self) -> None:
        with self.lock:
            self._csv_f.close()
            self._ev.close()


class Avatar:
    """受信したパラメータの最新値と、落下・リスポーンの判定。

    値は変化時しか届かないので、最後に受けた値を保持する。
    """

    def __init__(self, log: Log):
        self.log = log
        self.lock = threading.Lock()
        self.grounded: bool | None = None
        self.vel_y = 0.0
        self.magnitude = 0.0
        self.angular_y = 0.0
        self.last_turning = -1.0   # AngularY が 0 以外で届いた最後の時刻
        self.last_rx = 0.0
        self.air_since: float | None = None
        self.air_min_vy = 0.0
        # VelocityY の急な戻り。直後に Grounded=True が来なければリスポーン
        self.pending_jump: tuple[float, float, float] | None = None  # (t, 前の値, 新しい値)
        self.avatar_changed = False
        # 本体ループが拾って処理する (急変の時刻, 落ち始めの推定位置)
        self.respawns: list[tuple[float, mapper.Pose]] = []
        # 位置の推定（mapper.py の再生と同じ計算をリアルタイムで行う）
        self.dr = mapper.DeadReckoner()
        # 挟まりから抜け出した瞬間も VelocityY が一気に戻るので、その間はリスポーン扱いしない
        self.suppress_respawn_until = -1.0

    def on_message(self, address: str, *args) -> None:
        self.log.row("rx", address, args)
        t = self.log.now()
        if address == "/avatar/change":
            with self.lock:
                self.avatar_changed = True
            return
        if not args:
            return
        name = address.rsplit("/", 1)[-1]
        v = args[0]
        with self.lock:
            self.last_rx = t
            self.dr.feed(t, name, v)
            if name == "Grounded":
                self._on_grounded(bool(v), t)
            elif name == "VelocityY":
                prev, self.vel_y = self.vel_y, float(v)
                if self.air_since is not None:
                    self.air_min_vy = min(self.air_min_vy, self.vel_y)
                    if self.vel_y - prev > TELEPORT_JUMP and self.vel_y <= GRAVITY_STEP:
                        self.pending_jump = (t, prev, self.vel_y)
                        # 着地では VelocityY がちょうど 0 になる。テレポート直後は -0.16 のように
                        # 重力 1 サンプル分の値になるので、0 でなければ着地を待たずに確定する
                        # （スポーン地点が床の高さだと、テレポート直後すぐ着地することがある）
                        if self.vel_y != 0.0:
                            self._confirm_pending(t)
            elif name == "VelocityMagnitude":
                self.magnitude = float(v)
            elif name == "AngularY":
                self.angular_y = float(v)
                if self.angular_y != 0.0:
                    self.last_turning = t

    def _on_grounded(self, g: bool, t: float) -> None:
        if g and self.grounded is False:
            # 着地。VelocityY の急変はこの着地によるものだった
            if self.pending_jump is not None and t - self.pending_jump[0] <= LANDING_WINDOW:
                self.pending_jump = None
            if self.air_since is not None:
                dur = t - self.air_since
                # 段差では 17ms 程度の False/True が何度か続くので、短い空中は記録しない
                if dur >= 0.25:
                    self.log.event("drop", airtime=round(dur, 2), min_vy=round(self.air_min_vy, 2))
            self.air_since = None
        elif not g and self.grounded is not False:
            self.air_since = t
            self.air_min_vy = 0.0
        self.grounded = g

    def poll(self) -> None:
        """猶予を過ぎても着地しなかった急変をリスポーンとして確定する。"""
        t = self.log.now()
        with self.lock:
            pj = self.pending_jump
            if pj is None or t - pj[0] <= LANDING_WINDOW:
                return
            self._confirm_pending(t)

    def _confirm_pending(self, t: float) -> None:
        """pending_jump をリスポーン（挟まりからの脱出中なら wedge_released）として確定する。
        self.lock を持った状態で呼ぶ。"""
        pj, self.pending_jump = self.pending_jump, None
        air = (pj[0] - self.air_since) if self.air_since is not None else None
        released = pj[0] <= self.suppress_respawn_until
        if not released:
            self.respawns.append((pj[0], self.dr.respawn(pj[0])))
        # テレポート後もスポーン地点までは空中なので、落下の計測をやり直す
        self.air_since = t
        self.air_min_vy = 0.0
        if released:
            self.log.event("wedge_released", at=round(pj[0], 3), vy_before=round(pj[1], 2))
            return
        self.log.event("respawn", at=round(pj[0], 3), vy_before=round(pj[1], 2),
                       vy_after=round(pj[2], 2), fall_time=round(air, 2) if air else None)

    def snapshot(self) -> dict:
        with self.lock:
            # 直進中はメッセージが来ないので、今の時刻まで位置の推定を進める
            self.dr.advance(self.log.now())
            return dict(grounded=self.grounded, vel_y=self.vel_y, magnitude=self.magnitude,
                        angular_y=self.angular_y, last_turning=self.last_turning,
                        last_rx=self.last_rx, pose=self.dr.pose.copy(),
                        vx=self.dr.vx, vz=self.dr.vz,
                        avatar_changed=self.avatar_changed)


class Inputs:
    def __init__(self, client: SimpleUDPClient, log: Log):
        self.client = client
        self.log = log
        self.axes = {a: 0.0 for a in AXES}

    def set(self, vertical=0.0, horizontal=0.0, look=0.0) -> None:
        self.axes = {"/input/Vertical": vertical, "/input/Horizontal": horizontal,
                     "/input/LookHorizontal": look}
        self.resend()

    def resend(self) -> None:
        for a, v in self.axes.items():
            self.client.send_message(a, float(v))
        self.log.row("tx", "axes", list(self.axes.values()))

    def jump(self) -> None:
        # ボタン式は 1 のあと必ず 0 を送る
        self.client.send_message("/input/Jump", 1)
        time.sleep(0.1)
        self.client.send_message("/input/Jump", 0)
        self.log.row("tx", "/input/Jump", [1, 0])

    def release_all(self) -> None:
        self.axes = {a: 0.0 for a in AXES}
        for a in AXES:
            self.client.send_message(a, 0.0)
        for b in BUTTONS:
            self.client.send_message(b, 0)
        self.log.row("tx", "release_all", [])


class FrameBuffer:
    """直近 N 秒の画面を JPEG で保持する（リスポーン時に書き出す証拠）。

    log_dir を渡すと、log_every 秒に 1 枚をそこへ保存し続ける（semantic.py が「どこに何があるか」に使う）。
    """

    def __init__(self, hwnd: int, seconds: float, fps: float, log: Log,
                 log_dir: str | None = None, log_every: float = 1.0):
        self.log_dir = log_dir
        self.log_every = log_every
        self._last_logged = -1e9
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.cap = capture.WindowCapture(hwnd)
        self.buf: collections.deque = collections.deque(maxlen=max(1, int(seconds * fps)))
        self.interval = 1.0 / fps
        self.log = log
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        import cv2
        while not self.stop.is_set():
            t = self.log.now()
            img = self.cap.read()
            if img is not None:
                # 保存容量を抑えるため縮小して JPEG で持つ
                h, w = img.shape[:2]
                if w > 960:
                    img = cv2.resize(img, (960, int(h * 960 / w)))
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.buf.append((t, jpg))
                    if self.log_dir and t - self._last_logged >= self.log_every:
                        self._last_logged = t
                        jpg.tofile(os.path.join(self.log_dir, f"t{t:09.3f}.jpg"))
            self.stop.wait(self.interval)

    def motion(self) -> float | None:
        """直近 2 フレームの差（輝度の平均絶対差、0〜255）。

        実測: 挟まって静止 0.5 前後、何もない空間への落下 1.4〜2.6、歩行 5〜60。
        落下中も背景が一様だと小さいので、判定には使わず記録だけする。
        """
        import cv2
        import numpy as np
        with self.lock:
            if len(self.buf) < 2:
                return None
            a, b = self.buf[-2][1], self.buf[-1][1]
        ga, gb = (cv2.resize(cv2.imdecode(j, cv2.IMREAD_GRAYSCALE), (160, 90)).astype(np.float32)
                  for j in (a, b))
        return float(np.mean(np.abs(ga - gb)))

    def dump(self, out_dir: str, until: float) -> int:
        os.makedirs(out_dir, exist_ok=True)
        with self.lock:
            frames = [(t, j) for t, j in self.buf if t <= until]
        for t, jpg in frames:
            jpg.tofile(os.path.join(out_dir, f"t{t:09.3f}.jpg"))
        return len(frames)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)


def read_live_obs(path: str, pos: int, pose_history, landmarks: "semantic.LiveLandmarks") -> int:
    """relate_overlay.py が追記した観測を pos から読み、撮影時刻のアバターの位置に置く。次の読み出し位置を返す。

    推論に 0.1〜0.3 秒かかるので、今の位置ではなく撮影時刻に最も近い位置を使う。
    """
    if not os.path.exists(path):
        return pos
    with open(path, "rb") as f:
        f.seek(pos)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1          # 書きかけの行は次回に回す
    for line in chunk[:end].splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        near = min(pose_history, key=lambda p: abs(p[0] - rec["epoch"]), default=None)
        if near is None or abs(near[0] - rec["epoch"]) > 0.5:
            continue
        for hit in rec["hits"]:
            landmarks.add(hit, near[1], near[2], near[3], rec["epoch"])
    return pos + end


NAVI_ARRIVE_M = 2.0        # ナビの行き先にこの距離まで近づいたら到着 [m]


def navi_main(args) -> int:
    """ナビモード。人が自分で歩き、ツールは位置の推定と地図の表示だけをする。

    OSC の速度や旋回の値は、誰が操作していても同じように届くので、位置の推定はそのまま使える。
    - VRChat に入力は一切送らない（終了時の「全入力を 0 に戻す」も送らない。操作の邪魔をしないため）
    - ワールドの記録（累計の地図）を読み込んで表示するだけで、この走行は記録に加えない
      （人の移動にはメニューからのワープやポータルが混ざりうる）
    - 行き先は --dest-file の JSON。GUI が書き換えると、次の読み込みで追従する
    - 今の位置を <run_dir>/pose.json に書く（GUI の地図に出すため）
    """
    run_dir = os.path.join(_ROOT, "runs", time.strftime("navi_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log = Log(run_dir)
    avatar = Avatar(log)
    disp = Dispatcher()
    disp.set_default_handler(avatar.on_message)
    try:
        server = ThreadingOSCUDPServer(("127.0.0.1", args.recv_port), disp)
    except OSError as e:
        print(f"ポート {args.recv_port} を開けない: {e}\n（osc_probe.py などが起動したままになっていないか確認）")
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cur_world = ({"id": args.world_id, "name": args.world_id} if args.world_id else world.current_world())
    explorer = explore.Explorer()
    landmarks: list[dict] = []
    if cur_world:
        agg = knowledge.aggregate(knowledge.load(cur_world["id"]))
        knowledge.apply_to_explorer(agg, explorer)
        # 目印になるもの（何度も見えたもの）だけを地図に出す。床・地面はどこにでもあるので出さない
        landmarks = [{**m, "fade": 1.0} for m in agg["landmarks"]
                     if m["frames"] >= 3 and m["label"] not in semantic.GENERIC_LABELS][:80]
        print(f"{cur_world['name']} の地図: 走行 {agg['runs']} 回 / 踏破 {len(agg['visited']) * explore.CELL ** 2:.0f} m2"
              f" / 目印 {len(landmarks)}")
        if not agg["visited"]:
            print("このワールドの地図はまだありません（自動歩行で地図を作ってから使ってください）")
    else:
        print("VRChat のログからワールドが分からないので、地図なしで位置だけを表示する")
    aligned = not args.not_at_spawn
    log.event("start", run_dir=run_dir, mode="navi", duration=args.duration, at_spawn=aligned,
              world_id=cur_world["id"] if cur_world else None,
              world_name=cur_world["name"] if cur_world else None, epoch0=round(time.time() - log.now(), 3))
    if not aligned:
        print("スポーン地点以外から始めたので、リスポーンするまで位置は地図と合っていません")

    w = capture.find_vrchat_window()
    map_overlay = None
    if w is None:
        print("VRChat のウィンドウが見つからないので、ミニマップは出さない（GUI の地図には位置を出す）")
    else:
        capture.enable_dpi_awareness()
        map_overlay = overlay.MapOverlay(w[0], size=args.overlay_size,
                                         exclude_from_capture=args.hide_overlay_from_recording)
    print("ESC / q / Ctrl+C で終了（入力は送っていないので、止めても VRChat の操作には影響しない）")

    dest: dict | None = None
    dest_mtime = 0.0
    trail: list[tuple[float, float]] = []
    next_draw = next_plan = next_pose = 0.0
    arrived = False
    stop_reason = "duration"
    try:
        while True:
            t = log.now()
            if t >= args.duration:
                break
            if key_pressed_quit():
                stop_reason = "user"
                break
            if args.stop_file and os.path.exists(args.stop_file):
                stop_reason = "user"
                break
            snap = avatar.snapshot()
            if snap["avatar_changed"]:
                # ワールドを移ったら、この地図はもう使えない
                stop_reason = "avatar_change"
                log.event("abort", reason="/avatar/change を受信（ワールド移動の可能性）")
                break
            avatar.poll()
            with avatar.lock:
                respawns, avatar.respawns = avatar.respawns, []
            for _ in respawns:
                trail.clear()
                if not aligned:
                    aligned = True
                    log.event("map_aligned")
                    print("リスポーンしたので、ここから位置が地図と合う")
            pose = snap["pose"]
            # 行き先（GUI が書き換える）
            if args.dest_file and os.path.exists(args.dest_file) and os.path.getmtime(args.dest_file) != dest_mtime:
                dest_mtime = os.path.getmtime(args.dest_file)
                try:
                    with open(args.dest_file, encoding="utf-8") as f:
                        d = json.load(f)
                    dest = {"x": float(d["x"]), "y": float(d["y"]), "label": str(d.get("label", ""))} \
                        if d and "x" in d else None
                except (ValueError, KeyError, OSError):
                    dest = None
                arrived, next_plan = False, 0.0
                log.event("dest_set", **(dest or {"cleared": True}))
            if dest is not None and t >= next_plan:
                next_plan = t + 1.0
                dest["path"] = explorer.plan_path(pose.x, pose.y, dest) if aligned else None
                d = math.hypot(dest["x"] - pose.x, dest["y"] - pose.y)
                if d <= NAVI_ARRIVE_M and not arrived:
                    arrived = True
                    log.event("dest_arrived", label=dest["label"], x=round(dest["x"], 1), y=round(dest["y"], 1))
            if not trail or math.hypot(pose.x - trail[-1][0], pose.y - trail[-1][1]) > 0.3:
                trail.append((pose.x, pose.y))
                del trail[:-600]
            if t >= next_pose:
                next_pose = t + 0.5
                write_json_atomic(os.path.join(run_dir, "pose.json"), {
                    "t": round(t, 1), "x": round(pose.x, 2), "y": round(pose.y, 2), "heading": round(pose.heading, 1),
                    "aligned": aligned, "dest": {k: v for k, v in (dest or {}).items() if k != "path"} or None,
                    "dest_dist": round(math.hypot(dest["x"] - pose.x, dest["y"] - pose.y), 1) if dest else None,
                    "path": [[round(px, 1), round(py, 1)] for px, py in (dest or {}).get("path") or []],
                    "arrived": arrived})
            if map_overlay is not None and t >= next_draw:
                next_draw = t + 0.2
                lines = ["NAVI" + ("" if aligned else "  (respawn to align map)")]
                if dest is not None:
                    dd = math.hypot(dest["x"] - pose.x, dest["y"] - pose.y)
                    label = dest["label"].encode("ascii", "ignore").decode() or "dest"
                    lines.append(f"{label}: {'ARRIVED' if arrived else f'{dd:.0f} m'}"
                                 + ("" if dest.get("path") or not aligned else "  (no known route)"))
                map_overlay.update(pose, explorer, trail, [], lines, landmarks=landmarks,
                                   goal=dest if aligned else None)
            time.sleep(0.02)
    except KeyboardInterrupt:
        stop_reason = "user"
    finally:
        # 入力は送っていないので、release_all もしない
        if map_overlay is not None:
            map_overlay.close()
        server.shutdown()
    log.event("end", stop_reason=stop_reason, elapsed=round(log.now(), 1), mode="navi")
    log.close()
    print(f"ナビを終了した（{stop_reason}）。記録: {run_dir}")
    return 0


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def key_pressed_quit() -> bool:
    if msvcrt is None:
        return False
    try:
        while msvcrt.kbhit():
            if msvcrt.getwch() in ("\x1b", "\x03", "q"):
                return True
    except Exception:
        pass
    return False


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--send-port", type=int, default=9000)
    ap.add_argument("--recv-port", type=int, default=9001)
    ap.add_argument("--duration", type=float, default=300.0, help="歩く秒数")
    ap.add_argument("--walk-speed", type=float, default=3.0, help="通常の歩行速度 [m/s]（実測 3.0）")
    ap.add_argument("--stuck-ratio", type=float, default=0.4,
                    help="VelocityMagnitude が歩行速度のこの割合を下回ったら詰まり候補")
    ap.add_argument("--stuck-sec", type=float, default=0.25, help="詰まり候補がこの秒数続いたら旋回")
    ap.add_argument("--stop-file", default="",
                    help="このファイルができたら止まる（GUI から止めるため。全入力を 0 に戻して報告まで作る）")
    ap.add_argument("--world-id", default="",
                    help="ワールドの記録に使う ID を指定する（既定は VRChat のログから。偽ワールドのテストでは sim_… を使う）")
    ap.add_argument("--navi", action="store_true",
                    help="ナビモード: 入力は一切送らず、自分で歩く位置を推定して、累計の地図と行き先への経路をミニマップに出す")
    ap.add_argument("--dest-file", default="",
                    help="ナビの行き先（{\"x\", \"y\", \"label\"} の JSON）。GUI が書き換えると追従する")
    ap.add_argument("--patrol", default="",
                    help="巡回範囲のファイル（GUI で塗って保存した worlds/<ID>/patrols/<名前>.json）。その中を回り続ける")
    ap.add_argument("--no-goal", action="store_true",
                    help="遠くの未踏エリアを目標にしない（近くの判断だけで探索する。比較用）")
    ap.add_argument("--fresh", action="store_true",
                    help="このワールドの過去の記録を読み込まず、この走行も記録に加えない（スポーン地点が複数あるワールドなど）")
    ap.add_argument("--no-climb", action="store_true",
                    help="壁に詰まったときにジャンプで乗り越えを試さない")
    ap.add_argument("--wedge-sec", type=float, default=3.0,
                    help="落下状態がこの秒数続いたら挟まりとみなして抜け出しを試す")
    ap.add_argument("--strategy", choices=("frontier", "random"), default="frontier",
                    help="frontier: 未踏の格子が多い向きを選ぶ / random: ランダムに旋回する（段階 2）")
    ap.add_argument("--no-capture", action="store_true", help="画面キャプチャをしない")
    ap.add_argument("--frame-log", type=float, default=1.0,
                    help="この秒数ごとに画面を frames/ に保存する（「どこに何があるか」用。0 で保存しない）")
    ap.add_argument("--overlay", action="store_true",
                    help="VRChat の画面の右上に、ルートと作りかけの地図を重ねて表示する")
    ap.add_argument("--overlay-size", type=int, default=overlay.SIZE, help="ミニマップの一辺 [px]")
    ap.add_argument("--relate-overlay", action="store_true",
                    help="RelateAnything の検出枠と状況説明を VRChat の画面に重ねる（別プロセス・GPU を使う）")
    ap.add_argument("--hide-overlay-from-recording", action="store_true",
                    help="オーバーレイを OBS や Win+Shift+R の録画に写さない（既定では写る）")
    ap.add_argument("--landmark-ttl", type=float, default=120.0,
                    help="RelateAnything で見つけたものをミニマップに残す秒数（最後に見えてから）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--not-at-spawn", action="store_true",
                    help="スポーン地点以外から歩き始める（最初のリスポーンまでは地図に載せない）")
    args = ap.parse_args()
    if args.navi:
        return navi_main(args)
    rng = random.Random(args.seed)

    run_dir = os.path.join(_ROOT, "runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log = Log(run_dir)
    avatar = Avatar(log)

    disp = Dispatcher()
    disp.set_default_handler(avatar.on_message)
    try:
        server = ThreadingOSCUDPServer(("127.0.0.1", args.recv_port), disp)
    except OSError as e:
        print(f"ポート {args.recv_port} を開けない: {e}\n（osc_probe.py などが起動したままになっていないか確認）")
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    inputs = Inputs(SimpleUDPClient(args.host, args.send_port), log)

    frames: FrameBuffer | None = None
    if not args.no_capture:
        capture.enable_dpi_awareness()
        w = capture.find_vrchat_window()
        if w is None:
            print("VRChat のウィンドウが見つからないので画面キャプチャなしで続ける")
        else:
            frames = FrameBuffer(w[0], seconds=4.0, fps=5.0, log=log,
                                 log_dir=os.path.join(run_dir, "frames") if args.frame_log > 0 else None,
                                 log_every=args.frame_log)
    map_overlay: overlay.MapOverlay | None = None
    if args.overlay:
        capture.enable_dpi_awareness()
        w = capture.find_vrchat_window()
        if w is None:
            print("VRChat のウィンドウが見つからないのでオーバーレイなしで続ける")
        else:
            map_overlay = overlay.MapOverlay(w[0], size=args.overlay_size,
                                             exclude_from_capture=args.hide_overlay_from_recording)
    relate_proc: subprocess.Popen | None = None
    # RelateAnything のオーバーレイが見つけた「足元／近くのもの」。確度の高いものをミニマップに残す
    live_obs_path = os.path.join(run_dir, "live_obs.jsonl")
    live_obs_pos = 0
    live_landmarks = semantic.LiveLandmarks(ttl=args.landmark_ttl)
    pose_history: collections.deque = collections.deque(maxlen=600)   # (エポック秒, x, y, 向き) 10 回/秒
    next_pose_log = 0.0
    if args.relate_overlay:
        # RelateAnything は torch 入りの別環境なので別プロセスで動かす。こちらが終わると自分で終わる
        if os.path.exists(report.RELATE_PYTHON):
            relate_proc = subprocess.Popen(
                [report.RELATE_PYTHON, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "relate_overlay.py"),
                 "--parent-pid", str(os.getpid()), "--obs-out", live_obs_path]
                + (["--hide-from-recording"] if args.hide_overlay_from_recording else []),
                # RelateAnything のフォルダで動かす（検出器が重みを作業フォルダにダウンロードするため）
                cwd=report.RELATE_ROOT,
                stdout=subprocess.DEVNULL, stderr=open(os.path.join(run_dir, "relate_overlay.log"), "w"))
            log.event("relate_overlay_started", pid=relate_proc.pid)
        else:
            print(f"RelateAnything の環境が見つからないのでオーバーレイなしで続ける: {report.RELATE_PYTHON}")
    overlay_trail: list[tuple[float, float]] = []   # 今の区間（リスポーンまで）のルート
    overlay_marks: list[tuple[float, float, str]] = []
    next_overlay = 0.0

    # 今いるワールド（VRChat のログから）。ワールドごとの記録（knowledge.py）に使う。
    # --fresh のときは記録を読み込まず、この走行も記録に加えない（スポーン地点が複数あるワールドなど）
    cur_world = ({"id": args.world_id, "name": args.world_id} if args.world_id else world.current_world())
    use_knowledge = not args.fresh
    patrol: explore.Patrol | None = None
    patrol_name = ""
    if args.patrol:
        patrol_name, patrol_cells = explore.load_patrol(args.patrol)
        patrol = explore.Patrol(patrol_cells)
        print(f"巡回範囲「{patrol_name}」: {len(patrol.blocks)} ブロック（{len(patrol_cells) * explore.CELL ** 2:.0f} m2）")
    # epoch0: ログの時刻 0 に当たるエポック秒。別プロセス（relate_overlay.py）の記録と時刻を合わせるのに使う
    log.event("start", run_dir=run_dir, duration=args.duration, seed=args.seed,
              at_spawn=not args.not_at_spawn, strategy=args.strategy,
              epoch0=round(time.time() - log.now(), 3),
              world_id=cur_world["id"] if cur_world else None,
              world_name=cur_world["name"] if cur_world else None,
              use_knowledge=use_knowledge, patrol=patrol_name or None,
              patrol_blocks=len(patrol.blocks) if patrol else None)
    print("ESC / q / Ctrl+C で停止")

    stuck_level = args.walk_speed * args.stuck_ratio
    state, state_until = "forward", 0.0
    forward_since = log.now()
    low_since: float | None = None
    last_turn_end = -99.0
    turn_checked = True
    turn_started = 0.0
    missing_turns = 0
    fall_since: float | None = None
    grounded_since: float | None = None
    escape_step = 0
    escape_kind = "fall"            # fall: 落下状態のまま挟まった / trap: 接地したまま閉じ込められた
    escape_origin = (0.0, 0.0)     # 挟まった位置と向き（抜けた後、そこを崖として避ける）
    escape_heading = 0.0
    climb_origin = (0.0, 0.0, 0.0, 0.0)   # ジャンプで乗り越えを試した位置・向き・高さ
    loop_breaks: collections.deque = collections.deque(maxlen=20)   # (時刻, x, y)
    trapped = False
    heading = 0.0          # スポーン時の向きを 0 とした推定ヨー角 [deg]
    explorer = explore.Explorer()
    # スポーン地点以外から始めたときは、最初のリスポーンまで地図の座標が合っていない
    aligned = not args.not_at_spawn
    if cur_world and use_knowledge:
        # 過去の走行の記録を読み込む: 歩いていない場所を優先し、過去に落ちた・挟まった場所は最初から避ける。
        # スポーン地点以外から始めたときは、最初のリスポーン（座標が合った時点）で読み込む
        agg = knowledge.aggregate(knowledge.load(cur_world["id"]))
        if agg["runs"]:
            print(f"{cur_world['name']} の過去の記録を読み込んだ: 走行 {agg['runs']} 回 / "
                  f"累計の踏破 {len(agg['visited']) * explore.CELL ** 2:.0f} m2 / 崖 {len(agg['cliffs'])} マス")
            if aligned:
                knowledge.apply_to_explorer(agg, explorer)
        prior = agg
    else:
        prior = None
    last_plan = 0.0
    turn_avoid = 0.0       # 次の旋回で避ける、正面から左右の角度 [deg]
    planned: float | None = None   # 再計画で決めた向き
    stuck_log: collections.deque = collections.deque(maxlen=50)   # (時刻, x, y)
    # 遠くの目標（歩いた帯に接する未踏のまとまり）。近くの判断は 8m 先までしか見ないので、
    # 周りを歩き尽くすと実質ランダムになっていた（記録を読み込んだ走行で、新しい場所の割合が改善しなかった）
    goal: dict | None = None
    # 巡回（--patrol）: GUI で塗った範囲の中を回り続ける。未踏の格子は数えず（explore_w=0）、行き先へ向かうことに専念する
    explore_w = 0.0 if patrol is not None else 1.0
    next_patrol_touch = 0.0
    goal_track = {"best": math.inf, "since": 0.0, "avoid": [], "bumps": 0}

    def steer_point(g: dict | None, p) -> dict | None:
        """向かう方角の目安。歩いた場所だけを通る経路が見つかれば、その 3m 先。なければ目標そのもの。

        目標の方角へまっすぐ向かうと、途中の壁や袋小路にぶつかって初めて分かり、壁越しに一直線に
        向かおうとし続けた（実機で、詰まり 9 回のうち 8 回が目標を追っている最中）。
        """
        if g is None:
            return None
        path = explorer.plan_path(p.x, p.y, g)
        g["path"] = path
        if path is None or len(path) < 2:
            return g
        return explore.Explorer.waypoint(path, p.x, p.y)

    def update_patrol_goal(g: dict | None, now: float, p) -> dict | None:
        """巡回中の行き先。最後に訪れてから一番時間が経ったブロックへ。"""
        if not aligned:
            return None   # スポーン地点以外から始めたときは、最初のリスポーンまで範囲の座標が合っていない
        if g is not None:
            d = math.hypot(g["x"] - p.x, g["y"] - p.y)
            if d <= explore.PATROL_TOUCH_M:
                counts["patrol_reached"] += 1
                log.event("patrol_reached", x=round(g["x"], 1), y=round(g["y"], 1),
                          coverage=round(patrol.coverage(), 3))
                g = None
            elif d < goal_track["best"] - 1.0:
                goal_track["best"], goal_track["since"] = d, now
            elif now - goal_track["since"] > PATROL_GIVEUP_SEC:
                counts["goal_abandoned"] += 1
                log.event("goal_abandoned", x=round(g["x"], 1), y=round(g["y"], 1), dist=round(d, 1))
                goal_track["avoid"].append((g["x"], g["y"]))
                g = None
        if g is None:
            g = patrol.next_target(p.x, p.y, now, explorer, goal_track["avoid"])
            if g is not None:
                goal_track["best"], goal_track["since"], goal_track["bumps"] = g["dist"], now, 0
                log.event("patrol_goal", x=round(g["x"], 1), y=round(g["y"], 1), dist=round(g["dist"], 1))
        return g

    def update_goal(g: dict | None, now: float, p) -> dict | None:
        if patrol is not None:
            return update_patrol_goal(g, now, p)
        if args.strategy != "frontier" or args.no_goal:
            return None
        if g is not None:
            d = math.hypot(g["x"] - p.x, g["y"] - p.y)
            if d <= explore.TARGET_REACHED_M:
                counts["goal_reached"] += 1
                log.event("goal_reached", x=round(g["x"], 1), y=round(g["y"], 1))
                g = None
            elif d < goal_track["best"] - 1.0:
                goal_track["best"], goal_track["since"] = d, now
            elif now - goal_track["since"] > GOAL_GIVEUP_SEC:
                # 近づけない（壁の向こう、崖の先など）。この付近はもう目標にしない
                counts["goal_abandoned"] += 1
                log.event("goal_abandoned", x=round(g["x"], 1), y=round(g["y"], 1), dist=round(d, 1))
                goal_track["avoid"].append((g["x"], g["y"]))
                g = None
        if g is None:
            cands = explorer.frontier_targets(p.x, p.y, goal_track["avoid"])
            if cands:
                g = cands[0]
                goal_track["best"], goal_track["since"] = math.hypot(g["x"] - p.x, g["y"] - p.y), now
                goal_track["bumps"] = 0
                log.event("goal_set", x=round(g["x"], 1), y=round(g["y"], 1), size=g["size"],
                          dist=round(g["dist"], 1), candidates=len(cands))
        return g
    counts = collections.Counter()
    stop_reason = "duration"
    inputs.set(vertical=1.0)
    prev_t = log.now()

    try:
        while True:
            t = log.now()
            dt, prev_t = t - prev_t, t
            if t >= args.duration:
                break
            if key_pressed_quit():
                stop_reason = "user"
                break
            # GUI（gui.py）から起動したときはキー入力を送れないので、停止ファイルができたら止まる
            if args.stop_file and int(t * 4) != int((t - dt) * 4) and os.path.exists(args.stop_file):
                stop_reason = "user"
                log.event("stop_requested", by="stop_file")
                break
            snap = avatar.snapshot()
            pose = snap["pose"]
            heading = pose.heading
            if snap["grounded"] and snap["vel_y"] >= FALLING_VY:
                explorer.mark_visited(pose.x, pose.y)

            if snap["avatar_changed"]:
                # ワールド移動（ポータル）やアバター変更。これ以上歩かせない
                stop_reason = "avatar_change"
                log.event("abort", reason="/avatar/change を受信（ワールド移動の可能性）")
                break

            avatar.poll()
            with avatar.lock:
                respawns, avatar.respawns = avatar.respawns, []
            for at, fall in respawns:
                counts["respawn"] += 1
                if frames is not None:
                    n = frames.dump(os.path.join(run_dir, f"respawn_{counts['respawn']:02d}"), until=at)
                    log.event("frames_saved", respawn=counts["respawn"], count=n)
                if aligned:
                    # 外周の可能性が高いので、同じ場所から何度も落ちないよう崖として避ける
                    explorer.mark_cliff(fall.x, fall.y, fall.heading)
                    overlay_marks.append((fall.x, fall.y, f"R{counts['respawn']}"))
                else:
                    explorer.clear()
                    overlay_marks.clear()
                    live_landmarks.clear()
                    aligned = True
                    log.event("map_aligned")
                    if prior is not None:
                        knowledge.apply_to_explorer(prior, explorer)
                overlay_trail.clear()
                # 遠くの目標はそのまま（座標はスポーン地点基準で変わらない）。スポーン地点から測り直す
                goal_track["best"], goal_track["since"] = math.inf, t
                # 位置が戻ったので少し待ってから歩き直す
                inputs.set()
                state, state_until = "pause", t + 1.0

            # 旋回を指示したのに AngularY が来ない = ロード画面やメニューなど、操作が効いていない
            if not turn_checked and t - turn_started > 0.3:
                turn_checked = True
                if snap["last_turning"] < turn_started:
                    missing_turns += 1
                    log.event("turn_not_applied", count=missing_turns)
                    if missing_turns >= 2:
                        stop_reason = "no_response"
                        log.event("abort", reason="旋回が 2 回続けて反映されない")
                        break
                else:
                    missing_turns = 0

            # 挟まり: 落下状態が続いているのにリスポーンも着地もしない。
            # コライダーの隙間に引っかかると VelocityY が重力どおり増え続けたまま位置が変わらない
            # （実測: 100 秒で -992 m/s）。このワールドの本当の落下は 1 秒前後でリスポーンした
            if snap["vel_y"] < FALLING_VY:
                fall_since = fall_since if fall_since is not None else t
            else:
                fall_since = None
            if snap["grounded"]:
                grounded_since = grounded_since if grounded_since is not None else t
            else:
                grounded_since = None
            if state == "escape":
                with avatar.lock:
                    avatar.suppress_respawn_until = t + 1.0
                if escape_kind == "fall":
                    # ジャンプ直後は挟まったままでも VelocityY が正になり、一瞬だけ接地することもある
                    # （実機で 50ms だけ Grounded=True になって、また落下状態に戻った）。続けて接地するまで待つ
                    freed = (fall_since is None and grounded_since is not None
                             and t - grounded_since >= UNWEDGE_GROUNDED_SEC)
                else:
                    # 閉じ込め: 閉じ込められた場所から離れられたら抜けた
                    freed = math.hypot(pose.x - escape_origin[0], pose.y - escape_origin[1]) >= TRAP_RADIUS_M
                if freed:
                    counts["unwedged"] += 1
                    log.event("unwedged", attempts=escape_step, heading=round(heading, 1))
                    # 抜けた後に同じ向きで前進を再開すると、数歩で同じ隙間に戻ってまた挟まる
                    # （実機で約 4 秒ごとに 15 回繰り返した）。挟まった場所を崖として記録し、向きを変えてから歩く
                    explorer.mark_cliff(escape_origin[0], escape_origin[1], escape_heading)
                    inputs.set()
                    state, state_until, turn_avoid, planned = "turn_start", t, 120.0, None
                elif t >= state_until:
                    if escape_step >= len(ESCAPES):
                        stop_reason = "wedged"
                        log.event("abort", reason="挟まりから抜け出せない（コライダーの隙間の可能性。"
                                                  "VRChat でリスポーンしてから再開する）")
                        break
                    name, axes, sec = ESCAPES[escape_step]
                    escape_step += 1
                    inputs.set(**axes)
                    inputs.jump()
                    state_until = t + sec
                    log.event("escape_try", step=escape_step, action=name)
            elif fall_since is not None and t - fall_since >= args.wedge_sec:
                counts["wedged"] += 1
                motion = frames.motion() if frames is not None else None
                log.event("wedged", fall_sec=round(t - fall_since, 1), vel_y=round(snap["vel_y"], 1),
                          heading=round(heading, 1),
                          screen_motion=round(motion, 2) if motion is not None else None)
                if frames is not None:
                    n = frames.dump(os.path.join(run_dir, f"wedge_{counts['wedged']:02d}"), until=t)
                    log.event("frames_saved", wedge=counts["wedged"], count=n)
                overlay_marks.append((pose.x, pose.y, f"W{counts['wedged']}"))
                state, state_until, escape_step, escape_kind = "escape", t, 0, "fall"
                escape_origin, escape_heading = (pose.x, pose.y), heading

            if state == "forward":
                # 家具の上などで接地と浮き上がりを 67ms ごとに繰り返してはまることがあるので、
                # Grounded ではなく水平の速さで見る。判定を止めるのは本当に落ちているときだけ
                falling = snap["vel_y"] < FALLING_VY
                horizontal = max(0.0, snap["magnitude"] ** 2 - snap["vel_y"] ** 2) ** 0.5
                warm = t - forward_since > 0.6   # 動き出しの加速中は判定しない
                if warm and not falling and horizontal < stuck_level:
                    low_since = low_since if low_since is not None else t
                    if t - low_since >= args.stuck_sec:
                        counts["stuck"] += 1
                        corner = t - last_turn_end < 1.5
                        log.event("stuck", horizontal=round(horizontal, 2),
                                  heading=round(heading, 1), corner=corner,
                                  x=round(pose.x, 1), y=round(pose.y, 1))
                    if t - low_since >= args.stuck_sec and not args.no_climb \
                            and explorer.climb_worth_trying(pose.x, pose.y, heading):
                        # 向きを変える前に、前に進みながら 1 回だけジャンプしてみる（跳べる高さは約 0.75m）。
                        # 外周の柵を越えて落ちないよう、既知の崖の近くでは跳ばない
                        explorer.mark_climb_tried(pose.x, pose.y, heading)
                        counts["climb_try"] += 1
                        log.event("climb_try", heading=round(heading, 1), x=round(pose.x, 1), y=round(pose.y, 1))
                        climb_origin = (pose.x, pose.y, heading, pose.z)
                        inputs.set(vertical=1.0)
                        inputs.jump()
                        state, state_until, low_since = "climb", t + CLIMB_SEC, None
                    elif t - low_since >= args.stuck_sec:
                        explorer.mark_wall(pose.x, pose.y, heading)
                        # 目標を追う途中で何度も壁に詰まる = 目標が壁の向こう。40 秒待たずにあきらめる
                        if goal is not None:
                            goal_track["bumps"] += 1
                            if goal_track["bumps"] >= GOAL_MAX_BUMPS:
                                counts["goal_abandoned"] += 1
                                log.event("goal_abandoned", x=round(goal["x"], 1), y=round(goal["y"], 1),
                                          reason="bumps", bumps=goal_track["bumps"])
                                goal_track["avoid"].append((goal["x"], goal["y"]))
                                goal = None
                        # 同じ場所で何度も詰まる = 点数の上では良く見える向きが実は塞がっている。
                        # 斜めの壁に沿って滑り、同じ角に運ばれる往復を実機で 60 秒続けたことがある
                        stuck_log.append((t, pose.x, pose.y))
                        repeats = sum(1 for st, sx, sy in stuck_log
                                      if t - st <= LOOP_WINDOW and math.hypot(sx - pose.x, sy - pose.y) <= 1.5)
                        if repeats >= LOOP_REPEATS:
                            counts["loop_break"] += 1
                            log.event("loop_break", repeats=repeats)
                            stuck_log.clear()
                            planned = (heading + rng.choice((-1.0, 1.0)) * rng.uniform(90, 180)) % 360.0
                            # ループを何度断っても同じ場所から動けない = 閉じ込め（岩の隙間などで四方が塞がる）。
                            # 実機で 75 秒間、同じ場所でループの断ち切りを繰り返したことがある
                            loop_breaks.append((t, pose.x, pose.y))
                            if sum(1 for lt, lx, ly in loop_breaks
                                   if t - lt <= TRAP_WINDOW
                                   and math.hypot(lx - pose.x, ly - pose.y) <= TRAP_RADIUS_M) >= TRAP_REPEATS:
                                trapped = True
                                loop_breaks.clear()
                        if corner:
                            # 角にはまった: 少し下がってから大きく回る
                            inputs.set(vertical=-1.0)
                            state, state_until = "backup", t + 0.4
                            turn_avoid = 90.0
                        else:
                            state, state_until = "turn_start", t
                            turn_avoid = 60.0
                        low_since = None
                else:
                    low_since = None
                    # 壁沿いに滑っている: 進んではいるが、向きと動く方向が 30° 以上ずれている。
                    # 正面に壁があるので記録する（詰まった点だけでは斜めの壁が地図に載らない）
                    norm = math.hypot(snap["vx"], snap["vz"])
                    if (warm and not falling and norm > 0.5
                            and abs(snap["vx"]) / norm > math.sin(math.radians(30))):
                        explorer.mark_wall(pose.x, pose.y, heading, d=0.6)
                if state == "forward" and args.strategy == "frontier" and warm and not falling:
                    if explorer.cliff_ahead(pose.x, pose.y, heading):
                        counts["avoid_cliff"] += 1
                        log.event("avoid_cliff", heading=round(heading, 1),
                                  x=round(pose.x, 1), y=round(pose.y, 1))
                        state, turn_avoid = "turn_start", 90.0
                    elif t - last_plan >= REPLAN_SEC:
                        # まっすぐ進んでいる間も、踏破済みの場所ばかりなら向きを変える
                        last_plan = t
                        goal = update_goal(goal, t, pose)
                        steer = steer_point(goal, pose)
                        cur = explorer.score_toward(pose.x, pose.y, heading, steer, explore_weight=explore_w)
                        best_h, best_s = explorer.best_heading(pose.x, pose.y, heading, rng, target=steer,
                                                               explore_weight=explore_w)
                        if (best_s - cur > REPLAN_GAIN and cur < 0.5 * best_s
                                and abs(explore.signed_delta(best_h, heading)) >= 30.0):
                            counts["replan"] += 1
                            log.event("replan", score_now=round(cur, 2), score_best=round(best_s, 2))
                            state, planned = "turn_start", best_h
            elif state == "backup" and t >= state_until:
                state, state_until = "turn_start", t
            elif state == "climb" and t >= state_until and (snap["grounded"] or t >= state_until + 1.5):
                # 着地したら、ジャンプ前の位置から向きの方向にどれだけ進めたかで判定する
                ox, oy, oh, oz = climb_origin
                progress = ((pose.x - ox) * math.sin(math.radians(oh))
                            + (pose.y - oy) * math.cos(math.radians(oh)))
                if progress >= CLIMB_PROGRESS_M:
                    counts["climbed"] += 1
                    log.event("climbed", x=round(ox, 2), y=round(oy, 2), heading=round(oh, 1),
                              progress=round(progress, 2), dz=round(pose.z - oz, 2))
                    state, forward_since = "forward", t
                else:
                    log.event("climb_failed", progress=round(progress, 2))
                    explorer.mark_wall(ox, oy, oh)
                    inputs.set()
                    state, state_until, turn_avoid = "turn_start", t, 60.0

            if trapped:
                trapped = False
                counts["wedged"] += 1
                log.event("wedged", reason="trapped", heading=round(heading, 1),
                          x=round(pose.x, 1), y=round(pose.y, 1))
                if frames is not None:
                    # 閉じ込められた後の画面はカメラがめり込んで何も写らないことが多いので、少し前から残す
                    n = frames.dump(os.path.join(run_dir, f"wedge_{counts['wedged']:02d}"), until=t)
                    log.event("frames_saved", wedge=counts["wedged"], count=n)
                overlay_marks.append((pose.x, pose.y, f"W{counts['wedged']}"))
                planned = None
                state, state_until, escape_step, escape_kind = "escape", t, 0, "trap"
                escape_origin, escape_heading = (pose.x, pose.y), heading

            if state == "turn_start":
                if args.strategy == "frontier":
                    if planned is not None:
                        target, score = planned, None
                    else:
                        goal = update_goal(goal, t, pose)
                        target, score = explorer.best_heading(pose.x, pose.y, heading, rng,
                                                              avoid_ahead=turn_avoid, target=steer_point(goal, pose),
                                                              explore_weight=explore_w)
                        score = round(score, 2)
                    angle = explore.signed_delta(target, heading)
                else:
                    lo, hi = (135, 225) if t - last_turn_end < 1.5 else (90, 180)
                    angle = rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi)
                    score = None
                planned, turn_avoid = None, 0.0
                log.event("turn", angle=round(angle, 1), score=score)
                if abs(angle) < 5.0:
                    inputs.set(vertical=1.0)
                    state, forward_since = "forward", t
                else:
                    inputs.set(look=1.0 if angle > 0 else -1.0)
                    state, state_until = "turn", t + abs(angle) / TURN_RATE
                    turn_started, turn_checked = t, False
            elif state == "turn" and t >= state_until:
                last_turn_end = t
                inputs.set(vertical=1.0)
                state, forward_since = "forward", t
            elif state == "pause" and t >= state_until:
                if args.strategy == "frontier":
                    # スポーン地点から、まだ歩いていない方向を選んで出直す
                    state, state_until = "turn_start", t
                else:
                    inputs.set(vertical=1.0)
                    state, forward_since = "forward", t

            if patrol is not None and aligned and t >= next_patrol_touch:
                next_patrol_touch = t + 0.2
                patrol.touch(pose.x, pose.y, t)
            if relate_proc is not None and t >= next_pose_log:
                next_pose_log = t + 0.1
                pose_history.append((time.time(), pose.x, pose.y, pose.heading))
            if map_overlay is not None and t >= next_overlay:
                next_overlay = t + 0.2
                if not overlay_trail or math.hypot(pose.x - overlay_trail[-1][0],
                                                   pose.y - overlay_trail[-1][1]) > 0.3:
                    overlay_trail.append((pose.x, pose.y))
                    del overlay_trail[:-600]
                if relate_proc is not None:
                    live_obs_pos = read_live_obs(live_obs_path, live_obs_pos, pose_history, live_landmarks)
                map_overlay.update(pose, explorer, overlay_trail, overlay_marks, [
                    f"t {t:.0f}/{args.duration:.0f}s  {state}",
                    f"R{counts['respawn']} W{counts['wedged']} stuck {counts['stuck']}",
                    f"z {pose.z:+.1f}m" + ("" if aligned else "  (map not aligned)"),
                ], landmarks=live_landmarks.visible(time.time()), goal=goal,
                   patrol_cells=patrol.cells if patrol is not None else None)

            # 軸入力は値が保持される想定だが、取りこぼし対策で送り直す
            if int(t * 10) != int((t - dt) * 10):
                inputs.resend()
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_reason = "user"
    finally:
        # 止め忘れると歩き続けるので、何があっても全入力を 0 に戻す
        inputs.release_all()
        if frames is not None:
            frames.close()
        if map_overlay is not None:
            map_overlay.close()
        if relate_proc is not None:
            relate_proc.terminate()
        server.shutdown()

    drops = sum(1 for e in log.events if e["kind"] == "drop")
    summary = dict(stop_reason=stop_reason, elapsed=round(log.now(), 1), stuck=counts["stuck"],
                   respawn=counts["respawn"], drop=drops, wedged=counts["wedged"],
                   unwedged=counts["unwedged"], heading_estimate=round(heading, 1))
    if patrol is not None:
        summary.update(patrol=patrol_name, patrol_blocks=len(patrol.blocks),
                       patrol_coverage=round(patrol.coverage(), 3), patrol_reached=counts["patrol_reached"],
                       patrol_visits=sum(b["visits"] for b in patrol.blocks.values()))
    log.event("end", **summary)
    log.close()
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n全入力を 0 に戻した。結果: {run_dir}")
    # 地図（mapper）の作成と、ワールドの記録への取り込みも report.build の中で行う
    out = report.build(run_dir)
    with open(os.path.join(run_dir, "map.json"), encoding="utf-8") as f:
        m = json.load(f)
    print(f"地図: {os.path.join(run_dir, 'map.png')}  歩行距離 {m['distance_m']} m / "
          f"踏破面積 {m['walked_area_m2']} m2")
    for i, r in enumerate(m["respawns"], 1):
        print(f"  R{i}: ({r['x']:+.1f}, {r['y']:+.1f}) {r['verdict']}")
    print(f"報告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
