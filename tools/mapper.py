"""段階 3: 地図。walker.py の記録（osc.csv / events.jsonl）を再生して位置を推定し、地図を描く。

    uv run tools/mapper.py runs/20260921_140726
    uv run tools/mapper.py runs/20260921_140726 --at-spawn   # walker.py の記録を上書きする

出力: <run_dir>/map.png（地図）、map.json（軌跡・出来事の座標）

座標系: リスポーン地点を原点、スポーン時の正面を +y（画像の上）、右を +x とする。
リスポーンすると位置も向きもスポーン地点に戻るので、推定もそこでリセットする。

位置の推定（SPEC.md 3 章の実測に基づく）:
- 速さ: VelocityMagnitude（実速度がすぐ反映される 3 次元の大きさ）から VelocityY を除いた水平成分
- 方向: VelocityX/Z（アバター基準。平滑化されているので方向にだけ使う）
- 向き: AngularY [deg/s] の積分。右旋回が +
値は変化時しか届かないので、次のメッセージまで直前の値が続いたものとして積分する。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

CELL = 0.5                 # 格子の大きさ [m]
TRAIL_STEP = 0.1           # メッセージが来ない間も、この間隔 [s] で軌跡を記録する
# 床抜け判定: 落下地点の周り 8 方向（45° ずつ）に、この距離の範囲で踏破済みの床があるかを見る。
# 位置の推定は数 m ずれるので、「前方に床があるか」だけでは端と穴を区別できない。
# 端なら床は片側（4〜5 方向）にしかなく、穴なら周りを囲まれている
RING_MIN, RING_MAX = 1.0, 4.0     # [m]
HOLE_OCTANTS = 7           # これ以上の方向に床があれば床抜けの疑い
EDGE_OCTANTS = 5           # これ以下なら外周の可能性（間は判定保留）
MIN_GROUNDED_SEC = 0.2     # これより短い接地は「最後に接地した位置」を更新しない [s]
TRACK_WINDOW = 2          # 床抜けの判定に使う床: 落ちた区間とその前後この数の区間だけ


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0   # [deg] 0 = +y、右回りが +
    z: float = 0.0         # 高さ [m]（スポーン地点が 0）

    def copy(self) -> "Pose":
        return Pose(self.x, self.y, self.heading, self.z)


@dataclass
class DeadReckoner:
    """パラメータを時刻順に受け取り、位置と向きを積分する。live でも再生でも使える。"""

    pose: Pose = field(default_factory=Pose)
    vx: float = 0.0
    vz: float = 0.0
    vy: float = 0.0
    magnitude: float = 0.0
    angular_y: float = 0.0
    grounded: bool = True
    last_t: float | None = None
    last_grounded_pose: Pose = field(default_factory=Pose)
    track: int = 0
    distance: float = 0.0
    grounded_since: float = 0.0

    def advance(self, t: float) -> None:
        """t まで積分する。値は変化時しか届かないので、直進中はメッセージがなくても呼ぶこと。"""
        if self.last_t is not None and t <= self.last_t:
            return   # 別スレッドから少し古い時刻で呼ばれても巻き戻さない
        if self.last_t is not None:
            dt = t - self.last_t
            h = math.radians(self.pose.heading)
            speed = math.sqrt(max(0.0, self.magnitude ** 2 - self.vy ** 2))
            norm = math.hypot(self.vx, self.vz)
            if speed > 0.0 and norm > 0.05:
                lx, lz = self.vx / norm * speed, self.vz / norm * speed
                # アバター基準（x: 右, z: 前）→ 地図（x: 右, y: 上）
                self.pose.x += (lz * math.sin(h) + lx * math.cos(h)) * dt
                self.pose.y += (lz * math.cos(h) - lx * math.sin(h)) * dt
                self.distance += speed * dt
            self.pose.heading = (self.pose.heading + self.angular_y * dt) % 360.0
            # 高さ: 接地中も坂や階段では VelocityY が届く（実測 +1.2 / -1.4 m/s）ので常に積分する
            self.pose.z += self.vy * dt
        self.last_t = t

    def feed(self, t: float, name: str, value) -> None:
        self.advance(t)
        if name == "VelocityX":
            self.vx = float(value)
        elif name == "VelocityZ":
            self.vz = float(value)
        elif name == "VelocityY":
            self.vy = float(value)
        elif name == "VelocityMagnitude":
            self.magnitude = float(value)
        elif name == "AngularY":
            self.angular_y = float(value)
        elif name == "Grounded":
            g = bool(value)
            if g and not self.grounded:
                self.grounded_since = t
            if self.grounded and not g and t - self.grounded_since >= MIN_GROUNDED_SEC:
                # 落ち始めの位置。段差でぱたぱたしても、最後の離陸点が残る。
                # ごく短い接地は数えない: 挟まりの途中で 50ms だけ接地することがあり、そのときの高さは
                # 落下の積分で大きく下がっている（30 分の走行で -53m になった）
                self.last_grounded_pose = self.pose.copy()
            self.grounded = g

    def respawn(self, t: float) -> Pose:
        """リスポーン。落ちる直前の接地位置を返し、スポーン地点に戻す。"""
        self.advance(t)
        fall_from = self.last_grounded_pose.copy()
        self.pose = Pose()
        self.last_grounded_pose = Pose()
        self.track += 1
        return fall_from


def parse_value(s: str):
    s = s.split(" ")[0] if s else ""
    if s in ("True", "False"):
        return s == "True"
    try:
        return float(s)
    except ValueError:
        return s


def replay(run_dir: str) -> dict:
    events = []
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    # リスポーンは「急変した時刻（at）」で位置をリセットする
    marks = []
    for e in events:
        if e["kind"] == "respawn":
            marks.append((e["at"], "respawn", e))
        elif e["kind"] in ("stuck", "drop", "wedged", "unwedged", "climb_try", "climbed"):
            marks.append((e["t"], e["kind"], e))
    marks.sort(key=lambda m: m[0])
    # リスポーンまでの落差（0.5 g t²）の典型的な値。これより深く落ちてリスポーンせずに着地することはない。
    # 最大値だと、壁を滑り降りてからリスポーンした異常に長い落下（2.4 秒・28m）に引っ張られたので 75% 点を使う
    drops = sorted(0.5 * 9.81 * (e.get("fall_time") or 0.0) ** 2 for e in events if e["kind"] == "respawn")
    max_drop = drops[int(len(drops) * 0.75)] if len(drops) >= 3 else 30.0

    dr = DeadReckoner()
    trail: list[tuple[int, float, float, bool]] = []   # (track, x, y, 接地中か)
    trail_z: list[float] = []                          # trail と同じ並びの高さ [m]
    trail_t: list[float] = []                          # trail と同じ並びの時刻 [s]
    # 向きも時刻から引けるよう、trail と同じ並びで残す（semantic.py が使う）
    trail_h: list[float] = []
    out = {"respawns": [], "walls": [], "drops": [], "wedges": [], "climbs": []}
    last_climb_try: list[float] = [-99.0]
    wedge_z: list[float | None] = [None]   # 挟まった高さ（抜けたときに戻す）
    mi = 0

    def flush_marks(until: float) -> None:
        nonlocal mi
        while mi < len(marks) and marks[mi][0] <= until:
            t, kind, e = marks[mi]
            mi += 1
            if kind == "respawn":
                track = dr.track
                p = dr.respawn(t)
                out["respawns"].append(dict(t=t, track=track, x=p.x, y=p.y, z=p.z, heading=p.heading,
                                            fall_time=e.get("fall_time")))
            else:
                dr.advance(t)
                p = dr.pose
                if kind == "stuck":
                    # 壁は正面 0.3m 先にあるとみなす
                    h = math.radians(p.heading)
                    out["walls"].append(dict(t=t, track=dr.track, x=p.x + 0.3 * math.sin(h),
                                             y=p.y + 0.3 * math.cos(h), z=p.z,
                                             px=p.x, py=p.y, heading=p.heading))   # 詰まった位置と向き
                elif kind == "wedged":
                    # 挟まった位置 = 落下状態に入る直前の接地位置。閉じ込め（接地したまま動けない）は今の位置
                    trapped = e.get("reason") == "trapped"
                    q = dr.pose if trapped else dr.last_grounded_pose
                    out["wedges"].append(dict(t=t, track=dr.track, x=q.x, y=q.y, z=q.z,
                                              fall_sec=e.get("fall_sec"), reason=e.get("reason", "fall")))
                    # 挟まっている間も VelocityY は重力どおり増え続けるが、実際には動いていない。
                    # 積分した高さは使えないので、挟まる直前の接地の高さに戻す（実機で -108 m になった）
                    wedge_z[0] = q.z
                    dr.pose.z = q.z
                    # 挟まりは落下状態が 3 秒続いて初めて分かるので、その間の軌跡の高さもさかのぼって直す
                    since = t - (e.get("fall_sec") or 3.0)
                    i = len(trail_t) - 1
                    while i >= 0 and trail_t[i] >= since and trail[i][0] == dr.track:
                        trail_z[i] = q.z
                        i -= 1
                elif kind == "unwedged":
                    if wedge_z[0] is not None:
                        dr.pose.z = wedge_z[0]
                        wedge_z[0] = None
                elif kind == "climb_try":
                    last_climb_try[0] = t
                elif kind == "climbed":
                    # ジャンプで乗り越えた壁（跳ぶ前の位置）。dz は乗り越えた後の高さの変化
                    out["climbs"].append(dict(t=t, track=dr.track, x=e["x"], y=e["y"], z=p.z - (e.get("dz") or 0.0),
                                              heading=e.get("heading"), dz=e.get("dz")))
                elif t - last_climb_try[0] > 1.6:
                    # 自分でジャンプした後の着地は段差として数えない
                    out["drops"].append(dict(t=t, track=dr.track, x=p.x, y=p.y, z=p.z,
                                             airtime=e.get("airtime")))

    with open(os.path.join(run_dir, "osc.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "rx":
                continue
            t = float(row["t"])
            # 直進中はメッセージが来ないので、間を刻んで軌跡を補う
            while dr.last_t is not None and dr.last_t + TRAIL_STEP < t:
                step_to = dr.last_t + TRAIL_STEP
                flush_marks(step_to)
                dr.advance(step_to)
                trail.append((dr.track, dr.pose.x, dr.pose.y, dr.grounded))
                trail_z.append(dr.pose.z)
                trail_t.append(step_to)
                trail_h.append(dr.pose.heading)
            flush_marks(t)
            name = row["address"].rsplit("/", 1)[-1]
            value = parse_value(row["values"])
            landing = name == "Grounded" and value is True and not dr.grounded
            dr.feed(t, name, value)
            if landing and dr.last_grounded_pose.z - dr.pose.z > max_drop:
                # 壁に擦れながら落下状態で滑り降りると、VelocityY は自由落下のように増え続けるが、実際にはそこまで
                # 落ちていない（30 分の走行で 3 秒滑って -44m になった）。リスポーンせずに着地したなら、
                # リスポーンまでの落差より深くは落ちていないはずなので打ち止めにする
                dr.pose.z = dr.last_grounded_pose.z - max_drop
            trail.append((dr.track, dr.pose.x, dr.pose.y, dr.grounded))
            trail_z.append(dr.pose.z)
            trail_t.append(t)
            trail_h.append(dr.pose.heading)
    flush_marks(float("inf"))
    out["trail"] = trail
    out["trail_z"] = trail_z
    out["trail_t"] = trail_t
    out["trail_h"] = trail_h
    out["distance"] = dr.distance
    return out


def classify_falls(data: dict, include_track0: bool, extra_visited: set | None = None,
                   track_window: int = TRACK_WINDOW) -> None:
    """落下地点の周りを歩いた床が囲んでいるかで「床抜けの疑い」と「外周の可能性」を分ける。

    判定に使う床は、落ちた区間とその前後 track_window 区間（リスポーンで区切った区間）だけ。
    位置の推定は区間ごとにずれ方が違うので、全区間を重ねると外周の外側にまで「床」ができ、
    外周の落下が床抜けの疑いに見える（30 分・71 区間の走行で 9 件。画面で確かめた 4 件はすべて外周だった）。
    前後 2 区間にすると 0 件になった。
    """
    by_track: dict[int, set] = {}
    for track, x, y, grounded in data["trail"]:
        if grounded and (include_track0 or track > 0):
            by_track.setdefault(track, set()).add((math.floor(x / CELL), math.floor(y / CELL)))
    for r in data["respawns"]:
        if r["track"] == 0 and not include_track0:
            r["verdict"] = "unaligned"   # 開始位置が不明な区間なので判定しない
            continue
        visited = set(extra_visited or ())
        for tr, cells in by_track.items():
            if abs(tr - r["track"]) <= track_window:
                visited |= cells
        covered = []
        for k in range(8):
            hit = False
            for a in (-15.0, 0.0, 15.0):
                h = math.radians(r["heading"] + 45.0 * k + a)
                d = RING_MIN
                while d <= RING_MAX and not hit:
                    hit = (math.floor((r["x"] + d * math.sin(h)) / CELL),
                           math.floor((r["y"] + d * math.cos(h)) / CELL)) in visited
                    d += CELL / 2
                if hit:
                    break
            covered.append(hit)
        n = sum(covered)
        r["octants_with_floor"] = n
        r["floor_ahead"] = covered[0]     # k=0 が落ちた方向
        if n >= HOLE_OCTANTS and covered[0]:
            r["verdict"] = "hole_suspect"
        elif n <= EDGE_OCTANTS:
            r["verdict"] = "edge_likely"
        else:
            r["verdict"] = "unclear"


def height_range(zs) -> tuple[float, float]:
    zs = list(zs)
    if not zs:
        return 0.0, 1.0
    lo, hi = min(zs), max(zs)
    return (lo, hi) if hi - lo >= 0.5 else (lo - 0.25, lo + 0.25)


def height_color(z: float, lo: float, hi: float) -> tuple[int, int, int]:
    """高さの色（BGR）。低いほど青緑、高いほど黄色。地図の背景（暗い灰色）に対して落ち着いた明るさ。"""
    t = min(1.0, max(0.0, (z - lo) / (hi - lo)))
    stops = [(0.0, (110, 80, 40)), (0.5, (90, 120, 60)), (1.0, (60, 140, 150))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return tuple(int(a + (b - a) * k) for a, b in zip(c0, c1))
    return stops[-1][1]


def render(data: dict, path: str, include_track0: bool, px_per_m: float = 20.0) -> dict:
    """地図を描いて保存し、座標と画像ピクセルの対応を返す（px = (x - x0) * s, py = (y1 - y) * s）。"""
    pts = [(x, y) for tr, x, y, _ in data["trail"] if include_track0 or tr > 0]
    pts += [(r["x"], r["y"]) for r in data["respawns"]] + [(0.0, 0.0)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    margin = 3.0
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    W, H = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    img = np.full((H, W, 3), 32, np.uint8)

    def P(x, y):
        return int((x - x0) * px_per_m), int((y1 - y) * px_per_m)

    # 1m 格子
    for gx in range(math.ceil(x0), math.floor(x1) + 1):
        cv2.line(img, P(gx, y0), P(gx, y1), (48, 48, 48) if gx % 5 else (64, 64, 64), 1)
    for gy in range(math.ceil(y0), math.floor(y1) + 1):
        cv2.line(img, P(x0, gy), P(x1, gy), (48, 48, 48) if gy % 5 else (64, 64, 64), 1)

    # 踏破した格子。高さで色分けする（格子ごとの平均の高さ）
    zsum: dict[tuple[int, int], list[float]] = {}
    for (tr, x, y, g), z in zip(data["trail"], data["trail_z"]):
        if g and (include_track0 or tr > 0):
            zsum.setdefault((math.floor(x / CELL), math.floor(y / CELL)), []).append(z)
    cell_z = {c: sum(v) / len(v) for c, v in zsum.items()}
    z_lo, z_hi = height_range(cell_z.values())
    for (cx, cy), z in cell_z.items():
        cv2.rectangle(img, P(cx * CELL, (cy + 1) * CELL), P((cx + 1) * CELL, cy * CELL),
                      height_color(z, z_lo, z_hi), -1)

    # 軌跡（区間ごとに色を変える。track 0 は開始位置不明なら薄く）
    palette = [(255, 200, 80), (80, 200, 255), (200, 120, 255), (120, 255, 160), (255, 140, 140)]
    prev = None
    for tr, x, y, g in data["trail"]:
        if tr == 0 and not include_track0:
            prev = None
            continue
        p = P(x, y)
        if prev is not None and prev[0] == tr:
            color = palette[tr % len(palette)] if g else (160, 160, 160)
            cv2.line(img, prev[1], p, color, 1, cv2.LINE_AA)
        prev = (tr, p)

    for w in data["walls"]:
        if w["track"] > 0 or include_track0:
            cv2.circle(img, P(w["x"], w["y"]), 3, (230, 230, 230), -1)
    for d in data["drops"]:
        if d["track"] > 0 or include_track0:
            cv2.circle(img, P(d["x"], d["y"]), 4, (0, 220, 255), 1)
    for c in data.get("climbs", []):
        if c["track"] > 0 or include_track0:
            cv2.drawMarker(img, P(c["x"], c["y"]), (80, 230, 80), cv2.MARKER_TRIANGLE_UP, 10, 2)
    for i, r in enumerate(data["respawns"], 1):
        if r["verdict"] == "unaligned":
            continue
        color = {"hole_suspect": (0, 0, 255), "edge_likely": (0, 140, 255)}.get(r["verdict"], (200, 0, 200))
        c = P(r["x"], r["y"])
        cv2.drawMarker(img, c, color, cv2.MARKER_TILTED_CROSS, 16, 2)
        h = math.radians(r["heading"])
        cv2.arrowedLine(img, c, P(r["x"] + 1.5 * math.sin(h), r["y"] + 1.5 * math.cos(h)), color, 1)
        cv2.putText(img, f"R{i}", (c[0] + 8, c[1] - 8), 0, 0.5, color, 1, cv2.LINE_AA)
    for i, w in enumerate(data["wedges"], 1):
        if w["track"] > 0 or include_track0:
            c = P(w["x"], w["y"])
            cv2.drawMarker(img, c, (255, 0, 255), cv2.MARKER_SQUARE, 14, 2)
            cv2.putText(img, f"W{i}", (c[0] + 8, c[1] - 8), 0, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(img, P(0, 0), (255, 255, 255), cv2.MARKER_STAR, 14, 1)
    cv2.putText(img, "spawn", (P(0, 0)[0] + 8, P(0, 0)[1] + 16), 0, 0.4, (255, 255, 255), 1)

    legend = ["1 cell = 1m (bold every 5m)",
              f"walked: blue (low {z_lo:+.1f}m) -> yellow (high {z_hi:+.1f}m)", "white dot: wall",
              "yellow ring: step drop", "red X: hole suspect", "orange X: edge likely",
              "purple X: unclear", "magenta square: wedged in colliders",
              "green triangle: climbed over by jumping"]
    for i, s in enumerate(legend):
        cv2.putText(img, s, (8, 16 + 14 * i), 0, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return {"x0": x0, "y1": y1, "px_per_m": px_per_m, "width": W, "height": H}


def started_at_spawn(run_dir: str) -> bool:
    """walker.py の start イベントに記録した値。記録がない古い走行は False とみなす。"""
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "start":
                return bool(e.get("at_spawn", False))
    return False


def build(run_dir: str, at_spawn: bool | None = None, extra_visited: set | None = None) -> dict:
    if at_spawn is None:
        at_spawn = started_at_spawn(run_dir)
    data = replay(run_dir)
    classify_falls(data, include_track0=at_spawn, extra_visited=extra_visited)
    geometry = render(data, os.path.join(run_dir, "map.png"), include_track0=at_spawn)
    summary = {
        "distance_m": round(data["distance"], 1),
        "walked_area_m2": len({(math.floor(x / CELL), math.floor(y / CELL))
                               for tr, x, y, g in data["trail"]
                               if g and (at_spawn or tr > 0)}) * CELL * CELL,
        "respawns": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in data["respawns"]],
        "walls": len(data["walls"]),
        "drops": len(data["drops"]),
        "wall_points": [[w["track"], round(w["x"], 2), round(w["y"], 2), round(w["z"], 2)]
                        for w in data["walls"]],
        "drop_points": [[d["track"], round(d["x"], 2), round(d["y"], 2), round(d["z"], 2)]
                        for d in data["drops"]],
        "climbs": len(data["climbs"]),
        "climb_points": [[c["track"], round(c["x"], 2), round(c["y"], 2), round(c["z"], 2)]
                         for c in data["climbs"]],
        "wedges": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in w.items()}
                   for w in data["wedges"]],
        "started_at_spawn": at_spawn,
        "map_geometry": geometry,
    }
    with open(os.path.join(run_dir, "map.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "trail": [[tr, round(x, 3), round(y, 3), g]
                                        for tr, x, y, g in data["trail"]],
                   "trail_z": [round(z, 3) for z in data["trail_z"]]},
                  f, ensure_ascii=False)
    return summary


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--at-spawn", dest="at_spawn", action="store_true", default=None,
                   help="スポーン地点から歩き始めた（既定は walker.py の記録に従う）")
    g.add_argument("--not-at-spawn", dest="at_spawn", action="store_false",
                   help="スポーン地点以外から歩き始めた（最初のリスポーンまでの区間を地図に載せない）")
    args = ap.parse_args()
    s = build(args.run_dir, at_spawn=args.at_spawn)
    labels = {"hole_suspect": "床抜けの疑い（周りを歩いた床が囲んでいる）",
              "edge_likely": "外周の可能性（周りの一部にしか床がない）",
              "unclear": "判定保留（周りの探索が足りない）",
              "unaligned": "判定なし（開始位置が不明な区間）"}
    print(f"歩行距離 {s['distance_m']} m / 踏破面積 {s['walked_area_m2']} m2 / "
          f"壁 {s['walls']} / 段差 {s['drops']}")
    for i, r in enumerate(s["respawns"], 1):
        print(f"  R{i}: t={r['t']} 落下開始 ({r['x']:+.1f}, {r['y']:+.1f}) 向き {r['heading']:.0f}° → "
              f"{labels[r['verdict']]}" + (f"（床のある方向 {r['octants_with_floor']}/8）"
                                             if "octants_with_floor" in r else ""))
    print(f"地図: {os.path.join(args.run_dir, 'map.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
