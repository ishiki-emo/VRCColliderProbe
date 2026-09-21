"""「どこに何があるか」。走行中に保存した画面を RelateAnything にかけ、アバターの推定位置に結び付ける。

    uv run tools/semantic.py runs/<日時>        # 単体で作る（report.py からも呼ばれる）

出力: <run_dir>/semantic.json

使う関係はアバターが主語のものだけ:
- 足元（surface）: 「アバター は X の上にいる／立っている」→ その時のアバターの位置に X がある
- 近く（nearby）  : 「アバター は X の近く／隣／前／後ろにいる」→ アバターから、画面上で X が写っている方角へ
  NEARBY_OFFSET_M ずらした位置に置く（三人称のカメラはアバターの向きを見ているので、画面の横位置で方角が出る）
同じ種類の物が CLUSTER_M 以内で何度も見つかれば 1 つにまとめる。位置の精度は位置の推定と同じく数 m。

画面は walker.py が frames/ に 1 秒ごとに保存したもの。frames/ がない古い走行では、
落下や挟まりの前後に保存した画面を使う（場所が偏るので参考程度）。
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapper  # noqa: E402

SURFACE_PREDICATES = {"on", "standing on", "walking on", "sitting on", "lying on"}
NEARBY_PREDICATES = {"near", "next to", "beside", "in front of", "behind", "against", "leaning on",
                     "along", "in"}
IGNORE_LABELS = {"sky", "edge"}      # 場所を表さない（「sky の近く」はどこでも出る）
GENERIC_LABELS = {"floor", "ground"}   # ミニマップには出さない
FLOOR_LABELS = {"floor", "ground", "stairs", "step", "slope", "ramp", "platform", "bridge", "walkway",
                "ledge", "grass", "water", "pool", "rock"}
MIN_SCORE = 0.30
CAMERA_HFOV = 80.0                    # 三人称カメラの横の視野角の目安 [deg]（デスクトップ版の既定に近い値）
NEARBY_OFFSET_M = 1.5
CLUSTER_M = {"surface": 2.5, "nearby": 3.0}

LABELS_JA = {
    "floor": "床", "ground": "地面", "stairs": "階段", "step": "段差", "slope": "坂", "ramp": "スロープ",
    "platform": "台・足場", "bridge": "橋", "walkway": "通路", "wall": "壁", "pillar": "柱", "arch": "アーチ",
    "railing": "手すり", "fence": "柵", "door": "ドア", "window": "窓", "ledge": "縁", "edge": "端",
    "table": "テーブル", "chair": "椅子", "sofa": "ソファ", "cushion": "クッション", "bench": "ベンチ",
    "bed": "ベッド", "shelf": "棚", "lamp": "ランプ", "lantern": "ランタン", "sign": "看板", "poster": "ポスター",
    "plant": "植物", "tree": "木", "bush": "茂み", "flower": "花", "rock": "岩", "grass": "草",
    "water": "水面", "pool": "プール", "fountain": "噴水", "sky": "空", "portal": "ポータル",
}


def frame_paths(run_dir: str) -> tuple[list[str], str]:
    """(画面のパス, どこから取ったか)。"""
    logged = sorted(glob.glob(os.path.join(run_dir, "frames", "t*.jpg")))
    if logged:
        return logged, "frames"
    # 古い走行: 落下・挟まりの前後に保存した画面（5 枚/秒）を 1 秒に 1 枚に間引いて使う
    others, last = [], {}
    for f in sorted(glob.glob(os.path.join(run_dir, "respawn_*", "t*.jpg"))
                    + glob.glob(os.path.join(run_dir, "wedge_*", "t*.jpg"))):
        folder, t = os.path.dirname(f), frame_time(f)
        if t - last.get(folder, -9.0) >= 1.0:
            others.append(f)
            last[folder] = t
    return others, "event_frames"


def frame_time(path: str) -> float:
    return float(os.path.basename(path)[1:-4])


class PoseLookup:
    def __init__(self, data: dict):
        self.t = data["trail_t"]
        self.data = data

    def at(self, t: float):
        """時刻 t の (track, x, y, z, heading, grounded)。範囲外なら None。"""
        i = bisect.bisect_right(self.t, t) - 1
        if i < 0 or i >= len(self.t) or t - self.t[i] > 0.5:
            return None
        tr, x, y, g = self.data["trail"][i]
        return tr, x, y, self.data["trail_z"][i], self.data["trail_h"][i], g


def relation_hits(ann: dict) -> list[dict]:
    """注釈から「アバターの足元／近くにあるもの」を拾う。位置はまだ付けない（place で付ける）。

    cx は画面上の横位置（0〜1）。リアルタイム表示（relate_overlay.py → walker.py）でも使う。
    """
    import report
    objs = ann.get("objects", [])
    me = report.avatar_index(ann)
    if me is None:
        return []
    W = ann["size"][0]
    out = []
    for r in ann.get("relations", []):
        if r["subject"] != me or r["object"] >= len(objs) or r["score"] < MIN_SCORE:
            continue
        o = objs[r["object"]]
        label = o["label"]
        if label in report.AVATAR_LABELS or label in IGNORE_LABELS:
            continue
        # 「上にいる」を足元として扱うのは床らしいものだけ（「テーブルの上にいる」のような誤りが出た）
        if r["predicate"] in SURFACE_PREDICATES and label in FLOOR_LABELS:
            kind = "surface"
        elif r["predicate"] in NEARBY_PREDICATES or r["predicate"] in SURFACE_PREDICATES:
            kind = "nearby"
        else:
            continue
        out.append(dict(kind=kind, label=label, score=r["score"], predicate=r["predicate"],
                        cx=round((o["box"][0] + o["box"][2]) / 2 / W, 3)))
    return out


def place(hit: dict, x: float, y: float, heading: float) -> tuple[float, float]:
    """足元はアバターの位置、近くは画面上の方角へ NEARBY_OFFSET_M ずらした位置。"""
    if hit["kind"] == "surface":
        return x, y
    bearing = math.radians(heading + (hit["cx"] - 0.5) * CAMERA_HFOV)
    return x + NEARBY_OFFSET_M * math.sin(bearing), y + NEARBY_OFFSET_M * math.cos(bearing)


def observations(ann: dict, pose, path: str, t: float) -> list[dict]:
    tr, x, y, z, heading, _ = pose
    out = []
    for h in relation_hits(ann):
        ox, oy = place(h, x, y, heading)
        out.append(dict(kind=h["kind"], label=h["label"], x=ox, y=oy, z=z, score=h["score"], t=t,
                        frame=path, predicate=h["predicate"]))
    return out


class LiveLandmarks:
    """走行中のミニマップ用。見つけたものを報告と同じ半径でまとめ、確度の高いものだけを返す。

    位置の推定は時間とともにずれるので、最後に見えてから ttl 秒で消す（古いほど薄く描く）。
    """

    MIN_HITS = 3          # これ以上見えたら確度が高い
    MIN_BEST = 0.5        # または確信度がこれ以上

    def __init__(self, ttl: float = 120.0):
        self.ttl = ttl
        self.items: list[dict] = []

    def clear(self) -> None:
        self.items.clear()

    def add(self, hit: dict, x: float, y: float, heading: float, now: float) -> None:
        ox, oy = place(hit, x, y, heading)
        radius = CLUSTER_M[hit["kind"]]
        for it in self.items:
            if it["kind"] == hit["kind"] and it["label"] == hit["label"] \
                    and math.hypot(it["x"] - ox, it["y"] - oy) <= radius:
                n = it["n"] + 1
                it["x"] += (ox - it["x"]) / n
                it["y"] += (oy - it["y"]) / n
                it["n"], it["best"], it["last"] = n, max(it["best"], hit["score"]), now
                return
        self.items.append(dict(kind=hit["kind"], label=hit["label"], x=ox, y=oy, n=1,
                               best=hit["score"], last=now))

    def visible(self, now: float) -> list[dict]:
        self.items = [it for it in self.items if now - it["last"] <= self.ttl]
        out = []
        for it in self.items:
            # 床・地面はどこにでもあって目印にならない（ミニマップが floor だらけになった）
            if it["label"] in GENERIC_LABELS:
                continue
            if it["n"] >= self.MIN_HITS or it["best"] >= self.MIN_BEST:
                out.append({**it, "fade": max(0.35, 1.0 - (now - it["last"]) / self.ttl)})
        return out


def cluster(obs: list[dict]) -> list[dict]:
    """同じ (種類, ラベル) で近いものをまとめる（貪欲法。観測の多い順に核を決める）。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for o in obs:
        groups.setdefault((o["kind"], o["label"]), []).append(o)
    marks = []
    for (kind, label), items in groups.items():
        radius = CLUSTER_M[kind]
        clusters: list[list[dict]] = []
        for o in sorted(items, key=lambda o: -o["score"]):
            for c in clusters:
                cx = sum(p["x"] for p in c) / len(c)
                cy = sum(p["y"] for p in c) / len(c)
                if math.hypot(o["x"] - cx, o["y"] - cy) <= radius:
                    c.append(o)
                    break
            else:
                clusters.append([o])
        for c in clusters:
            # 見えた回数は観測した瞬間で数える（リアルタイムの観測には保存した画面がない）
            seen = {p["frame"] or f"live@{p['t']:.2f}" for p in c}
            best = max(c, key=lambda p: p["score"])
            with_frame = [p for p in c if p["frame"]]
            thumb = max(with_frame, key=lambda p: p["score"])["frame"] if with_frame else None
            marks.append(dict(
                kind=kind, label=label, label_ja=LABELS_JA.get(label, label),
                x=round(sum(p["x"] for p in c) / len(c), 2), y=round(sum(p["y"] for p in c) / len(c), 2),
                z=round(sum(p["z"] for p in c) / len(c), 2), frames=len(seen),
                score=round(best["score"], 2), best_frame=thumb,
                first_t=round(min(p["t"] for p in c), 1), last_t=round(max(p["t"] for p in c), 1)))
    # 1 回しか見えていない弱い検出は雑音のことが多い
    marks = [m for m in marks if m["frames"] >= 2 or m["score"] >= 0.5]
    marks.sort(key=lambda m: (-m["frames"], -m["score"]))
    return marks


def epoch0_of(run_dir: str) -> float | None:
    """ログの時刻 0 に当たるエポック秒。walker.py の start イベントに記録がなければ、
    frames/ の画面ファイルの更新時刻（= 保存した時刻）とファイル名の時刻の差から推定する。"""
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "start":
                if e.get("epoch0") is not None:
                    return float(e["epoch0"])
                break
    frames = sorted(glob.glob(os.path.join(run_dir, "frames", "t*.jpg")))[:20]
    if not frames:
        return None
    # 保存はファイル名の時刻の直後なので、差の最小値が一番近い
    return min(os.path.getmtime(f) - frame_time(f) for f in frames)


def live_observations(run_dir: str, lookup: "PoseLookup", at_spawn: bool) -> list[dict]:
    """走行中に relate_overlay.py が記録した観測（live_obs.jsonl）。毎秒約 4 回なので、
    1 秒ごとに保存した画面より多くの物を拾える（実機で 3 分 535 件。ランプ・看板・ポータルなど）。"""
    path = os.path.join(run_dir, "live_obs.jsonl")
    epoch0 = epoch0_of(run_dir)
    if not os.path.exists(path) or epoch0 is None:
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            t = rec["epoch"] - epoch0
            pose = lookup.at(t)
            if pose is None:
                continue
            tr, x, y, z, heading, grounded = pose
            if not grounded or (tr == 0 and not at_spawn):
                continue
            for h in rec["hits"]:
                ox, oy = place(h, x, y, heading)
                out.append(dict(kind=h["kind"], label=h["label"], x=ox, y=oy, z=z, score=h["score"], t=t,
                                frame=None, predicate=h["predicate"]))
    return out


def build(run_dir: str, annotate=None) -> dict:
    """semantic.json を作って返す。annotate(paths) -> {path: 注釈} を渡さなければ report.annotate を使う。"""
    import report
    run_dir = os.path.normpath(run_dir)
    paths, source = frame_paths(run_dir)
    at_spawn = mapper.started_at_spawn(run_dir)
    result = {"source": source, "frames": len(paths), "landmarks": [], "problem": None}
    if paths:
        notes, problem = (annotate or (lambda ps: report.annotate(run_dir, ps)))(paths)
        result["problem"] = problem
        data = mapper.replay(run_dir)
        lookup = PoseLookup(data)
        obs, used = [], 0
        for p in paths:
            ann = notes.get(p)
            pose = lookup.at(frame_time(p))
            if ann is None or pose is None:
                continue
            tr, _, _, _, _, grounded = pose
            if not grounded or (tr == 0 and not at_spawn):   # 空中や、位置が合っていない区間は使わない
                continue
            used += 1
            obs += observations(ann, pose, os.path.relpath(p, run_dir), frame_time(p))
        live = live_observations(run_dir, lookup, at_spawn)
        result.update(frames_used=used, observations=len(obs), live_observations=len(live),
                      landmarks=cluster(obs + live))
    with open(os.path.join(run_dir, "semantic.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    return result


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args()
    r = build(args.run_dir)
    print(f"画面 {r['frames']} 枚（{r['source']}）→ 使えた {r.get('frames_used', 0)} 枚 / "
          f"観測 {r.get('observations', 0)} / まとめて {len(r['landmarks'])} 件")
    for m in r["landmarks"][:20]:
        where = "足元" if m["kind"] == "surface" else "近く"
        print(f"  {m['label_ja']:<8} {where}  右 {m['x']:+6.1f} m / 前 {m['y']:+6.1f} m / 高さ {m['z']:+.1f} m"
              f"  {m['frames']} 枚  最高 {m['score']}")
    if r.get("problem"):
        print("注意:", r["problem"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
