"""ワールドごとの記録。走行を重ねるほど地図と判定が充実する。

    worlds/<ワールドID>/knowledge.json

走行ごとの寄与（歩いた格子・壁・崖・落下・挟まり・乗り越え・見つけたもの）を分けて保存し、読み込むときに合算する。
同じ走行を取り込み直しても二重に数えず、失敗した走行は runs から消せば外せる。

使い方:
- walker.py の開始時: 過去の記録を探索の地図（explore.Explorer）に読み込む → 歩いていない場所を優先し、
  過去に落ちた・挟まった場所は最初から崖として避ける
- report.py: 床抜けの判定に、他の走行で歩いた床も使う（走るほど「外周か床抜けか」が正確になる）。
  今回の走行を取り込み、累計の地図を描く

座標はスポーン地点が原点なので、スポーン地点から始めた走行だけを取り込む。
スポーン地点が複数あるワールドでは位置が食い違うので、walker.py --fresh で累計を使わない。

    uv run tools/knowledge.py                       # 記録のあるワールドの一覧
    uv run tools/knowledge.py --backfill runs/*     # 過去の走行を取り込む（ワールドはログから引く）
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore  # noqa: E402
import mapper  # noqa: E402
import semantic  # noqa: E402
import world  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS_DIR = os.path.join(_ROOT, "worlds")
CELL = mapper.CELL


def path_of(world_id: str) -> str:
    return os.path.join(WORLDS_DIR, world_id, "knowledge.json")


def load(world_id: str) -> dict:
    p = path_of(world_id)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"world": {"id": world_id, "name": ""}, "runs": {}}


def save(k: dict) -> None:
    p = path_of(k["world"]["id"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(k, f, ensure_ascii=False)
    os.replace(tmp, p)


def run_world(run_dir: str) -> dict | None:
    """走行のワールド。start イベントに記録があればそれ、なければ開始時刻から VRChat のログで引く。"""
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "start":
                if e.get("world_id"):
                    return {"id": e["world_id"], "name": e.get("world_name", "")}
                break
    name = os.path.basename(os.path.normpath(run_dir))
    try:
        when = time.strftime("%Y.%m.%d %H:%M:%S", time.strptime(name, "%Y%m%d_%H%M%S"))
    except ValueError:
        return None
    return world.world_at(when)


# ------------------------------------------------------------------ 合算
def aggregate(k: dict, exclude: str | None = None) -> dict:
    """全走行（exclude を除く）の寄与を合算する。"""
    visited, walls, cliffs = set(), set(), set()
    falls, wedges, climbs, landmarks = [], [], [], []
    for name, r in sorted(k["runs"].items()):
        if name == exclude:
            continue
        visited.update(map(tuple, r["visited"]))
        walls.update(map(tuple, r["walls"]))
        cliffs.update(map(tuple, r["cliffs"]))
        falls += [{**f, "run": name} for f in r["falls"]]
        wedges += [{**w, "run": name} for w in r["wedges"]]
        climbs += [{**c, "run": name} for c in r["climbs"]]
        landmarks += [{**m, "run": name} for m in r["landmarks"]]
    return dict(visited=visited, walls=walls, cliffs=cliffs, falls=falls, wedges=wedges, climbs=climbs,
                landmarks=merge_landmarks(landmarks), runs=len(k["runs"]) - (1 if exclude in k["runs"] else 0))


def merge_landmarks(items: list[dict]) -> list[dict]:
    """走行をまたいで、同じ種類・近い位置の見つけたものをまとめる（見えた回数は足し合わせる）。"""
    out: list[dict] = []
    for m in sorted(items, key=lambda m: -m["frames"]):
        radius = semantic.CLUSTER_M[m["kind"]]
        for o in out:
            if o["kind"] == m["kind"] and o["label"] == m["label"] \
                    and math.hypot(o["x"] - m["x"], o["y"] - m["y"]) <= radius:
                n = o["frames"] + m["frames"]
                o["x"] = (o["x"] * o["frames"] + m["x"] * m["frames"]) / n
                o["y"] = (o["y"] * o["frames"] + m["y"] * m["frames"]) / n
                o["frames"], o["score"] = n, max(o["score"], m["score"])
                o["runs"].add(m["run"])
                break
        else:
            out.append({**m, "runs": {m["run"]}})
    for o in out:
        o["runs"] = len(o["runs"])
    out.sort(key=lambda o: (-o["runs"], -o["frames"]))
    return out


def apply_to_explorer(agg: dict, ex: explore.Explorer) -> None:
    """過去の記録を探索の地図に読み込む。"""
    for cx, cy in agg["visited"]:
        ex.mark_visited((cx + 0.5) * CELL, (cy + 0.5) * CELL)
    ex.walls.update(agg["walls"])
    ex.cliffs.update(agg["cliffs"])


# ------------------------------------------------------------------ 取り込み
def contribution(run_dir: str) -> dict | None:
    """1 回の走行の寄与。スポーン地点から始めていない走行は、最初のリスポーンより後の区間だけを使う。"""
    at_spawn = mapper.started_at_spawn(run_dir)
    data = mapper.replay(run_dir)
    # 落下の判定はこの走行の中だけで行う（走行ごとに位置が数 m ずれるので、累計の床では判定しない）
    mapper.classify_falls(data, include_track0=at_spawn)
    ok = lambda tr: at_spawn or tr > 0  # noqa: E731
    visited = {mapper_cell(x, y) for tr, x, y, g in data["trail"] if g and ok(tr)}
    if not visited:
        return None
    ex = explore.Explorer()
    # 壁は走行中の地図（explore.mark_wall）と同じく横 1.5m で残す。1 マスだと少し斜めの向きの通路がすり抜ける
    for w in data["walls"]:
        if ok(w["track"]):
            ex.mark_wall(w["px"], w["py"], w["heading"])
    walls = set(ex.walls)
    for r in data["respawns"]:
        if ok(r["track"]):
            ex.mark_cliff(r["x"], r["y"], r["heading"])
    for w in data["wedges"]:
        if ok(w["track"]):
            ex.mark_cliff(w["x"], w["y"], 0.0)
    rnd = lambda v: round(v, 2)  # noqa: E731
    sem_path = os.path.join(run_dir, "semantic.json")
    landmarks = []
    if os.path.exists(sem_path):
        with open(sem_path, encoding="utf-8") as f:
            for m in json.load(f).get("landmarks", []):
                landmarks.append({k: m[k] for k in ("kind", "label", "label_ja", "x", "y", "z", "frames", "score")})
    return {
        "added": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance_m": round(data["distance"], 1),
        "visited": sorted(visited), "walls": sorted(walls), "cliffs": sorted(ex.cliffs),
        "falls": [dict(x=rnd(r["x"]), y=rnd(r["y"]), z=rnd(r.get("z", 0)), heading=rnd(r["heading"]),
                       verdict=r["verdict"])
                  for r in data["respawns"] if ok(r["track"])],
        "wedges": [dict(x=rnd(w["x"]), y=rnd(w["y"]), z=rnd(w.get("z", 0)), reason=w.get("reason", "fall"))
                   for w in data["wedges"] if ok(w["track"])],
        "climbs": [dict(x=rnd(c["x"]), y=rnd(c["y"]), z=rnd(c.get("z", 0))) for c in data["climbs"] if ok(c["track"])],
        "landmarks": landmarks,
    }


def mapper_cell(x: float, y: float) -> tuple[int, int]:
    return math.floor(x / CELL), math.floor(y / CELL)


def add_run(run_dir: str, w: dict | None = None) -> tuple[dict, dict] | None:
    """走行を取り込む（同じ走行は置き換える）。(記録, 寄与) を返す。ワールドが分からなければ None。"""
    w = w or run_world(run_dir)
    if not w:
        return None
    c = contribution(run_dir)
    if c is None:
        return None
    k = load(w["id"])
    if w.get("name"):
        k["world"]["name"] = w["name"]
    k["runs"][os.path.basename(os.path.normpath(run_dir))] = c
    save(k)
    return k, c


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", nargs="*", help="取り込む走行フォルダ")
    args = ap.parse_args()
    for run_dir in args.backfill or []:
        r = add_run(run_dir)
        print(f"{run_dir}: " + (f"{r[0]['world']['name']} に取り込んだ（{len(r[1]['visited']) * CELL * CELL:.0f} m²）"
                                if r else "取り込めない（ワールドが不明か、位置の分かる区間がない）"))
    for p in sorted(glob.glob(os.path.join(WORLDS_DIR, "*", "knowledge.json"))):
        with open(p, encoding="utf-8") as f:
            k = json.load(f)
        agg = aggregate(k)
        print(f"{k['world']['name'] or '(名前不明)'}  {k['world']['id']}  走行 {agg['runs']} 回 / "
              f"累計の踏破 {len(agg['visited']) * CELL * CELL:.0f} m² / 落下 {len(agg['falls'])} / "
              f"挟まり {len(agg['wedges'])} / 見つけたもの {len(agg['landmarks'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
