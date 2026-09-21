"""RelateAnything で画面キャプチャに注釈を付ける（物体の検出と、物体どうしの関係）。

このスクリプトは RelateAnything の環境（torch 入り）で動かす。VRCColliderProbe 側の環境には
torch を入れないので、report.py からは別プロセスとして呼ぶ。

    E:\\AIProject\\RelateAnything\\.venv\\Scripts\\python.exe tools/scene_annotate.py \\
        --out scene.json img1.jpg img2.jpg ...

出力（JSON）: {画像パス: {"objects": [{"label", "score", "box": [x1, y1, x2, y2]}],
                         "relations": [{"subject", "predicate", "object", "score"}]}}
モデルの読み込みに時間がかかるので、複数枚をまとめて 1 回で処理する。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

RELATE_ROOT = os.environ.get("RELATE_ANYTHING_ROOT", r"E:\AIProject\RelateAnything")

# 検出する物体の語彙。何でも拾う検出器（4585 種類）だと、アバターの髪が umbrella、画面全体が
# studio shot になるなど雑音が多かったので、床抜けの調査に関係するものに絞る
CLASSES = [
    "person", "floor", "ground", "stairs", "step", "slope", "ramp", "platform", "bridge", "walkway",
    "wall", "pillar", "arch", "railing", "fence", "door", "window", "ledge", "edge",
    "table", "chair", "sofa", "cushion", "bench", "bed", "shelf", "lamp", "lantern", "sign", "poster",
    "plant", "tree", "bush", "flower", "rock", "grass", "water", "pool", "fountain", "sky", "portal",
]


def load_pipeline(classes: str = ",".join(CLASSES), det_conf: float = 0.1):
    """RelateAnything のパイプライン。game_scene.py の CLI 既定値（チェックポイントなど）を使う。

    語彙を絞ると床や縁の確信度が低く出る（0.1〜0.3）。0.25 だとほぼ person しか残らなかった。
    """
    sys.path.insert(0, os.path.join(RELATE_ROOT, "tools"))
    sys.path.insert(0, RELATE_ROOT)
    import game_scene
    gs_args = game_scene.build_parser().parse_args(
        (["--classes", classes, "--det", "yoloe-11m-seg.pt"] if classes else [])
        + ["--det-conf", str(det_conf)])
    return game_scene.build_pipeline(gs_args)


def to_annotation(res, frame) -> dict:
    """SceneResult → report.py / relate_overlay.py が読む形の dict。"""
    objects = [{"label": str(l), "score": round(float(s), 3),
                "box": [round(float(v), 1) for v in b]}
               for l, s, b in zip(res.labels, res.scores, res.boxes_xyxy)]
    relations = [{"subject": int(si), "predicate": str(pred), "object": int(oi),
                  "score": round(float(score), 3)}
                 for si, pred, oi, score in res.triplets]
    return {"objects": objects, "relations": relations,
            "size": [int(frame.shape[1]), int(frame.shape[0])]}


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--list", help="画像パスを 1 行に 1 つ書いたファイル（枚数が多いとコマンドラインに収まらない）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--det-conf", type=float, default=0.1, help="検出の確信度のしきい値")
    ap.add_argument("--classes", default=",".join(CLASSES),
                    help="検出する物体の語彙（カンマ区切り）。空にすると何でも拾う検出器を使う")
    args = ap.parse_args()

    import cv2
    import numpy as np

    images = list(args.images)
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            images += [line.strip() for line in f if line.strip()]
    pipe = load_pipeline(args.classes, args.det_conf)

    out = {}
    for path in images:
        buf = np.fromfile(path, np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if frame is None:
            out[path] = {"error": "読めない画像"}
            continue
        res = pipe(frame, top_k=args.top_k, score_thr=args.score_thr)
        out[path] = to_annotation(res, frame)
        print(f"{os.path.basename(path)}: 物体 {len(out[path]['objects'])} / "
              f"関係 {len(out[path]['relations'])}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
