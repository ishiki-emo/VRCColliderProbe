"""段階 5: 報告。1 回の走行（runs/<日時>/）を 1 枚の HTML にまとめる。

    uv run tools/report.py runs/20260921_150810

出力: <run_dir>/report.html（地図・画面キャプチャをすべて埋め込んだ 1 ファイル。ブラウザで開くだけで見られる）

載せるもの:
- 概要（時間・歩行距離・踏破面積・各出来事の件数）
- 要確認の地点: 床抜けの疑い、挟まり、判定保留。地点ごとに周辺の地図と直前の画面
- 外周の可能性が高い落下（折りたたみ）
- 全体の地図
map.json がなければ mapper.build で作る。
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapper  # noqa: E402
import semantic  # noqa: E402
import knowledge  # noqa: E402

FRAME_OFFSETS = (-1.2, -0.8, -0.4, 0.0)   # 落ち始めを 0 として載せる画面の時刻 [s]
SEMANTIC_MAP_MAX = 60      # 地図に名前を載せる件数
SEMANTIC_TABLE_MAX = 80    # 一覧表に載せる件数（全件は semantic.json）
CUMULATIVE_OBJECTS_MAX = 80  # 累計の節の「見つけたもの」の表に載せる件数（全件は worlds/<ID>/knowledge.json）
SEMANTIC_THUMBS = 15       # 代表の画面を載せる件数（ページが重くなりすぎないように）
KEY_FRAME_OFFSET = -0.4   # RelateAnything で注釈を付ける画面の時刻（落ち始めが 0）[s]
FRAME_WIDTH = 640          # 埋め込むときの横幅 [px]
MINOR_FRAME_WIDTH = 420    # 外周の落下の画面の横幅 [px]（件数が多いので小さく）
RELATE_ROOT = os.environ.get("RELATE_ANYTHING_ROOT", r"E:\AIProject\RelateAnything")
RELATE_PYTHON = os.environ.get(
    "RELATE_ANYTHING_PYTHON", os.path.join(RELATE_ROOT, ".venv", "Scripts", "python.exe"))
# RelateAnything の検出でアバターとして扱うラベル（三人称視点で自分のアバターが写る）
AVATAR_LABELS = {"person", "girl", "boy", "woman", "man", "doll", "child"}
PREDICATES_JA = {
    "on": "の上にいる", "standing on": "の上に立っている", "walking on": "の上を歩いている",
    "sitting on": "に座っている", "lying on": "の上に寝ている", "in": "の中にいる",
    "near": "の近くにいる", "next to": "の隣にいる", "beside": "のそばにいる",
    "in front of": "の前にいる", "behind": "の後ろにいる", "above": "の上方にいる",
    "under": "の下にいる", "below": "の下方にいる", "over": "の上にかかっている",
    "against": "にもたれている", "leaning on": "にもたれている", "along": "に沿っている",
    "holding": "を持っている", "wearing": "を身につけている", "has": "を持っている",
    "carrying": "を運んでいる", "looking at": "を見ている", "attached to": "に付いている",
    "hanging from": "からぶら下がっている", "covering": "を覆っている", "part of": "の一部",
}
SPATIAL_PREDICATES = {
    "on", "standing on", "walking on", "sitting on", "lying on", "in", "near", "next to", "beside",
    "in front of", "behind", "above", "under", "below", "over", "against", "leaning on", "along",
}
CROP_RADIUS_M = 8.0        # 地点の周りの地図を切り出す半径 [m]
WEDGE_MERGE_M = 2.0        # この距離以内の挟まりは同じ場所として 1 件にまとめる [m]

VERDICTS = {
    # 並び順, 見出し, 説明
    "hole_suspect": (0, "床抜けの疑い", "周りを歩いた床が囲んでいる場所で落ちた。コライダーの張り忘れの可能性が高い"),
    "wedge": (1, "挟まり", "落下状態のまま動けなくなった。コライダーの隙間に引っかかる場所"),
    "unclear": (2, "判定保留", "周りの探索が足りず、外周か床抜けか決められない"),
    "edge_likely": (3, "外周の可能性", "床のある方向が片側だけ。ワールドの端から落ちた可能性が高い"),
    "unaligned": (4, "位置不明", "スポーン地点以外から歩き始めた区間なので、位置が合っていない"),
}


def imread(path: str) -> np.ndarray | None:
    buf = np.fromfile(path, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def data_uri(img: np.ndarray, ext: str = ".jpg", quality: int = 80) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def pick_frames(folder: str, ref_t: float) -> list[tuple[float, str]]:
    """落ち始め（ref_t）の前後の画面を選ぶ。(ref_t からの秒数, パス) のリスト。

    保存した画面は「リスポーンまでの 4 秒」なので、最後の数枚は落下中で何も写っていない。
    落ち始めの 1.2 秒前から直後までを等間隔に選ぶ。
    """
    files = sorted(glob.glob(os.path.join(folder, "t*.jpg")))
    if not files:
        return []
    stamped = [(float(os.path.basename(f)[1:-4]) - ref_t, f) for f in files]
    picked = []
    for target in FRAME_OFFSETS:
        dt, f = min(stamped, key=lambda s: abs(s[0] - target))
        if f not in [p[1] for p in picked]:
            picked.append((dt, f))
    return picked


def frame_uri(path: str, annotation: dict | None = None, width: int = FRAME_WIDTH) -> str:
    img = imread(path)
    if img is None:
        return ""
    if annotation:
        img = draw_annotation(img, annotation)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return data_uri(img, quality=75 if width < FRAME_WIDTH else 80)


def draw_annotation(img: np.ndarray, ann: dict) -> np.ndarray:
    """RelateAnything の検出枠を描く。アバター（person）は太く目立たせる。"""
    img = img.copy()
    sx = img.shape[1] / ann["size"][0]
    sy = img.shape[0] / ann["size"][1]
    me = avatar_index(ann)
    for i, o in enumerate(ann.get("objects", [])):
        x1, y1, x2, y2 = o["box"]
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        avatar = i == me
        color = (0, 220, 255) if avatar else (255, 200, 80)
        cv2.rectangle(img, p1, p2, color, 3 if avatar else 1, cv2.LINE_AA)
        text = o["label"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(p1[1], th + 6)
        cv2.rectangle(img, (p1[0], ty - th - 6), (p1[0] + tw + 6, ty), color, -1)
        cv2.putText(img, text, (p1[0] + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def avatar_index(ann: dict) -> int | None:
    """検出した person のうち、自分のアバターを 1 つ選ぶ。

    ポスターの人物、鏡に映った自分、アバター一覧のサムネイルも person になる（実機で 5 つ出た）。
    三人称視点では自分のアバターは画面の中央・下寄りに写るので、確信度 × 中央への近さで選ぶ。
    """
    W, H = ann.get("size", (1, 1))
    best, best_s = None, 0.05
    for i, o in enumerate(ann.get("objects", [])):
        if o["label"] not in AVATAR_LABELS:
            continue
        x1, y1, x2, y2 = o["box"]
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        centrality = max(0.0, 1.0 - abs(cx - 0.5) * 2.5)
        s = o["score"] * centrality * (1.0 if cy > 0.35 else 0.5)
        if s > best_s:
            best, best_s = i, s
    return best


def describe_relations(ann: dict, limit: int = 6) -> list[str]:
    """関係を日本語の短文にする。アバターが主語の関係を先に並べる。"""
    objs = ann.get("objects", [])
    me = avatar_index(ann)
    rels = []
    for r in ann.get("relations", []):
        if r["subject"] >= len(objs) or r["object"] >= len(objs):
            continue
        s, o = objs[r["subject"]]["label"], objs[r["object"]]["label"]
        # 環境の語彙に「身につける」「持つ」は当てはまらない（sky を身につけている、などが出た）
        if s == o or r["predicate"] not in SPATIAL_PREDICATES:
            continue
        # 自分以外の person（ポスターや鏡の中の人物）が絡む関係は、アバターの状況説明にならない
        if (r["subject"] != me and s in AVATAR_LABELS) or (r["object"] != me and o in AVATAR_LABELS):
            continue
        avatar_first = 0 if r["subject"] == me else 1
        rels.append((avatar_first, -r["score"], s, r["predicate"], o, r["score"]))
    rels.sort()
    # アバターが関わる関係があれば、それだけにする（water は sky の中、のような背景どうしは役に立たない）
    if any(r[0] == 0 for r in rels):
        rels = [r for r in rels if r[0] == 0]
    out, seen = [], set()
    for first, _, s, p, o, score in rels:
        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)
        subj = "アバター" if first == 0 else s
        jp = PREDICATES_JA.get(p)
        text = f"{subj} は {o} {jp}" if jp else f"{subj} —{p}→ {o}"
        out.append(f"{text}（{score:.2f}）")
        if len(out) >= limit:
            break
    return out


def map_crop(map_img: np.ndarray, geo: dict, x: float, y: float, heading: float | None) -> str:
    """地図から地点の周りを切り出し、地点に円（と落ちた向きの矢印）を描く。"""
    s = geo["px_per_m"]
    cx, cy = int((x - geo["x0"]) * s), int((geo["y1"] - y) * s)
    r = int(CROP_RADIUS_M * s)
    pad = cv2.copyMakeBorder(map_img, r, r, r, r, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    crop = pad[cy:cy + 2 * r, cx:cx + 2 * r].copy()
    cv2.circle(crop, (r, r), int(1.2 * s), (0, 255, 255), 2, cv2.LINE_AA)
    if heading is not None:
        h = np.radians(heading)
        tip = (int(r + 2.5 * s * np.sin(h)), int(r - 2.5 * s * np.cos(h)))
        cv2.arrowedLine(crop, (r, r), tip, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    return data_uri(crop, ".png")


def collect_items(run_dir: str, m: dict, map_img: np.ndarray) -> list[dict]:
    geo = m["map_geometry"]
    items = []
    for i, r in enumerate(m["respawns"], 1):
        v = r["verdict"]
        detail = [f"落下時間 {r['fall_time']} 秒" if r.get("fall_time") else None]
        if "octants_with_floor" in r:
            detail.append(f"周り 8 方向のうち床を歩いた方向 {r['octants_with_floor']}")
        if r.get("z") is not None:
            detail.append(f"落ちた地点の高さ {r['z']:+.1f} m（スポーン地点基準）")
        fall_start = r["t"] - (r.get("fall_time") or 1.0)
        items.append(dict(
            id=f"R{i}", verdict=v, t=r["t"], x=r["x"], y=r["y"], heading=r["heading"],
            detail=[d for d in detail if d],
            frames=pick_frames(os.path.join(run_dir, f"respawn_{i:02d}"), fall_start),
            crop=None if v == "unaligned" else map_crop(map_img, geo, r["x"], r["y"], r["heading"]),
        ))
    # 同じ隙間に何度も挟まることがある（実機で同じ場所に 15 回）。2m 以内の挟まりは 1 件にまとめる
    groups: list[list[tuple[int, dict]]] = []
    for i, w in enumerate(m.get("wedges", []), 1):
        aligned = m["started_at_spawn"] or w["track"] > 0
        for g in groups:
            g0 = g[0][1]
            g0_aligned = m["started_at_spawn"] or g0["track"] > 0
            if aligned and g0_aligned and math.hypot(w["x"] - g0["x"], w["y"] - g0["y"]) <= WEDGE_MERGE_M:
                g.append((i, w))
                break
        else:
            groups.append([(i, w)])
    for g in groups:
        i, w = g[0]
        aligned = m["started_at_spawn"] or w["track"] > 0
        # 挟まりは位置が分からなくても（画面で場所が分かるので）要確認に載せる
        detail = (["どの向きに進もうとしても動けなくなった（閉じ込め。岩の隙間などで四方が塞がった）"]
                  if w.get("reason") == "trapped" else [f"落下状態が {w.get('fall_sec')} 秒続いた"])
        if len(g) > 1:
            detail.append(f"同じ場所で {len(g)} 回挟まった（開始から "
                          + "、".join(f"{x['t']:.0f}" for _, x in g[:8]) + (" …" if len(g) > 8 else "") + " 秒）")
        if not aligned:
            detail.append("スポーン地点以外から歩き始めた区間なので、地図上の位置は出せない")
        items.append(dict(
            id=f"W{i}" if len(g) == 1 else f"W{i}–W{g[-1][0]}", verdict="wedge", aligned=aligned,
            t=w["t"], x=w["x"], y=w["y"], heading=None, detail=detail,
            frames=pick_frames(os.path.join(run_dir, f"wedge_{i:02d}"), w["t"] - (w.get("fall_sec") or 3.0)),
            crop=map_crop(map_img, geo, w["x"], w["y"], None) if aligned else None,
        ))
    for it in items:
        # 注釈を付けるのは、落ち始めの少し前（まだ床の上にいる）の 1 枚
        it["key_frame"] = min(it["frames"], key=lambda f: abs(f[0] - KEY_FRAME_OFFSET))[1] \
            if it["frames"] else None
    items.sort(key=lambda it: (VERDICTS[it["verdict"]][0], it["t"]))
    return items


def annotate(run_dir: str, paths: list[str]) -> tuple[dict, str | None]:
    """RelateAnything で注釈を付ける。結果は scene.json にキャッシュする。(注釈, 使えなかった理由)"""
    cache_path = os.path.join(run_dir, "scene.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    rel = {p: os.path.relpath(p, run_dir) for p in paths}
    missing = [p for p in paths if rel[p] not in cache]
    reason = None
    if missing:
        if not os.path.exists(RELATE_PYTHON):
            reason = f"RelateAnything の環境が見つからない（{RELATE_PYTHON}）"
        else:
            tmp = os.path.abspath(os.path.join(run_dir, "scene_new.json"))
            list_path = os.path.abspath(os.path.join(run_dir, "scene_list.txt"))
            # RelateAnything は RelateAnything のフォルダで動かす（このプロジェクトで動かすと、検出器が
            # Apple MobileCLIP の重み mobileclip_blt.ts を作業フォルダにダウンロードしてしまう）ので、パスは絶対パスで渡す
            by_abs = {os.path.abspath(p): p for p in missing}
            with open(list_path, "w", encoding="utf-8") as f:
                f.write("\n".join(by_abs))
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_annotate.py")
            print(f"RelateAnything で {len(missing)} 枚に注釈を付ける…", flush=True)
            proc = subprocess.run([RELATE_PYTHON, script, "--out", tmp, "--list", list_path], cwd=RELATE_ROOT,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            os.remove(list_path)
            if proc.returncode == 0 and os.path.exists(tmp):
                new = json.load(open(tmp, encoding="utf-8"))
                os.remove(tmp)
                cache.update({rel[by_abs[p]]: a for p, a in new.items() if p in by_abs})
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            else:
                reason = "RelateAnything の実行に失敗: " + (proc.stderr or "").strip().splitlines()[-1:][0] \
                    if (proc.stderr or "").strip() else "RelateAnything の実行に失敗"
    return {p: cache[rel[p]] for p in paths if rel[p] in cache}, reason


def item_html(it: dict, notes: dict) -> str:
    order, label, desc = VERDICTS[it["verdict"]]
    pos = "" if it["verdict"] == "unaligned" or not it.get("aligned", True) else \
        f"スポーン地点から 右 {it['x']:+.1f} m / 前 {it['y']:+.1f} m"
    ann = notes.get(it["key_frame"]) if it["key_frame"] else None
    figs = []
    # 外周の落下は数が多い（30 分で 60 件以上）ので、落ち始めの直前と落ち始めの 2 枚だけを小さく載せる
    minor = it["verdict"] in ("edge_likely", "unaligned")
    frames_shown = [f for f in it["frames"] if f[0] >= -0.6] if minor else it["frames"]
    for dt, path in frames_shown:
        is_key = ann is not None and path == it["key_frame"]
        when = "落ち始め" if abs(dt) < 0.1 else f"落ち始めの {-dt:.1f} 秒前" if dt < 0 else f"落ち始めの {dt:.1f} 秒後"
        cap = when + ("・RelateAnything の検出枠" if is_key else "")
        uri = frame_uri(path, ann if is_key else None, width=MINOR_FRAME_WIDTH if minor else FRAME_WIDTH)
        figs.append(f'<figure{" class=key" if is_key else ""}><img src="{uri}" '
                    f'alt="{it["id"]} の{when}の画面" loading="lazy"><figcaption>{cap}</figcaption></figure>')
    frames = "".join(figs) or '<p class="muted">画面キャプチャなし</p>'
    scene = ""
    if ann is not None:
        rels = describe_relations(ann)
        labels = sorted({o["label"] for o in ann.get("objects", [])})
        scene = ('<div class="scene"><b>画面から読み取った状況</b>（RelateAnything）'
                 + ("<ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in rels) + "</ul>" if rels else
                    "<p class='muted'>関係は見つからなかった</p>")
                 + (f"<p class='muted'>写っているもの: {html.escape(', '.join(labels))}</p>" if labels else "")
                 + "</div>")
    crop = (f'<figure class="crop"><img src="{it["crop"]}" alt="{it["id"]} の周辺の地図">'
            f"<figcaption>周辺 {int(CROP_RADIUS_M * 2)} m 四方（黄色の円が地点"
            + ("、矢印が落ちた向き" if it["heading"] is not None else "") + "）</figcaption></figure>"
            if it["crop"] else "")
    detail = "".join(f"<li>{html.escape(d)}</li>" for d in it["detail"])
    return f"""
<article class="item v-{it['verdict']}">
  <header>
    <span class="tag">{label}</span>
    <h3>{it['id']}</h3>
    <span class="meta">開始から {it['t']:.1f} 秒 {('・' + pos) if pos else ''}</span>
  </header>
  <p class="desc">{html.escape(desc)}</p>
  <ul class="detail">{detail}</ul>
  {scene}
  <div class="media">{crop}<div class="frames">{frames}</div></div>
</article>"""


LANDMARK_COLORS = {"surface": (180, 200, 60), "nearby": (200, 120, 255)}   # BGR: 足元 = 青緑、近く = ピンク


def semantic_map(map_img: np.ndarray, geo: dict, landmarks: list[dict]) -> str:
    """地図に「どこに何があるか」のラベルを重ねた画像。"""
    img = (map_img * 0.7).astype(np.uint8)       # 軌跡を少し沈めてラベルを目立たせる
    s = geo["px_per_m"]
    for m in landmarks[:SEMANTIC_MAP_MAX]:
        p = (int((m["x"] - geo["x0"]) * s), int((geo["y1"] - m["y"]) * s))
        color = LANDMARK_COLORS[m["kind"]]
        if m["kind"] == "surface":
            cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, p, color, cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
        text = m["label"]
        cv2.putText(img, text, (p[0] + 8, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, text, (p[0] + 8, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return data_uri(img, ".png")


def semantic_html(run_dir: str, sem: dict | None, map_img: np.ndarray, geo: dict) -> str:
    if not sem:
        return "<p class='muted'>この走行には「どこに何があるか」のデータがありません。</p>"
    marks = sem.get("landmarks", [])
    src = ("走行中に 1 秒ごとに保存した画面" if sem.get("source") == "frames"
           else "落下や挟まりの前後に保存した画面（この走行は 1 秒ごとの画面がないので、場所が偏っています）")
    head = (f"<p>{html.escape(src)} {sem.get('frames', 0)} 枚のうち、位置が分かる {sem.get('frames_used', 0)} 枚を "
            f"RelateAnything で調べ、{len(marks)} 件にまとめました。</p>")
    if sem.get("problem"):
        head += f"<p class='note'>{html.escape(sem['problem'])}</p>"
    if not marks:
        return head + "<p class='muted'>見つかったものはありません。</p>"
    rows = []
    for i, m in enumerate(marks[:SEMANTIC_TABLE_MAX]):
        thumb = ""
        if i < SEMANTIC_THUMBS and m.get("best_frame"):
            path = os.path.join(run_dir, m["best_frame"])
            if os.path.exists(path):
                img = imread(path)
                h, w = img.shape[:2]
                thumb = f'<img class="thumb" src="{data_uri(cv2.resize(img, (200, int(h * 200 / w))))}" alt="">'
        kind = "足元" if m["kind"] == "surface" else "近く"
        rows.append(
            f"<tr><td><span class='lm lm-{m['kind']}'></span>{html.escape(m['label_ja'])} "
            f"<span class='muted'>{html.escape(m['label'])}</span></td><td>{kind}</td>"
            f"<td>右 {m['x']:+.1f} m / 前 {m['y']:+.1f} m</td><td>{m['z']:+.1f} m</td>"
            f"<td>{m['frames']}</td><td>{thumb}</td></tr>")
    more = (f"<p class='muted small'>ほか {len(marks) - SEMANTIC_TABLE_MAX} 件は semantic.json にあります。</p>"
            if len(marks) > SEMANTIC_TABLE_MAX else "")
    return f"""{head}
  <div class="map"><img src="{semantic_map(map_img, geo, marks)}" alt="どこに何があるかの地図"></div>
  <ul class="legend"><li><span class="lm lm-surface"></span>足元（その場所の床の種類）</li>
    <li><span class="lm lm-nearby"></span>近くにあるもの</li><li>地図上の文字は英語のラベル（下の表に日本語）</li></ul>
  <div class="table-wrap"><table class="lm-table">
    <thead><tr><th>もの</th><th>種類</th><th>位置（スポーン地点から）</th><th>高さ</th><th>見えた枚数</th><th>代表の画面</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>{more}
  <p class="muted small">位置は、画面を撮ったときのアバターの推定位置です（数 m の誤差があります）。「近く」は、
  画面上で写っている方角へ {semantic.NEARBY_OFFSET_M} m ずらしています。1 枚でしか見えていない弱い検出は除いています。</p>"""


def cumulative_map(agg: dict, falls: list[dict] | None = None, places: list[list[dict]] | None = None,
                   px_per_m: float = 12.0) -> tuple[np.ndarray, dict]:
    """全走行を重ねた地図を描く。(画像, 座標の対応 {x0, y1, px_per_m})。GUI（gui.py）でも使う。

    画像のピクセル (px, py) と地図の座標 (x, y) は px = (x - x0) * s、py = (y1 - y) * s で対応する。
    """
    cell = knowledge.CELL
    visited = agg["visited"]
    falls = falls if falls is not None else [{**f, "verdict": f.get("verdict", "edge_likely")} for f in agg["falls"]]
    places = places if places is not None else wedge_places(agg["wedges"])
    xs = [(c[0] + 0.5) * cell for c in visited] + [f["x"] for f in falls] + [0.0]
    ys = [(c[1] + 0.5) * cell for c in visited] + [f["y"] for f in falls] + [0.0]
    s, margin = px_per_m, 3.0
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    img = np.full((int((y1 - y0) * s), int((x1 - x0) * s), 3), 32, np.uint8)
    P = lambda x, y: (int((x - x0) * s), int((y1 - y) * s))  # noqa: E731
    for g in range(math.ceil(x0 / 5) * 5, int(x1) + 1, 5):
        cv2.line(img, P(g, y0), P(g, y1), (52, 52, 50), 1)
    for g in range(math.ceil(y0 / 5) * 5, int(y1) + 1, 5):
        cv2.line(img, P(x0, g), P(x1, g), (52, 52, 50), 1)
    for cx, cy in visited:
        cv2.rectangle(img, P(cx * cell, (cy + 1) * cell), P((cx + 1) * cell, cy * cell), (70, 110, 70), -1)
    for m in [m for m in agg["landmarks"] if m["kind"] == "nearby" and m["frames"] >= 3][:SEMANTIC_MAP_MAX]:
        p = P(m["x"], m["y"])
        cv2.drawMarker(img, p, LANDMARK_COLORS["nearby"], cv2.MARKER_DIAMOND, 9, 1, cv2.LINE_AA)
        cv2.putText(img, m["label"], (p[0] + 6, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    LANDMARK_COLORS["nearby"], 1, cv2.LINE_AA)
    for c in agg["climbs"]:
        cv2.drawMarker(img, P(c["x"], c["y"]), (80, 230, 80), cv2.MARKER_TRIANGLE_UP, 9, 2)
    for p in places:
        cv2.drawMarker(img, P(p[0]["x"], p[0]["y"]), (255, 0, 255), cv2.MARKER_SQUARE, 12, 2)
    colors = {"hole_suspect": (0, 0, 255), "edge_likely": (0, 140, 255), "unclear": (200, 0, 200)}
    for f in sorted(falls, key=lambda f: f["verdict"] == "hole_suspect"):   # 床抜けの疑いを最後（上）に描く
        cv2.drawMarker(img, P(f["x"], f["y"]), colors.get(f["verdict"], (150, 150, 150)),
                       cv2.MARKER_TILTED_CROSS, 16 if f["verdict"] == "hole_suspect" else 10, 2)
    cv2.drawMarker(img, P(0, 0), (240, 240, 240), cv2.MARKER_STAR, 14, 1)
    return img, {"x0": x0, "y1": y1, "px_per_m": s, "width": img.shape[1], "height": img.shape[0]}


def wedge_places(wedges: list[dict]) -> list[list[dict]]:
    """挟まりを場所ごと（WEDGE_MERGE_M 以内）にまとめる。"""
    places: list[list[dict]] = []
    for w in wedges:
        for p in places:
            if math.hypot(w["x"] - p[0]["x"], w["y"] - p[0]["y"]) <= WEDGE_MERGE_M:
                p.append(w)
                break
        else:
            places.append([w])
    return places


def cumulative_html(k: dict, agg: dict, run_name: str) -> str:
    """このワールドの累計（knowledge.py）。全走行を重ねた地図と、累計で判定した落下地点。"""
    cell = knowledge.CELL
    visited = agg["visited"]
    if not visited:
        return "<p class='muted'>このワールドの記録はまだありません。</p>"
    # 落下の判定は各走行の中で行ったもの（knowledge.contribution）。累計の床で判定し直すと、
    # 走行ごとの位置のずれで外周の落下が床抜けの疑いに見える
    falls = [{**f, "verdict": f.get("verdict", "edge_likely")} for f in agg["falls"]]
    places = wedge_places(agg["wedges"])   # 挟まりは場所ごとにまとめる
    img, _ = cumulative_map(agg, falls, places)

    holes = [f for f in falls if f["verdict"] == "hole_suspect"]
    stats = [("走行", f"{agg['runs']} 回"), ("累計の踏破面積", f"{len(visited) * cell * cell:.0f} m²"),
             ("落下", f"{len(falls)} 回"), ("床抜けの疑い", f"{len(holes)} か所"),
             ("挟まる場所", f"{len(places)} か所"), ("ジャンプで越えた", f"{len(agg['climbs'])} 回"),
             ("見つけたもの", f"{len(agg['landmarks'])} 件")]
    stat_html = "".join(f'<div class="stat"><dt>{a}</dt><dd>{b}</dd></div>' for a, b in stats)
    hole_html = ""
    if holes:
        rows = "".join(f"<li>右 {f['x']:+.1f} m / 前 {f['y']:+.1f} m（走行 "
                       f"<a href='../{html.escape(f['run'])}/report.html'>{html.escape(f['run'])}</a>）</li>"
                       for f in holes)
        hole_html = f"<p class='note'>各走行の判定で、床抜けの疑いがある落下地点です。</p><ul>{rows}</ul>"
    # 走行をまたいでまとめた「見つけたもの」。複数の走行で見えたものほど確か
    lm_rows = "".join(
        f"<tr><td><span class='lm lm-{m['kind']}'></span>{html.escape(m.get('label_ja', m['label']))} "
        f"<span class='muted'>{html.escape(m['label'])}</span></td>"
        f"<td>{'足元' if m['kind'] == 'surface' else '近く'}</td>"
        f"<td>右 {m['x']:+.1f} m / 前 {m['y']:+.1f} m</td><td>{m.get('z', 0):+.1f} m</td>"
        f"<td>{m['runs']}</td><td>{m['frames']}</td></tr>"
        for m in agg["landmarks"][:CUMULATIVE_OBJECTS_MAX])
    lm_table = (f"<h3 class='sub-h'>見つけたもの（全走行・{len(agg['landmarks'])} 件のうち上位 "
                f"{min(len(agg['landmarks']), CUMULATIVE_OBJECTS_MAX)} 件）</h3>"
                "<div class='table-wrap'><table class='lm-table'><thead><tr><th>もの</th><th>種類</th>"
                "<th>位置（スポーン地点から）</th><th>高さ</th><th>見えた走行</th><th>見えた回数</th></tr></thead>"
                f"<tbody>{lm_rows}</tbody></table></div>") if agg["landmarks"] else ""
    runs = "".join(
        f"<li>{'<b>' if name == run_name else ''}<a href='../{html.escape(name)}/report.html'>{html.escape(name)}</a>"
        f"{'（この走行）</b>' if name == run_name else ''} ・ 歩行 {r['distance_m']:.0f} m ・ 踏破 "
        f"{len(r['visited']) * cell * cell:.0f} m² ・ 落下 {len(r['falls'])}</li>"
        for name, r in sorted(k["runs"].items(), reverse=True))
    return f"""
  <p>{html.escape(k['world']['name'] or '（名前不明のワールド）')} <span class="muted">{html.escape(k['world']['id'])}</span>
  の、スポーン地点から始めた走行をすべて重ねた記録です。次の走行は、ここで歩いていない場所を優先し、
  過去に落ちた・挟まった場所を避けて歩きます。位置の推定は走行ごとに数 m ずれるので、重ねた地図は少しぼやけています。
  落下の判定（外周か床抜けか）は、各走行の中で行ったものです。</p>
  <dl class="stats">{stat_html}</dl>
  {hole_html}
  <div class="map"><img src="{data_uri(img, '.png')}" alt="全走行を重ねた地図"></div>
  <ul class="legend"><li>緑: 歩いた場所（全走行）</li><li>赤い ×: 床抜けの疑い</li><li>橙の ×: 外周の可能性</li>
    <li>紫の ×: 判定保留</li><li>赤紫の □: 挟まる場所</li><li>緑の▲: ジャンプで越えた段差</li>
    <li>ピンクの◇: 見つけたもの（3 回以上）</li></ul>
  {lm_table}
  <details><summary>走行の一覧（{len(k['runs'])} 回）</summary><ul class="small">{runs}</ul></details>"""


def view3d_data(m: dict, landmarks: list[dict] | None = None) -> dict | None:
    """3D ビュー用のデータ。地図に載せる区間（位置が合っている区間）だけを使う。"""
    if "trail_z" not in m:
        return None
    ok = lambda tr: m["started_at_spawn"] or tr > 0  # noqa: E731
    pts = []
    for (tr, x, y, g), z in zip(m["trail"], m["trail_z"]):
        if ok(tr):
            pts.append([tr, round(x, 2), round(y, 2), round(z, 2), 1 if g else 0])
    # 3600 点/3 分ほどあるので、近すぎる点を間引く
    thin, last = [], None
    for p in pts:
        if last is None or p[0] != last[0] or p[4] != last[4] or \
                (p[1] - last[1]) ** 2 + (p[2] - last[2]) ** 2 + (p[3] - last[3]) ** 2 > 0.09:
            thin.append(p)
            last = p
    marks = []
    for i, r in enumerate(m["respawns"], 1):
        if r["verdict"] != "unaligned":
            marks.append({"kind": r["verdict"], "label": f"R{i}", "x": r["x"], "y": r["y"], "z": r.get("z", 0)})
    for i, w in enumerate(m.get("wedges", []), 1):
        if ok(w["track"]):
            marks.append({"kind": "wedge", "label": f"W{i}", "x": w["x"], "y": w["y"], "z": w.get("z", 0)})
    for lm in (landmarks or [])[:SEMANTIC_MAP_MAX]:
        marks.append({"kind": "lm_" + lm["kind"], "label": lm["label_ja"], "x": lm["x"], "y": lm["y"], "z": lm["z"]})
    walls = [p[1:] for p in m.get("wall_points", []) if ok(p[0])]
    drops = [p[1:] for p in m.get("drop_points", []) if ok(p[0])]
    climbs = [p[1:] for p in m.get("climb_points", []) if ok(p[0])]
    return {"trail": thin, "marks": marks, "walls": walls, "drops": drops, "climbs": climbs}


VIEW3D_JS = r"""
(() => {
  const D = JSON.parse(document.getElementById('v3d-data').textContent);
  const cv = document.getElementById('v3d'), ctx = cv.getContext('2d');
  const exEl = document.getElementById('v3d-ex'), exOut = document.getElementById('v3d-ex-out');
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const zs = D.trail.filter(p => p[4]).map(p => p[3]);
  const zLo = zs.length ? Math.min(...zs) : 0, zHi = zs.length ? Math.max(...zs) : 1;
  const span = Math.max(zHi - zLo, 0.5);
  // 高さの色: 低い = 青緑、高い = 黄（2D 地図と同じ向き）
  const stops = [[0, [70, 140, 200]], [0.5, [90, 190, 130]], [1, [230, 200, 80]]];
  const hcol = z => {
    const t = Math.min(1, Math.max(0, (z - zLo) / span));
    for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i], k = (t - t0) / (t1 - t0);
      return `rgb(${c0.map((v, j) => Math.round(v + (c1[j] - v) * k)).join(',')})`;
    }
    return `rgb(${stops[stops.length - 1][1].join(',')})`;
  };
  let xs = D.trail.map(p => p[1]), ys = D.trail.map(p => p[2]);
  const cx = (Math.min(...xs, 0) + Math.max(...xs, 0)) / 2, cy = (Math.min(...ys, 0) + Math.max(...ys, 0)) / 2;
  const radius = Math.max(10, ...D.trail.map(p => Math.hypot(p[1] - cx, p[2] - cy)));
  let yaw = -0.6, pitch = 0.75, dist = radius * 2.6, ex = +exEl.value;
  // 地図の座標 (x: 右, y: 前, z: 上) → カメラ座標
  function project(x, y, z) {
    const X = x - cx, Y = y - cy, Z = z * ex;
    const x1 = X * Math.cos(yaw) - Y * Math.sin(yaw), y1 = X * Math.sin(yaw) + Y * Math.cos(yaw);
    const depth = y1 * Math.cos(pitch) + Z * Math.sin(pitch) + dist;
    const up = Z * Math.cos(pitch) - y1 * Math.sin(pitch);
    const f = cv.clientHeight * 1.1 / Math.max(depth, 0.1);
    return [cv.clientWidth / 2 + x1 * f, cv.clientHeight / 2 - up * f, depth];
  }
  function draw() {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css('--v3d-bg'); ctx.fillRect(0, 0, W, H);
    // 高さ 0（スポーン地点）の格子
    ctx.strokeStyle = css('--v3d-grid'); ctx.lineWidth = 1;
    const g = Math.ceil(radius / 5) * 5;
    for (let v = -g; v <= g; v += 5) {
      let a = project(cx + v, cy - g, 0), b = project(cx + v, cy + g, 0);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      a = project(cx - g, cy + v, 0); b = project(cx + g, cy + v, 0);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    // 軌跡を線分に分けて、奥から描く
    const segs = [];
    for (let i = 1; i < D.trail.length; i++) {
      const p = D.trail[i - 1], q = D.trail[i];
      if (p[0] !== q[0]) continue;
      const a = project(p[1], p[2], p[3]), b = project(q[1], q[2], q[3]);
      segs.push([(a[2] + b[2]) / 2, a, b, q[4], (p[3] + q[3]) / 2]);
    }
    segs.sort((s, t) => t[0] - s[0]);
    for (const [, a, b, grounded, z] of segs) {
      ctx.strokeStyle = grounded ? hcol(z) : css('--v3d-air');
      ctx.lineWidth = grounded ? 3 : 1;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    ctx.fillStyle = css('--v3d-wall');
    for (const [x, y, z] of D.walls) { const p = project(x, y, z); ctx.fillRect(p[0] - 2, p[1] - 2, 4, 4); }
    ctx.strokeStyle = '#f0c000';
    for (const [x, y, z] of D.drops) { const p = project(x, y, z); ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.stroke(); }
    ctx.fillStyle = '#50e650';
    for (const [x, y, z] of (D.climbs || [])) { const p = project(x, y, z); ctx.beginPath(); ctx.moveTo(p[0], p[1] - 6); ctx.lineTo(p[0] - 5, p[1] + 4); ctx.lineTo(p[0] + 5, p[1] + 4); ctx.fill(); }
    const colors = { hole_suspect: css('--hole'), edge_likely: css('--edge'), unclear: css('--unclear'), wedge: css('--wedge') };
    ctx.font = '12px system-ui'; ctx.lineWidth = 2.5;
    // 「どこに何があるか」は小さな丸と名前（落下地点より控えめに）
    ctx.font = '11px system-ui';
    for (const m of D.marks.filter(m => m.kind.startsWith('lm_'))) {
      const p = project(m.x, m.y, m.z), c = m.kind === 'lm_surface' ? '#3cc8b4' : '#ff78c8';
      ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p[0], p[1], 3.5, 0, 7); ctx.fill();
      ctx.fillText(m.label, p[0] + 6, p[1] + 4);
    }
    ctx.font = '12px system-ui';
    for (const m of D.marks.filter(m => !m.kind.startsWith('lm_'))) {
      const p = project(m.x, m.y, m.z), c = colors[m.kind] || '#888';
      const base = project(m.x, m.y, zLo - 0.5);
      ctx.strokeStyle = c; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(base[0], base[1]); ctx.stroke();
      ctx.setLineDash([]); ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(p[0] - 6, p[1] - 6); ctx.lineTo(p[0] + 6, p[1] + 6);
      ctx.moveTo(p[0] + 6, p[1] - 6); ctx.lineTo(p[0] - 6, p[1] + 6); ctx.stroke();
      ctx.fillStyle = c; ctx.fillText(m.label, p[0] + 8, p[1] - 8);
    }
    const s = project(0, 0, 0);
    ctx.fillStyle = '#ecebe6'; ctx.beginPath(); ctx.arc(s[0], s[1], 5, 0, 7); ctx.fill();
    ctx.fillText('スポーン', s[0] + 8, s[1] + 14);
  }
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!drag) return;
    yaw = drag[2] + (e.clientX - drag[0]) * 0.008;
    pitch = Math.min(1.55, Math.max(0.05, drag[3] + (e.clientY - drag[1]) * 0.008));
    draw();
  });
  cv.addEventListener('pointerup', () => drag = null);
  cv.addEventListener('wheel', e => { e.preventDefault(); dist = Math.min(radius * 8, Math.max(radius * 0.6, dist * Math.exp(e.deltaY * 0.001))); draw(); }, { passive: false });
  exEl.addEventListener('input', () => { ex = +exEl.value; exOut.textContent = '×' + ex; draw(); });
  document.getElementById('v3d-reset').addEventListener('click', () => { yaw = -0.6; pitch = 0.75; dist = radius * 2.6; draw(); });
  new ResizeObserver(draw).observe(cv);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  document.getElementById('v3d-range').textContent = `${zLo.toFixed(1)} m 〜 ${zHi.toFixed(1)} m`;
  draw();
})();
"""


def view3d_html(data: dict | None) -> str:
    if not data or len(data["trail"]) < 2:
        return "<p class='muted'>高さのデータがないので 3D 表示はできません（古い走行です）。</p>"
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""
  <div class="v3d-wrap">
    <canvas id="v3d" aria-label="歩いた軌跡の 3D 表示。ドラッグで回転、ホイールで拡大縮小"></canvas>
    <div class="v3d-ctrl">
      <label>高さの強調 <input id="v3d-ex" type="range" min="1" max="10" step="1" value="3"> <output id="v3d-ex-out">×3</output></label>
      <button id="v3d-reset" type="button">視点を戻す</button>
      <span class="muted">ドラッグで回転・ホイールで拡大縮小・高さ <span id="v3d-range"></span>（スポーン地点が 0）</span>
    </div>
  </div>
  <script type="application/json" id="v3d-data">{payload}</script>
  <script>{VIEW3D_JS}</script>"""


def build(run_dir: str, use_relate: bool = True) -> str:
    run_dir = os.path.normpath(run_dir)
    run_name = os.path.basename(run_dir)
    start = {}
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "start":
                start = e
                break
    # ワールドの記録（knowledge.py）。床抜けの判定には使わない: 位置の推定は走行ごとに数 m ずれるので、
    # 他の走行の軌跡を重ねると外周の外側にまで「床」ができ、外周の落下が床抜けの疑いに見える（実機で 25 か所）
    kworld = knowledge.run_world(run_dir) if start.get("use_knowledge", True) else None
    map_json = os.path.join(run_dir, "map.json")
    if not os.path.exists(map_json) or "map_geometry" not in json.load(open(map_json, encoding="utf-8")):
        mapper.build(run_dir)
    m = json.load(open(map_json, encoding="utf-8"))
    summary_path = os.path.join(run_dir, "summary.json")
    summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else {}

    map_img = imread(os.path.join(run_dir, "map.png"))
    items = collect_items(run_dir, m, map_img)
    scene_notes, relate_problem = ({}, None)
    sem = None
    sem_path = os.path.join(run_dir, "semantic.json")
    if use_relate:
        scene_notes, relate_problem = annotate(run_dir, [it["key_frame"] for it in items if it["key_frame"]])
        if not relate_problem:
            sem = semantic.build(run_dir, annotate=lambda ps: annotate(run_dir, ps))
    elif os.path.exists(sem_path):
        sem = json.load(open(sem_path, encoding="utf-8"))
    # この走行をワールドの記録に取り込み（同じ走行は置き換え）、累計を作る
    cumulative = "<p class='muted'>ワールドが分からない（または --fresh で走った）ので、累計はありません。</p>"
    if kworld:
        added = knowledge.add_run(run_dir, kworld)
        k_all = added[0] if added else knowledge.load(kworld["id"])
        cumulative = cumulative_html(k_all, knowledge.aggregate(k_all), run_name)
    counts = {k: sum(1 for it in items if it["verdict"] == k) for k in VERDICTS}
    attention = [it for it in items if it["verdict"] in ("hole_suspect", "wedge", "unclear")]
    edges = [it for it in items if it["verdict"] in ("edge_likely", "unaligned")]

    name = os.path.basename(run_dir)
    try:
        started = time.strftime("%Y-%m-%d %H:%M", time.strptime(name, "%Y%m%d_%H%M%S"))
    except ValueError:
        started = name
    stop_reasons = {"duration": "指定時間が経過", "user": "手動で停止", "avatar_change": "ワールド移動を検知",
                    "no_response": "操作が効かなくなった", "wedged": "挟まりから抜け出せなかった"}
    elapsed = summary.get("elapsed")
    stats = [
        ("実行時間", f"{elapsed:.0f} 秒" if elapsed else "—"),
        ("歩行距離", f"{m['distance_m']:.0f} m"),
        ("踏破面積", f"{m['walked_area_m2']:.0f} m²"),
        ("壁に詰まった", f"{m['walls']} 回"),
        ("段差", f"{m['drops']} 回"),
        ("ジャンプで越えた", f"{m.get('climbs', 0)} 回"),
        ("落下してリスポーン", f"{len(m['respawns'])} 回"),
    ]
    shown = ["hole_suspect", "wedge", "unclear", "edge_likely"] + (["unaligned"] if counts["unaligned"] else [])
    verdict_line = " ・ ".join(
        f'<span class="v-{k}"><b>{counts[k]}</b> {VERDICTS[k][1]}</span>' for k in shown)
    headline = ("床抜けの疑いがある地点が見つかりました" if counts["hole_suspect"]
                else "コライダーの隙間に挟まる地点が見つかりました" if counts["wedge"]
                else "床抜けの疑いがある地点は見つかりませんでした")
    notes = []
    if not m["started_at_spawn"]:
        notes.append("スポーン地点以外から歩き始めたので、最初のリスポーンまでの区間は地図に載せていません。")
    if summary.get("stop_reason") not in (None, "duration"):
        notes.append(f"走行は途中で止まりました（{stop_reasons.get(summary['stop_reason'], summary['stop_reason'])}）。")
    if relate_problem:
        notes.append(f"画面の注釈を付けられませんでした: {relate_problem}")

    stat_html = "".join(f'<div class="stat"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in stats)
    attention_html = "".join(item_html(it, scene_notes) for it in attention) or \
        '<p class="muted">要確認の地点はありません。</p>'
    edges_html = "".join(item_html(it, scene_notes) for it in edges)
    notes_html = "".join(f"<p class='note'>{html.escape(n)}</p>" for n in notes)
    map_uri = data_uri(map_img, ".png")

    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>床抜けチェック報告 {html.escape(started)}</title>
<style>
:root {{
  --bg: #f6f5f2; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6a66; --line: #e2e0da;
  --hole: #d6332a; --wedge: #b8338f; --unclear: #7a55c9; --edge: #d9822b; --unaligned: #8a8a86;
  --v3d-bg: #2a2a27; --v3d-grid: #45443e; --v3d-air: rgba(200,200,200,0.45); --v3d-wall: #e8e6df;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #171716; --panel: #22221f; --text: #ecebe6; --muted: #a09e97; --line: #36352f;
    --v3d-bg: #1c1c1a; --v3d-grid: #34332e; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 "Hiragino Sans", "Yu Gothic UI", "Meiryo", system-ui, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 19px; margin: 40px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }}
h3 {{ font-size: 17px; margin: 0; }}
.sub, .muted, .meta, figcaption {{ color: var(--muted); }}
.headline {{ font-size: 18px; margin: 20px 0 6px; }}
.verdicts span {{ white-space: nowrap; }}
.verdicts b {{ font-size: 18px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 20px 0 0; }}
.stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; }}
.stat dt {{ font-size: 12px; color: var(--muted); }}
.stat dd {{ margin: 2px 0 0; font-size: 20px; font-weight: 600; }}
.note {{ background: var(--panel); border-left: 3px solid var(--edge); padding: 8px 12px; margin: 12px 0 0; }}
.item {{ background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--c);
  border-radius: 8px; padding: 14px 16px; margin: 0 0 14px; }}
.item header {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }}
.tag {{ background: var(--c); color: #fff; font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 99px; }}
.desc {{ margin: 8px 0 2px; }}
.detail {{ margin: 0 0 10px; padding-left: 20px; color: var(--muted); font-size: 14px; }}
.media {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.frames {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; flex: 1 1 420px; }}
figure {{ margin: 0; }}
figure img {{ width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line); }}
figure.crop {{ flex: 0 0 220px; }}
figure.key img {{ border: 2px solid #f0b400; }}
.scene {{ background: var(--bg); border-radius: 6px; padding: 8px 12px; margin: 0 0 10px; font-size: 14px; }}
.scene ul {{ margin: 4px 0; padding-left: 20px; }}
.lm {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: 0; }}
.lm-surface {{ background: #3cc8b4; border-radius: 50%; }}
.lm-nearby {{ background: #ff78c8; transform: rotate(45deg) scale(0.85); }}
.table-wrap {{ overflow-x: auto; margin-top: 12px; }}
.sub-h {{ font-size: 16px; margin: 24px 0 0; }}
.lm-table {{ border-collapse: collapse; width: 100%; font-size: 14px; background: var(--panel); }}
.lm-table th, .lm-table td {{ border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left;
  vertical-align: middle; white-space: nowrap; }}
.lm-table th {{ font-size: 12px; color: var(--muted); font-weight: 600; }}
.lm-table td:nth-child(5) {{ text-align: right; }}
.thumb {{ width: 140px; border-radius: 4px; display: block; }}
.scene p {{ margin: 2px 0 0; font-size: 13px; }}
figcaption {{ font-size: 12px; margin-top: 2px; }}
.v-hole_suspect {{ --c: var(--hole); }} .v-wedge {{ --c: var(--wedge); }} .v-unclear {{ --c: var(--unclear); }}
.v-edge_likely {{ --c: var(--edge); }} .v-unaligned {{ --c: var(--unaligned); }}
.verdicts .v-hole_suspect b {{ color: var(--hole); }} .verdicts .v-wedge b {{ color: var(--wedge); }}
.verdicts .v-unclear b {{ color: var(--unclear); }} .verdicts .v-edge_likely b {{ color: var(--edge); }}
.verdicts .v-unaligned b {{ color: var(--unaligned); }}
details > summary {{ cursor: pointer; font-weight: 600; margin: 0 0 12px; }}
.map img {{ max-width: 100%; border-radius: 8px; border: 1px solid var(--line); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 13px; color: var(--muted); margin: 8px 0 0; padding: 0; list-style: none; }}
footer {{ margin-top: 48px; font-size: 13px; color: var(--muted); }}
.small {{ font-size: 13px; }}
.v3d-wrap canvas {{ width: 100%; height: 440px; display: block; border-radius: 8px; border: 1px solid var(--line);
  touch-action: none; cursor: grab; }}
.v3d-wrap canvas:active {{ cursor: grabbing; }}
.v3d-ctrl {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; margin-top: 8px; font-size: 13px; }}
.v3d-ctrl input {{ vertical-align: middle; }}
.v3d-ctrl button {{ font: inherit; padding: 3px 12px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--panel); color: var(--text); cursor: pointer; }}
@media (max-width: 600px) {{ figure.crop {{ flex-basis: 100%; }} }}
</style>
</head>
<body>
<main>
  <h1>床抜けチェック報告</h1>
  <div class="sub">{html.escape(started)} 開始 ・ 探索方式 {html.escape(str(start.get('strategy', '—')))} ・ {html.escape(name)}</div>
  <p class="headline">{headline}</p>
  <div class="verdicts">{verdict_line}</div>
  <dl class="stats">{stat_html}</dl>
  {notes_html}

  <h2>要確認の地点</h2>
  {attention_html}

  <h2>全体の地図</h2>
  <div class="map"><img src="{map_uri}" alt="歩いた軌跡と出来事の地図"></div>
  <ul class="legend">
    <li>1 マス = 1 m（太線は 5 m ごと）・上がスポーン時の正面</li><li>緑: 歩いた場所</li><li>白い点: 壁</li>
    <li>黄色の輪: 段差</li><li>緑の▲: ジャンプで越えた段差</li><li>赤い ×: 床抜けの疑い</li><li>橙の ×: 外周の可能性</li>
    <li>紫の ×: 判定保留</li><li>赤紫の □: 挟まり</li>
  </ul>

  <h2>このワールドの累計</h2>
  {cumulative}

  <h2>どこに何があるか</h2>
  {semantic_html(run_dir, sem, map_img, m["map_geometry"])}

  <h2>高さ（3D）</h2>
  {view3d_html(view3d_data(m, (sem or {}).get("landmarks")))}
  <p class="muted small">高さは OSC の縦速度（VelocityY）の積分による推定です。別々の区間で同じ場所を通ったときの差は、
  実測で 9 割が 0.5 m 以内でした。段差を上るときは縦速度が届かないことがあり、低めに出る場合があります。</p>

  <h2>外周の可能性が高い落下</h2>
  <details><summary>{len(edges)} 件を表示</summary>{edges_html or '<p class="muted">なし</p>'}</details>

  <footer>
    位置は OSC の速度と旋回の積分による推定で、数 m ずれることがあります。リスポーンのたびにスポーン地点を基準に戻しています。
    「外周の可能性」「床抜けの疑い」は、落ちた地点の周り 8 方向のうち何方向に歩いた床があるかで機械的に分けたものです。
    画面キャプチャで確かめてください。
  </footer>
</main>
</body>
</html>"""
    out = os.path.join(run_dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--no-relate", action="store_true", help="RelateAnything の注釈を付けない")
    args = ap.parse_args()
    out = build(args.run_dir, use_relate=not args.no_relate)
    print(f"報告: {out}（{os.path.getsize(out) / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
