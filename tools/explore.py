"""段階 4: 探索。歩きながら格子地図を更新し、未踏の格子が多い方向を選ぶ。

walker.py から使う。位置は mapper.DeadReckoner の推定（数 m の誤差がある）なので、
細かい経路計画はせず「どの向きに歩くと未踏の床を踏めそうか」だけを決める。

格子:
- covered: 歩いた跡を幅 1.5m の帯として塗ったもの（探索の判断用）
- walls:   詰まった地点の正面
- cliffs:  リスポーンした落下の開始地点とその先（外周）。近づかない

向きの点数: その向きの幅 1.5m・長さ 1〜8m の通路にある未踏の格子を数える。壁や崖で打ち切る。
踏破済みの帯から 3m 以内の格子は重く、それより遠い格子は軽く数える（帯に沿って塗り広げる）。

試して分かったこと（偽ワールドでの比較）:
- 軌跡（幅 1 マス）の隣を「フロンティア」として重く数えると、自分の軌跡に沿って引き返す向きが
  いつも最高点になり、同じ線の近くを往復するだけになった。帯で塗って通路で数えることで避ける
- 未踏の格子を一律に数えると、リスポーン地点（中央）から毎回外へ一直線に進んで落ち、
  放射状の線を引くだけになった。ランダム歩行より踏破面積が 3 割少なかった
"""
from __future__ import annotations

import math
import random

CELL = 0.5
COVER_RADIUS = 1           # 歩いた格子の周り何マスまで踏破済みとみなすか（1 → 幅 1.5m）
RAY_MIN, RAY_MAX = 1.0, 8.0
BLOCK_MIN = 0.5            # 壁や崖はこの距離から見る [m]（未踏の格子は RAY_MIN から数える）
CORRIDOR = (-0.75, 0.0, 0.75)   # 通路の横方向のずれ [m]
NEAR_RADIUS = 6            # 踏破済みの帯からこのマス数（3m）以内の未踏の格子を重く数える
NEAR_W, FAR_W = 1.0, 0.2   # その重みと、それより遠い（何もないかもしれない）格子の重み
NEAR_BLOCK_PENALTY = 3.0   # すぐ先（1.5m 以内）に壁や崖がある向きの減点
CLIFF_LOOKAHEAD = 2.5      # この距離以内に既知の崖があれば避ける [m]
TARGET_MIN_CELLS = 6       # 目標にするフロンティアの最小の大きさ（マス数）
TARGET_CLIFF_M = 2.0       # 崖からこの距離以内のフロンティアは目標にしない [m]
TARGET_DIST_SCALE = 15.0   # 目標の価値 = 大きさ / (1 + 距離 / これ)
TARGET_REACHED_M = 2.5     # 目標にこの距離まで近づいたら着いた [m]
TARGET_AVOID_M = 4.0       # たどり着けなかった目標のこの距離以内は、もう目標にしない [m]
TARGET_WEIGHT = 12.0       # 目標の方角への加点（通路の点数は開けた未知の場所で 20〜40、歩き尽くした場所で 0〜5）
PLAN_MAX_EXPAND = 30000    # 経路探索で調べる格子の上限（30 分の走行で歩いた格子は約 6500）
WAYPOINT_M = 3.0           # 経路上のこの距離先を向かう方角の目安にする [m]
CLIMB_CLIFF_M = 3.0      # 既知の崖からこの距離以内では、ジャンプで壁を越えようとしない [m]


def cell_of(x: float, y: float) -> tuple[int, int]:
    return math.floor(x / CELL), math.floor(y / CELL)


def ahead(x: float, y: float, heading: float, d: float, side: float = 0.0) -> tuple[float, float]:
    h = math.radians(heading)
    return (x + d * math.sin(h) + side * math.cos(h),
            y + d * math.cos(h) - side * math.sin(h))


class Explorer:
    def __init__(self) -> None:
        self.covered: set[tuple[int, int]] = set()
        self.near: set[tuple[int, int]] = set()    # 帯から NEAR_RADIUS 以内
        self.walls: set[tuple[int, int]] = set()
        self.cliffs: set[tuple[int, int]] = set()
        self.climb_tried: set[tuple[int, int]] = set()   # ジャンプで越えようとした壁
        self._last_cell: tuple[int, int] | None = None

    def clear(self) -> None:
        self.covered.clear()
        self.near.clear()
        self.walls.clear()
        self.cliffs.clear()
        self.climb_tried.clear()
        self._last_cell = None

    def mark_visited(self, x: float, y: float) -> None:
        c = cell_of(x, y)
        if c == self._last_cell:   # 毎ループ呼ばれるので、格子が変わったときだけ塗る
            return
        self._last_cell = c
        cx, cy = c
        for dx in range(-COVER_RADIUS, COVER_RADIUS + 1):
            for dy in range(-COVER_RADIUS, COVER_RADIUS + 1):
                self.covered.add((cx + dx, cy + dy))
        r = COVER_RADIUS + NEAR_RADIUS
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                self.near.add((cx + dx, cy + dy))

    def mark_wall(self, x: float, y: float, heading: float, d: float = 0.4) -> None:
        """正面 d [m] 先に、横幅 1.5m の壁を記録する（1 マスだと少し斜めの向きの通路がすり抜ける）。"""
        for side in (-0.5, 0.0, 0.5):
            self.walls.add(cell_of(*ahead(x, y, heading, d, side)))

    def mark_cliff(self, x: float, y: float, heading: float) -> None:
        """落ち始めの地点から先を崖にする。位置の誤差を考えて左右にも広げる。"""
        for d in (0.0, 0.5, 1.0, 1.5):
            for a in (-30.0, 0.0, 30.0):
                self.cliffs.add(cell_of(*ahead(x, y, heading + a, d)))

    def score(self, x: float, y: float, heading: float) -> float:
        return self.score_detail(x, y, heading)[0]

    def score_detail(self, x: float, y: float, heading: float) -> tuple[float, float]:
        """(通路の点数, 壁や崖までの距離 [m]。塞がっていなければ RAY_MAX より大きい値)。

        壁は詰まった地点の 0.4m 先に記録するので、壁や崖は BLOCK_MIN（0.5m）から見る。
        1m 先から見ていたときは、壁のすぐ手前にいるとその壁が見えず、同じ壁に何度もぶつかった
        （30 分の走行で詰まり 153 回のうち 95 回が、同じ場所に同じような向きで再びぶつかったもの）。
        """
        s = 0.0
        d = BLOCK_MIN
        while d <= RAY_MAX:
            c = cell_of(*ahead(x, y, heading, d))
            if c in self.walls or c in self.cliffs:
                if d <= 1.5:
                    s -= NEAR_BLOCK_PENALTY
                return s, d
            if d >= RAY_MIN:
                for side in CORRIDOR:
                    cs = cell_of(*ahead(x, y, heading, d, side))
                    if cs not in self.covered:
                        w = NEAR_W if cs in self.near else FAR_W
                        s += w * (1.0 - d / (RAY_MAX * 1.5))   # 近いほど少し重い
            d += CELL / 2
        return s, RAY_MAX + CELL

    def score_toward(self, x: float, y: float, heading: float, target: dict | None,
                     bearing: float | None = None, explore_weight: float = 1.0) -> float:
        """通路の点数に、目標の方角への加点を足したもの。

        加点は、その向きで壁や崖に当たるまでの距離に比例させる（1m 先が壁ならほぼ 0、RAY_MAX 以上開けていれば満点）。
        一律に加点していたときは、周りを歩き尽くした場所では目標の方向がいつも勝ち、
        間に壁があっても離れては向き直ってぶつかる、を繰り返した。
        """
        s, block = self.score_detail(x, y, heading)
        if explore_weight != 1.0:
            # 巡回中は未踏の格子を数えず、すぐ先の壁や崖の減点だけを残す
            s = (-NEAR_BLOCK_PENALTY if block <= 1.5 else 0.0) + max(0.0, s) * explore_weight
        if target is not None:
            if bearing is None:
                bearing = math.degrees(math.atan2(target["x"] - x, target["y"] - y))
            openness = min(1.0, max(0.0, (block - 1.0) / (RAY_MAX - 1.0)))
            s += TARGET_WEIGHT * openness * max(0.0, math.cos(math.radians(heading - bearing)))
        return s

    def best_heading(self, x: float, y: float, heading: float, rng: random.Random,
                     avoid_ahead: float = 0.0, target: dict | None = None,
                     explore_weight: float = 1.0) -> tuple[float, float]:
        """15° 刻みの候補から点数が最大の向きを返す。(向き, 点数)

        avoid_ahead > 0 なら、今の向きから ±avoid_ahead° の範囲（壁や崖がある側）は選ばない。
        target（frontier_targets の要素）があれば、その方角に近い向きほど加点する。
        explore_weight=0 なら未踏の格子を数えない（巡回中）。
        """
        best = (heading, -math.inf)
        bearing = math.degrees(math.atan2(target["x"] - x, target["y"] - y)) if target else None
        for k in range(24):
            h = (heading + 15.0 * k) % 360.0
            diff = abs((h - heading + 180.0) % 360.0 - 180.0)
            if diff < avoid_ahead:
                continue
            # 同点で毎回同じ向きを選ばないよう少し揺らす
            s = self.score_toward(x, y, h, target, bearing, explore_weight) + rng.uniform(0.0, 0.5)
            if s > best[1]:
                best = (h, s)
        return best

    def climb_worth_trying(self, x: float, y: float, heading: float) -> bool:
        """詰まった壁をジャンプで越えてみる価値があるか。

        - 既知の崖（リスポーンした落下地点・挟まった場所）から CLIMB_CLIFF_M 以内では跳ばない
          （外周の柵を越えると、そのまま落ちてリスポーンする）
        - 一度試して越えられなかった壁には二度跳ばない
        """
        if cell_of(*ahead(x, y, heading, 0.4)) in self.climb_tried:
            return False
        r = int(math.ceil(CLIMB_CLIFF_M / CELL))
        cx, cy = cell_of(x, y)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if (cx + dx, cy + dy) in self.cliffs and math.hypot(dx, dy) * CELL <= CLIMB_CLIFF_M:
                    return False
        return True

    def mark_climb_tried(self, x: float, y: float, heading: float) -> None:
        # 位置の誤差を考えて、正面の 3×3 マスを試したことにする
        cx, cy = cell_of(*ahead(x, y, heading, 0.4))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                self.climb_tried.add((cx + dx, cy + dy))

    # ------------------------------------------------------------ 遠くの目標
    def frontier_targets(self, x: float, y: float, avoid: list[tuple[float, float]] = ()) -> list[dict]:
        """地図全体から、歩いた帯に接している未踏の格子のまとまり（フロンティア）を探し、目標の候補を返す。

        近くの判断（best_heading）は 8m 先までしか見ないので、スポーン地点の周りを歩き尽くすと、
        どの向きも点数がほぼ 0 になって実質ランダムに歩いていた。遠くの未踏エリアを目標にして向かう。
        外周の崖の近くのフロンティアは、外に何もないので除く。avoid の近くはたどり着けなかった目標。
        """
        if not self.covered:
            return []
        near_cliff = set()
        r = int(math.ceil(TARGET_CLIFF_M / CELL))
        for cx, cy in self.cliffs:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    near_cliff.add((cx + dx, cy + dy))
        frontier = set()
        for cx, cy in self.covered:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (cx + dx, cy + dy)
                if c not in self.covered and c not in near_cliff and c not in self.walls:
                    frontier.add(c)
        # つながっているものをまとめる
        out, seen = [], set()
        for start in frontier:
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        c = (cx + dx, cy + dy)
                        if c in frontier and c not in seen:
                            seen.add(c)
                            stack.append(c)
            if len(comp) < TARGET_MIN_CELLS:
                continue
            # 目標はまとまりの中で今の位置に一番近い格子（重心だと壁の向こう側になりやすい）
            cx, cy = min(comp, key=lambda c: (c[0] * CELL - x) ** 2 + (c[1] * CELL - y) ** 2)
            tx, ty = (cx + 0.5) * CELL, (cy + 0.5) * CELL
            if any(math.hypot(tx - ax, ty - ay) <= TARGET_AVOID_M for ax, ay in avoid):
                continue
            dist = math.hypot(tx - x, ty - y)
            if dist < TARGET_REACHED_M:
                continue
            out.append(dict(x=tx, y=ty, size=len(comp), dist=dist,
                            value=len(comp) / (1.0 + dist / TARGET_DIST_SCALE)))
        out.sort(key=lambda t: -t["value"])
        return out

    def plan_path(self, x: float, y: float, target: dict) -> list[tuple[float, float]] | None:
        """歩いたことのある格子だけをたどって target まで行く経路（A*、8 近傍）。見つからなければ None。

        目標の方角へまっすぐ向かうと、途中の壁や袋小路にぶつかって初めて分かる。長い壁だと、記録済みの部分の
        横をすり抜けて未記録の部分にまたぶつかる（実機で、詰まり 9 回のうち 8 回が目標を追っている最中だった）。
        歩いた場所は通れることが分かっているので、そこだけを通って目標（未踏エリアの縁）の手前まで行く。
        """
        import heapq
        blocked = self.walls | self.cliffs

        def walkable(c: tuple[int, int]) -> bool:
            return c in self.covered and c not in blocked

        start = cell_of(x, y)
        if not walkable(start):
            # 壁際などで今の格子が通れない扱いのときは、近くの通れる格子から始める
            near = [(start[0] + dx, start[1] + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
            near = [c for c in near if walkable(c)]
            if not near:
                return None
            start = min(near, key=lambda c: (c[0] - start[0]) ** 2 + (c[1] - start[1]) ** 2)
        goal = cell_of(target["x"], target["y"])
        # 目標は未踏の格子なので、そこだけは通れなくても行き先にしてよい
        h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])  # noqa: E731
        came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost = {start: 0.0}
        heap = [(h(start), start)]
        expanded = 0
        while heap and expanded < PLAN_MAX_EXPAND:
            _, c = heapq.heappop(heap)
            expanded += 1
            if c == goal or (h(c) <= 1.5 and c != start):
                path = []
                while c is not None:
                    path.append(((c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL))
                    c = came[c]
                return path[::-1]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    n = (c[0] + dx, c[1] + dy)
                    if n != goal and not walkable(n):
                        continue
                    if dx and dy and not (walkable((c[0] + dx, c[1])) and walkable((c[0], c[1] + dy))):
                        continue   # 壁の角をすり抜ける斜め移動はしない
                    nc = cost[c] + (1.414 if dx and dy else 1.0)
                    if nc < cost.get(n, math.inf):
                        cost[n], came[n] = nc, c
                        heapq.heappush(heap, (nc + h(n), n))
        return None

    @staticmethod
    def waypoint(path: list[tuple[float, float]], x: float, y: float) -> dict:
        """経路上で、今の位置から WAYPOINT_M ほど先の地点（向かう方角の目安）。"""
        # 経路の中で今の位置に一番近い点から先を見る
        i0 = min(range(len(path)), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
        acc, prev = 0.0, (x, y)
        for px, py in path[i0:]:
            acc += math.hypot(px - prev[0], py - prev[1])
            prev = (px, py)
            if acc >= WAYPOINT_M:
                return {"x": px, "y": py}
        return {"x": path[-1][0], "y": path[-1][1]}

    def cliff_ahead(self, x: float, y: float, heading: float) -> bool:
        d = 0.5
        while d <= CLIFF_LOOKAHEAD:
            if cell_of(*ahead(x, y, heading, d)) in self.cliffs:
                return True
            d += CELL / 2
        return False


def signed_delta(target: float, current: float) -> float:
    """current から target への最短の回転角（右回りが +、-180〜180）。"""
    return (target - current + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ 巡回
PATROL_BLOCK_M = 2.0       # 巡回範囲を分けるブロックの一辺 [m]
PATROL_TOUCH_M = 1.5       # ブロックの中心にこの距離まで近づいたら「訪れた」[m]
PATROL_DIST_SCALE = 10.0   # 行き先の価値 = 最後に訪れてからの時間 / (1 + 距離 / これ)
PATROL_UNKNOWN_W = 0.3     # 一度も歩いたことのない場所のブロックの価値の倍率


class Patrol:
    """GUI で塗った範囲（0.5m の格子の集合）を巡回する。

    範囲を PATROL_BLOCK_M 四方のブロックに分け、最後に訪れてから一番時間が経った（近いものを少し優先）ブロックを
    次の行き先にする。まだ訪れていないブロックは一番優先。繰り返すと、範囲の中をまんべんなく回り続ける。
    """

    def __init__(self, cells: set[tuple[int, int]]):
        per = max(1, round(PATROL_BLOCK_M / CELL))
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cx, cy in cells:
            groups.setdefault((cx // per, cy // per), []).append((cx, cy))
        self.cells = set(cells)
        self.blocks: dict[tuple[int, int], dict] = {}
        for key, cs in groups.items():
            if len(cs) < 2:
                continue
            # 中心は、ブロックの中で塗られた格子の平均に一番近い格子（塗った範囲の外に出ないように）
            mx = sum(c[0] for c in cs) / len(cs)
            my = sum(c[1] for c in cs) / len(cs)
            c = min(cs, key=lambda c: (c[0] - mx) ** 2 + (c[1] - my) ** 2)
            self.blocks[key] = {"x": (c[0] + 0.5) * CELL, "y": (c[1] + 0.5) * CELL, "last": None, "visits": 0}

    def inside(self, x: float, y: float) -> bool:
        return cell_of(x, y) in self.cells

    def touch(self, x: float, y: float, now: float) -> None:
        """今の位置の近くのブロックを「訪れた」にする。"""
        for b in self.blocks.values():
            if abs(b["x"] - x) <= PATROL_TOUCH_M and abs(b["y"] - y) <= PATROL_TOUCH_M \
                    and math.hypot(b["x"] - x, b["y"] - y) <= PATROL_TOUCH_M:
                if b["last"] is None or now - b["last"] > 5.0:
                    b["visits"] += 1
                b["last"] = now

    def next_target(self, x: float, y: float, now: float, ex: Explorer,
                    avoid: list[tuple[float, float]] = ()) -> dict | None:
        """次の行き先。歩いた場所を通る経路が見つかる候補を優先する。"""
        cands = []
        for b in self.blocks.values():
            if any(math.hypot(b["x"] - ax, b["y"] - ay) <= TARGET_AVOID_M for ax, ay in avoid):
                continue
            d = math.hypot(b["x"] - x, b["y"] - y)
            if d <= PATROL_TOUCH_M:
                continue
            age = 1e6 if b["last"] is None else now - b["last"]
            value = age / (1.0 + d / PATROL_DIST_SCALE)
            # 一度も歩いたことのない場所のブロックは後回し（塗った範囲が壁や物の上にかかっていると、たどり着けない）
            if ex.covered and cell_of(b["x"], b["y"]) not in ex.covered:
                value *= PATROL_UNKNOWN_W
            cands.append((value, d, b))
        cands.sort(key=lambda c: -c[0])
        for _, d, b in cands[:6]:
            t = {"x": b["x"], "y": b["y"], "dist": d, "size": 0, "patrol": True}
            if ex.plan_path(x, y, t):
                return t
        if cands:
            _, d, b = cands[0]
            return {"x": b["x"], "y": b["y"], "dist": d, "size": 0, "patrol": True}
        return None

    def coverage(self) -> float:
        """範囲のブロックのうち、この走行で一度でも訪れた割合。"""
        return sum(1 for b in self.blocks.values() if b["last"] is not None) / max(1, len(self.blocks))


def load_patrol(path: str) -> tuple[str, set[tuple[int, int]]]:
    """GUI で保存した巡回範囲（worlds/<ID>/patrols/<名前>.json）。(名前, 格子の集合)"""
    import json
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if abs(d.get("cell", CELL) - CELL) > 1e-9:
        raise ValueError(f"格子の大きさが違う: {d.get('cell')}")
    return d.get("name", ""), {tuple(c) for c in d["cells"]}
