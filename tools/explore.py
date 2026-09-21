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
CLIMB_CLIFF_M = 3.0       # 既知の崖からこの距離以内では、ジャンプで壁を越えようとしない [m]


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
        s = 0.0
        d = RAY_MIN
        while d <= RAY_MAX:
            c = cell_of(*ahead(x, y, heading, d))
            if c in self.walls or c in self.cliffs:
                if d <= 1.5:
                    s -= NEAR_BLOCK_PENALTY
                break
            for side in CORRIDOR:
                cs = cell_of(*ahead(x, y, heading, d, side))
                if cs not in self.covered:
                    w = NEAR_W if cs in self.near else FAR_W
                    s += w * (1.0 - d / (RAY_MAX * 1.5))   # 近いほど少し重い
            d += CELL
        return s

    def score_toward(self, x: float, y: float, heading: float, target: dict | None,
                     bearing: float | None = None) -> float:
        """通路の点数に、目標の方角への加点を足したもの。"""
        s = self.score(x, y, heading)
        if target is not None:
            if bearing is None:
                bearing = math.degrees(math.atan2(target["x"] - x, target["y"] - y))
            s += TARGET_WEIGHT * max(0.0, math.cos(math.radians(heading - bearing)))
        return s

    def best_heading(self, x: float, y: float, heading: float, rng: random.Random,
                     avoid_ahead: float = 0.0, target: dict | None = None) -> tuple[float, float]:
        """15° 刻みの候補から点数が最大の向きを返す。(向き, 点数)

        avoid_ahead > 0 なら、今の向きから ±avoid_ahead° の範囲（壁や崖がある側）は選ばない。
        target（frontier_targets の要素）があれば、その方角に近い向きほど加点する。
        """
        best = (heading, -math.inf)
        bearing = math.degrees(math.atan2(target["x"] - x, target["y"] - y)) if target else None
        for k in range(24):
            h = (heading + 15.0 * k) % 360.0
            diff = abs((h - heading + 180.0) % 360.0 - 180.0)
            if diff < avoid_ahead:
                continue
            # 同点で毎回同じ向きを選ばないよう少し揺らす
            s = self.score_toward(x, y, h, target, bearing) + rng.uniform(0.0, 0.5)
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
