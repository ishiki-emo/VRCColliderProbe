"""偽ワールド: 半径 25m の島＋障害物。VRChat なしで walker.py を試すためのシミュレータ。

    uv run tools/sim_world.py <秒数> <結果.json> [ポート=19000]
    uv run tools/walker.py --send-port 19000 --recv-port 19001 --no-capture --duration <秒数>

<ポート> で入力を受け、<ポート>+1 に SPEC.md 3 章の実測どおりの形でパラメータを返す
（値は変化時のみ、歩行 3 m/s、旋回 200°/s、VelocityX/Z は平滑化、端から落ちると 1 秒でスポーン地点へ）。
ジャンプ（/input/Jump）にも対応し、高さ 0.5m の低い壁（LOW）は跳べば越えられる。
結果.json には本当に踏んだ面積（足元 / 幅 1.5m の帯）とリスポーン回数が入る。
ポートを変えれば複数を同時に走らせられる。9000/9001 は本物の VRChat と取り合うので使わない。
"""
import math, sys, threading, time, json
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
DUR = float(sys.argv[1]); OUT = sys.argv[2]; PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 19000
inp = {}
d = Dispatcher(); d.set_default_handler(lambda a, *v: inp.__setitem__(a, v[0] if v else 0))
srv = ThreadingOSCUDPServer(("127.0.0.1", PORT), d)
threading.Thread(target=srv.serve_forever, daemon=True).start()
out = SimpleUDPClient("127.0.0.1", PORT + 1)
R = 25.0
BOXES = [(-4, 1.5, 4, 2.5), (5, 5, 10, 8), (-12, -6, -8, 4), (-3, -15, 6, -12), (10, -12, 14, -2), (-6, 10, 2, 12)]
# low walls (0.5m): jumping over them works while the avatar is higher than LOW_H
LOW = [(-3, -6.2, 3, -5.8), (6, -3, 6.4, 3)]
LOW_H = 0.5
def blocked(x, y, jz=0.0):
    if any(a <= x <= c and b <= y <= e for a, b, c, e in BOXES):
        return True
    return jz < LOW_H and any(a <= x <= c and b <= y <= e for a, b, c, e in LOW)
last = {}
def send(n, v):
    if last.get(n) != v:
        last[n] = v; out.send_message("/avatar/parameters/" + n, v)
x = y = h = 0.0; vy = 0.0; grounded = True; fall_t = None; sx = sz = 0.0
jz = jvy = 0.0; jumping = False; climbs = 0
visited = set(); band = set(); dt = 1 / 60; t0 = time.perf_counter(); respawns = 0
prev = time.perf_counter()
while time.perf_counter() - t0 < DUR:
    now = time.perf_counter(); dt = max(1e-3, now - prev); prev = now
    V = inp.get("/input/Vertical", 0.0); H = inp.get("/input/Horizontal", 0.0); L = inp.get("/input/LookHorizontal", 0.0)
    turn = 1 if L > 0.9 else -1 if L < -0.9 else 0
    h = (h + 200 * turn * dt) % 360; send("AngularY", 200.0 * turn)
    lx, lz = H, V; n = math.hypot(lx, lz)
    if n > 0: lx, lz = lx / n * 3, lz / n * 3
    if grounded:
        if not jumping and inp.get("/input/Jump", 0) == 1:
            jumping, jvy = True, 3.84; send("Grounded", False)
        if jumping:
            jz += jvy * dt; jvy -= 9.81 * dt
            if jz <= 0.0:
                jz, jvy, jumping = 0.0, 0.0, False
                send("VelocityY", 0.0); send("Grounded", True)
            else:
                send("VelocityY", round(jvy, 4))
        r = math.radians(h)
        wx, wy = lz * math.sin(r) + lx * math.cos(r), lz * math.cos(r) - lx * math.sin(r)
        nx, ny = x + wx * dt, y + wy * dt
        if blocked(nx, ny, jz):
            if not blocked(nx, y, jz): ny = y
            elif not blocked(x, ny, jz): nx = x
            else: nx, ny = x, y
        spd = math.hypot(nx - x, ny - y) / dt
        # 実際の移動をアバター基準に戻す
        ax, ay = (nx - x) / dt, (ny - y) / dt
        alx, alz = ax * math.cos(r) - ay * math.sin(r), ax * math.sin(r) + ay * math.cos(r)
        x, y = nx, ny
        cx, cy = math.floor(x / 0.5), math.floor(y / 0.5); visited.add((cx, cy))
        band.update((cx + i, cy + j) for i in (-1, 0, 1) for j in (-1, 0, 1))
        if math.hypot(x, y) > R:
            grounded = False; fall_t = 0.0; send("Grounded", False)
    else:
        fall_t += dt; vy -= 9.81 * dt; spd = 0.0; alx = alz = 0.0
        if fall_t > 1.0:
            x = y = h = 0.0; vy = -0.164; fall_t = -99; respawns += 1
        elif fall_t < -98.8:
            vy = 0.0; grounded = True; fall_t = None
            send("VelocityY", 0.0); send("Grounded", True)
    if fall_t is not None and fall_t < -90: fall_t += dt
    sx += max(-10 * dt, min(10 * dt, alx - sx)); sz += max(-10 * dt, min(10 * dt, alz - sz))
    send("VelocityX", round(sx, 4)); send("VelocityZ", round(sz, 4))
    if not grounded: send("VelocityY", round(vy, 4))
    send("VelocityMagnitude", round(math.hypot(spd, vy if not grounded else (jvy if jumping else 0.0)), 4))
    time.sleep(1 / 60)
json.dump({"true_area_m2": len(visited) * 0.25, "band_area_m2": len(band) * 0.25, "respawns": respawns}, open(OUT, "w"))
