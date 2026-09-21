"""簡易 GUI。ブラウザで開くローカル専用の画面から、走行の開始・停止、状況の確認、報告や地図の閲覧をする。

    uv run tools/gui.py            # http://127.0.0.1:8790/ を開く
    uv run tools/gui.py --port 8800 --no-browser

- 標準ライブラリだけで動く（http.server）。127.0.0.1 でだけ待ち受け、他のサイトからの操作（CSRF）は Origin で弾く
- 走行は walker.py を子プロセスで起動する。停止は停止ファイル（walker.py --stop-file）で行い、
  walker が全入力を 0 に戻して報告まで作る
- 地図は report.cumulative_map（報告の「このワールドの累計」と同じ絵）
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge  # noqa: E402
import world  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(_ROOT, "runs")
WALKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walker.py")
EXTRA_ALLOWED = {"--send-port", "--recv-port", "--world-id", "--seed"}
_map_cache: dict[str, tuple[float, dict]] = {}   # map.json のパス -> (更新時刻, 要約)


# ------------------------------------------------------------------ 走行の管理
class RunManager:
    """walker.py の子プロセスを 1 つだけ管理する。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.run_dir: str | None = None
        self.stop_file: str | None = None
        self.phase = "idle"          # idle / running / reporting
        self.options: dict = {}
        self.started = 0.0
        self.log: collections.deque = collections.deque(maxlen=300)
        self.exit_code: int | None = None

    def start(self, opt: dict) -> str | None:
        """走行を始める。始められなければ理由を返す。"""
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return "すでに走行中です"
            duration = max(10, min(int(opt.get("duration", 180)), 4 * 3600))
            self.stop_file = os.path.join(tempfile.gettempdir(), f"vrccp_stop_{os.getpid()}_{int(time.time())}")
            # -u: パイプ越しだと出力がまとめて書き出され、走行中のログが画面に届かない
            cmd = [sys.executable, "-u", WALKER, "--duration", str(duration), "--stop-file", self.stop_file]
            if opt.get("overlay"):
                cmd += ["--overlay", "--overlay-size", str(int(opt.get("overlay_size", 520)))]
            if opt.get("relate"):
                cmd.append("--relate-overlay")
            for key, flag in (("not_at_spawn", "--not-at-spawn"), ("fresh", "--fresh"),
                              ("no_climb", "--no-climb"), ("no_capture", "--no-capture")):
                if opt.get(key):
                    cmd.append(flag)
            # 画面には出さないテスト用の引数（偽ワールドのポートなど）。許可したものだけ通す
            for flag, value in (opt.get("extra") or {}).items():
                if flag in EXTRA_ALLOWED and re.fullmatch(r"[A-Za-z0-9_.-]+", str(value)):
                    cmd += [flag, str(value)]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            self.proc = subprocess.Popen(cmd, cwd=_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.phase, self.run_dir, self.exit_code = "running", None, None
            self.options, self.started = {**opt, "duration": duration}, time.time()
            self.log.clear()
            threading.Thread(target=self._read_output, args=(self.proc,), daemon=True).start()
            return None

    def _read_output(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = line.rstrip()
            with self.lock:
                self.log.append(line)
                m = re.search(r"run_dir=(\S+)", line)
                if m and self.run_dir is None:
                    self.run_dir = m.group(1)
                if "全入力を 0 に戻した" in line:
                    self.phase = "reporting"
        code = proc.wait()
        with self.lock:
            self.phase, self.exit_code = "idle", code
            if self.stop_file and os.path.exists(self.stop_file):
                os.remove(self.stop_file)

    def stop(self) -> str | None:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return "走行していません"
            open(self.stop_file, "w").close()
            return None

    def status(self) -> dict:
        with self.lock:
            st = {"phase": self.phase, "run": os.path.basename(self.run_dir) if self.run_dir else None,
                  "options": self.options, "log": list(self.log)[-60:], "exit_code": self.exit_code,
                  "wall": round(time.time() - self.started, 1) if self.started else 0}
            run_dir = self.run_dir
        st["progress"] = run_progress(run_dir) if run_dir else None
        return st


def run_progress(run_dir: str) -> dict:
    """events.jsonl から走行中の数字を拾う。"""
    counts = collections.Counter()
    t = 0.0
    goal = None
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            t = e.get("t", t)
            counts[e["kind"]] += 1
            if e["kind"] == "goal_set":
                goal = {"x": e["x"], "y": e["y"], "dist": e.get("dist")}
            elif e["kind"] in ("goal_reached", "goal_abandoned"):
                goal = None
    return {"t": t, "respawn": counts["respawn"], "wedged": counts["wedged"], "stuck": counts["stuck"],
            "climbed": counts["climbed"], "goal_reached": counts["goal_reached"], "drop": counts["drop"],
            "goal": goal}


# ------------------------------------------------------------------ データ
def runs_list(limit: int = 40) -> list[dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS_DIR, "*")), reverse=True)[:limit]:
        name = os.path.basename(d)
        start, end = {}, {}
        ev = os.path.join(d, "events.jsonl")
        if not os.path.exists(ev):
            continue
        with open(ev, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e["kind"] == "start":
                    start = e
                elif e["kind"] == "end":
                    end = e
        m = map_summary(os.path.join(d, "map.json"))
        # 古い走行の map.json には無い項目もある
        respawns, wedges = m.get("respawns") or [], m.get("wedges") or []
        verdicts = collections.Counter(r.get("verdict") for r in respawns)
        out.append({
            "name": name, "world": start.get("world_name") or "", "duration": start.get("duration"),
            "elapsed": end.get("elapsed"), "stop_reason": end.get("stop_reason"),
            "distance_m": m.get("distance_m"), "area_m2": m.get("walked_area_m2"),
            "respawns": len(respawns), "hole_suspect": verdicts.get("hole_suspect", 0),
            "wedges": len(wedges), "has_report": os.path.exists(os.path.join(d, "report.html")),
        })
    return out


def map_summary(path: str) -> dict:
    """map.json の要約（軌跡を除く）。30 分の走行で 5MB あるので、更新されていなければ前回の結果を使う。"""
    if not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    hit = _map_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    summary = {k: m.get(k) for k in ("distance_m", "walked_area_m2", "respawns", "wedges")}
    _map_cache[path] = (mtime, summary)
    return summary


def world_summary() -> dict:
    w = world.current_world()
    out = {"current": w, "worlds": []}
    for p in sorted(glob.glob(os.path.join(knowledge.WORLDS_DIR, "*", "knowledge.json"))):
        with open(p, encoding="utf-8") as f:
            k = json.load(f)
        agg = knowledge.aggregate(k)
        out["worlds"].append({"id": k["world"]["id"], "name": k["world"]["name"], "runs": agg["runs"],
                              "area_m2": round(len(agg["visited"]) * knowledge.CELL ** 2),
                              "falls": len(agg["falls"]), "landmarks": len(agg["landmarks"])})
    return out


def world_map_png(world_id: str) -> bytes | None:
    import cv2
    import report
    k = knowledge.load(world_id)
    agg = knowledge.aggregate(k)
    if not agg["visited"]:
        return None
    img, _ = report.cumulative_map(agg)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


def reset_world(world_id: str) -> str | None:
    p = knowledge.path_of(world_id)
    if not os.path.exists(p):
        return "このワールドの記録はありません"
    backup = os.path.join(os.path.dirname(p), time.strftime("knowledge_backup_%Y%m%d_%H%M%S.json"))
    os.replace(p, backup)
    return None


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    manager: RunManager
    port: int

    def log_message(self, fmt, *args) -> None:   # アクセスログは出さない
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/status":
            return self._json(self.manager.status())
        if u.path == "/api/runs":
            return self._json(runs_list())
        if u.path == "/api/world":
            return self._json(world_summary())
        if u.path == "/api/world/map.png":
            wid = (q.get("id") or [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", wid):
                return self._send(400, b"bad id", "text/plain")
            png = world_map_png(wid)
            return self._send(200, png, "image/png") if png else self._send(404, b"no map", "text/plain")
        if u.path.startswith("/runs/"):
            return self._serve_run_file(unquote(u.path[len("/runs/"):]))
        self._send(404, b"not found", "text/plain")

    def _serve_run_file(self, rel: str) -> None:
        # runs/ の外は見せない
        path = os.path.normpath(os.path.join(RUNS_DIR, rel))
        if not path.startswith(os.path.normpath(RUNS_DIR) + os.sep) or not os.path.isfile(path):
            return self._send(404, b"not found", "text/plain")
        ext = os.path.splitext(path)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg",
                 ".json": "application/json; charset=utf-8"}.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self) -> None:
        # 他のサイトのページからの操作を弾く（ブラウザは別のサイトからの POST に Origin を付ける）
        origin = self.headers.get("Origin")
        if origin not in (None, f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"):
            return self._json({"error": "許可されていない操作元です"}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "JSON が読めません"}, 400)
        u = urlparse(self.path)
        if u.path == "/api/start":
            err = self.manager.start(body)
        elif u.path == "/api/stop":
            err = self.manager.stop()
        elif u.path == "/api/world/reset":
            wid = str(body.get("id", ""))
            err = reset_world(wid) if re.fullmatch(r"[A-Za-z0-9_-]+", wid) else "ワールド ID が不正です"
        else:
            return self._json({"error": "not found"}, 404)
        self._json({"ok": err is None, "error": err}, 200 if err is None else 409)


PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRCColliderProbe</title>
<style>
:root { --bg:#f6f5f2; --panel:#fff; --text:#1d1d1b; --muted:#6b6a66; --line:#e2e0da; --accent:#2f6fdb;
  --ok:#2e8b57; --warn:#d9822b; --bad:#d6332a; }
@media (prefers-color-scheme: dark) { :root { --bg:#171716; --panel:#22221f; --text:#ecebe6; --muted:#a09e97;
  --line:#36352f; --accent:#6c9cff; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.6 "Yu Gothic UI","Meiryo",system-ui,sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:4px 16px;
  align-items:baseline; }
header h1 { font-size:18px; margin:0; }
main { display:grid; grid-template-columns: minmax(280px, 340px) 1fr; gap:16px; padding:16px 20px 40px;
  max-width:1400px; margin:0 auto; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; } }
section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin-bottom:16px; }
h2 { font-size:15px; margin:0 0 10px; }
label { display:flex; align-items:center; gap:8px; margin:6px 0; }
label.row { justify-content:space-between; }
input[type=number] { width:90px; font:inherit; padding:3px 6px; border:1px solid var(--line); border-radius:4px;
  background:var(--bg); color:var(--text); }
button { font:inherit; padding:6px 16px; border-radius:6px; border:1px solid var(--line); background:var(--panel);
  color:var(--text); cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
button.danger { color:var(--bad); }
button:disabled { opacity:.45; cursor:default; }
.buttons { display:flex; gap:8px; margin-top:12px; }
.muted, .hint { color:var(--muted); }
.hint { font-size:12px; margin:8px 0 0; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(96px,1fr)); gap:8px; }
.stat { border:1px solid var(--line); border-radius:6px; padding:6px 10px; }
.stat dt { font-size:11px; color:var(--muted); } .stat dd { margin:0; font-size:18px; font-weight:600; }
.phase { font-weight:600; } .phase.running { color:var(--ok); } .phase.reporting { color:var(--warn); }
pre.log { max-height:220px; overflow:auto; font:12px/1.5 Consolas,monospace; background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:8px; margin:10px 0 0; white-space:pre-wrap; }
.map img { max-width:100%; border-radius:6px; border:1px solid var(--line); display:block; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { font-size:12px; color:var(--muted); font-weight:600; }
.table-wrap { overflow-x:auto; }
.bad { color:var(--bad); font-weight:600; }
a { color:var(--accent); }
</style>
</head>
<body>
<header><h1>VRCColliderProbe</h1><span id="world" class="muted">ワールドを確認中…</span></header>
<main>
<div>
  <section>
    <h2>走行</h2>
    <label class="row">走る時間（分）<input id="minutes" type="number" min="1" max="240" value="3"></label>
    <label><input id="overlay" type="checkbox" checked> ミニマップを重ねる</label>
    <label class="row" style="padding-left:24px">大きさ（px）<input id="overlay_size" type="number" min="200" max="1200" value="520"></label>
    <label><input id="relate" type="checkbox" checked> RelateAnything（検出枠・見つけたもの）</label>
    <label><input id="not_at_spawn" type="checkbox"> スポーン地点以外から始める</label>
    <label><input id="no_climb" type="checkbox"> ジャンプで乗り越えない</label>
    <label><input id="fresh" type="checkbox"> ワールドの記録を使わない（--fresh）</label>
    <div class="buttons">
      <button id="start" class="primary">開始</button>
      <button id="stop" class="danger" disabled>停止</button>
    </div>
    <p class="hint">開始する前に、VRChat のメニューを閉じ、スポーン地点から始めるならリスポーンしておいてください。
    プライベートインスタンスか Build &amp; Test で、様子を見られる状態で使ってください。</p>
  </section>
  <section>
    <h2>状況 <span id="phase" class="phase muted">待機中</span></h2>
    <dl class="stats" id="stats"></dl>
    <pre class="log" id="log"></pre>
  </section>
</div>
<div>
  <section>
    <h2>このワールドの地図（累計）</h2>
    <div id="worldinfo" class="muted"></div>
    <div class="map" id="map"></div>
    <div class="buttons"><button id="reset" class="danger" disabled>このワールドの記録をリセット</button></div>
    <p class="hint">リセットしても消えるのは累計の記録だけで、今までの記録はバックアップに退避し、各走行（runs/）も残ります。</p>
  </section>
  <section>
    <h2>走行の一覧</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>日時</th><th>ワールド</th><th>時間</th><th>歩行距離</th><th>踏破面積</th><th>落下</th><th>床抜けの疑い</th><th>挟まり</th><th>報告</th></tr></thead>
      <tbody id="runs"></tbody></table></div>
  </section>
</div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let currentWorld = null, lastPhase = null;

async function post(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.ok === false) alert(j.error || '失敗しました');
  return j;
}
$('start').onclick = () => post('/api/start', {
  duration: Math.round(+$('minutes').value * 60), overlay: $('overlay').checked, overlay_size: +$('overlay_size').value,
  relate: $('relate').checked, not_at_spawn: $('not_at_spawn').checked, no_climb: $('no_climb').checked,
  fresh: $('fresh').checked }).then(refreshStatus);
$('stop').onclick = () => { if (confirm('走行を止めますか？（全入力を 0 に戻して報告を作ります）')) post('/api/stop').then(refreshStatus); };
$('reset').onclick = () => {
  if (currentWorld && confirm(`「${currentWorld.name || currentWorld.id}」の記録をリセットしますか？\n（今までの記録はバックアップに退避します）`))
    post('/api/world/reset', {id: currentWorld.id}).then(refreshWorld);
};

async function refreshStatus() {
  const s = await (await fetch('/api/status')).json();
  const labels = {idle: '待機中', running: '走行中', reporting: '報告を作成中'};
  $('phase').textContent = labels[s.phase] || s.phase;
  $('phase').className = 'phase ' + (s.phase === 'idle' ? 'muted' : s.phase);
  $('start').disabled = s.phase !== 'idle';
  $('stop').disabled = s.phase !== 'running';
  const p = s.progress || {};
  const dur = (s.options || {}).duration;
  const items = s.phase === 'idle' && !s.run ? [] : [
    ['経過', p.t != null ? `${Math.round(p.t)} / ${dur || '?'} 秒` : '—'],
    ['リスポーン', p.respawn ?? '—'], ['挟まり', p.wedged ?? '—'], ['詰まり', p.stuck ?? '—'],
    ['乗り越え', p.climbed ?? '—'], ['目標に到達', p.goal_reached ?? '—']];
  $('stats').innerHTML = items.map(([a, b]) => `<div class="stat"><dt>${a}</dt><dd>${esc(b)}</dd></div>`).join('');
  const log = $('log'), atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = (s.log || []).join('\n');
  if (atBottom) log.scrollTop = log.scrollHeight;
  if (lastPhase && lastPhase !== 'idle' && s.phase === 'idle') { refreshRuns(); refreshWorld(); }
  lastPhase = s.phase;
}
async function refreshRuns() {
  const runs = await (await fetch('/api/runs')).json();
  $('runs').innerHTML = runs.map(r => `<tr>
    <td>${esc(r.name.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3 $4:$5'))}</td>
    <td>${esc(r.world)}</td><td>${r.elapsed ? Math.round(r.elapsed / 60) + ' 分' : '—'}</td>
    <td>${r.distance_m != null ? Math.round(r.distance_m) + ' m' : '—'}</td>
    <td>${r.area_m2 != null ? Math.round(r.area_m2) + ' m²' : '—'}</td>
    <td>${r.respawns}</td><td class="${r.hole_suspect ? 'bad' : ''}">${r.hole_suspect}</td><td>${r.wedges}</td>
    <td>${r.has_report ? `<a href="/runs/${encodeURIComponent(r.name)}/report.html" target="_blank" rel="noopener">開く</a>` : '—'}</td></tr>`).join('');
}
async function refreshWorld() {
  const w = await (await fetch('/api/world')).json();
  currentWorld = w.current;
  $('world').textContent = w.current ? `今いるワールド: ${w.current.name}（${w.current.id}）` : 'VRChat のログからワールドが分かりません';
  const k = w.current && w.worlds.find(x => x.id === w.current.id);
  $('reset').disabled = !k;
  $('worldinfo').textContent = k ? `走行 ${k.runs} 回 ・ 累計の踏破 ${k.area_m2} m² ・ 落下 ${k.falls} ・ 見つけたもの ${k.landmarks} 件`
                                 : 'このワールドの記録はまだありません。';
  $('map').innerHTML = k ? `<img src="/api/world/map.png?id=${encodeURIComponent(k.id)}&t=${Date.now()}" alt="このワールドの累計の地図">` : '';
}
refreshStatus(); refreshRuns(); refreshWorld();
setInterval(refreshStatus, 1000);
</script>
</body>
</html>
"""


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = ap.parse_args()
    Handler.manager = RunManager()
    Handler.port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"VRCColliderProbe GUI: {url}（Ctrl+C で終了）")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # GUI を閉じても走行が残らないよう、走っていれば止める
        if Handler.manager.proc is not None and Handler.manager.proc.poll() is None:
            Handler.manager.stop()
            try:
                Handler.manager.proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                pass
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
