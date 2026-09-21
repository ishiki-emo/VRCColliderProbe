"""今いる VRChat のワールドを、VRChat のログファイルから調べる。

OSC ではワールドは分からないが、VRChat は %LOCALAPPDATA%Low\\VRChat\\VRChat\\output_log_*.txt に
ワールドに入るたびに次の行を書く:

    [Behaviour] Entering Room: <ワールド名>
    [Behaviour] Joining wrld_<ID>:<インスタンス>~private(usr_<ユーザーID>)~region(jp)

インスタンスの部分にはユーザー ID が含まれることがあるので、返す（保存する）のはワールド ID と名前だけにする。

    uv run tools/world.py        # 今いるワールドを表示
"""
from __future__ import annotations

import glob
import os
import re
import sys

LOG_DIR = os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "..", "LocalLow", "VRChat", "VRChat")
JOIN_RE = re.compile(r"^(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}).*\[Behaviour\] Joining (wrld_[0-9a-f-]+)")
ROOM_RE = re.compile(r"\[Behaviour\] Entering Room: (.+?)\s*$")


def current_world(log_dir: str = LOG_DIR) -> dict | None:
    """{"id": "wrld_…", "name": "…", "joined": "YYYY.MM.DD hh:mm:ss"}。分からなければ None。"""
    logs = sorted(glob.glob(os.path.join(log_dir, "output_log_*.txt")), key=os.path.getmtime)
    if not logs:
        return None
    world, name = None, None
    with open(logs[-1], encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ROOM_RE.search(line)
            if m:
                name = m.group(1)
                continue
            m = JOIN_RE.search(line)
            if m:
                world = {"id": m.group(2), "name": name or "", "joined": m.group(1)}
    return world


def world_at(when: str, log_dir: str = LOG_DIR) -> dict | None:
    """ある時刻（"YYYY.MM.DD hh:mm:ss"）にいたワールド。過去の走行の記録を後から取り込むときに使う。"""
    world, name = None, None
    for log in sorted(glob.glob(os.path.join(log_dir, "output_log_*.txt")), key=os.path.getmtime):
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = ROOM_RE.search(line)
                if m:
                    name = m.group(1)
                    continue
                m = JOIN_RE.search(line)
                if m:
                    if m.group(1) > when:   # 同じ書式なので文字列の比較で時刻順になる
                        return world
                    world = {"id": m.group(2), "name": name or "", "joined": m.group(1)}
    return world


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    w = current_world()
    if w is None:
        print(f"VRChat のログからワールドが分からない（{os.path.normpath(LOG_DIR)}）")
        return 1
    print(f"{w['name']}  {w['id']}  （{w['joined']} に入った）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
