"""VRChat ウィンドウの画面キャプチャ。

RelateAnything の tools/game_scene.py から流用（DPI 対応・ウィンドウ特定）。
撮影は既定で PrintWindow（ウィンドウの中身だけ）。上に重なったウィンドウやオーバーレイは写らない。
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import cv2
import numpy as np


def enable_dpi_awareness() -> None:
    """物理ピクセル座標で取得するため、プロセスを DPI aware にする。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


def list_windows() -> list[tuple[int, str, Rect]]:
    import win32gui
    out: list[tuple[int, str, Rect]] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        if r - l < 120 or b - t < 120:
            return
        out.append((hwnd, title, Rect(l, t, r - l, b - t)))

    win32gui.EnumWindows(cb, None)
    return out


def find_window(substr: str) -> tuple[int, str] | None:
    hits = [(h, t) for h, t, _ in list_windows() if substr.lower() in t.lower()]
    return hits[0] if hits else None


def process_exe(hwnd: int) -> str:
    """ウィンドウを持つプロセスの実行ファイル名（例: VRChat.exe）。分からなければ空文字。"""
    import win32process
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(len(buf))
        if not ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return ""
        return os.path.basename(buf.value)
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def find_vrchat_window() -> tuple[int, str] | None:
    """VRChat のゲーム画面のウィンドウ。実行ファイル名（VRChat.exe）で見分ける。

    タイトルの部分一致（find_window("VRChat")）だと、タイトルに「VRChat」を含むブラウザのタブ（X の投稿など）を
    先に見つけることがあり、オーバーレイがブラウザに重なり、画面キャプチャもブラウザを撮っていた。
    """
    hits = [(h, t) for h, t, _ in list_windows() if process_exe(h).lower() == "vrchat.exe"]
    if not hits:
        return None
    import win32gui
    # 同じプロセスに複数ある場合は、Unity のゲーム画面（UnityWndClass）を優先する
    hits.sort(key=lambda ht: win32gui.GetClassName(ht[0]) != "UnityWndClass")
    return hits[0]


def client_rect(hwnd: int) -> Rect:
    """タイトルバー・枠を除いたクライアント領域（スクリーン座標）。"""
    import win32gui
    l, t, r, b = win32gui.GetClientRect(hwnd)
    sx, sy = win32gui.ClientToScreen(hwnd, (l, t))
    return Rect(sx, sy, r - l, b - t)


def is_foreground(hwnd: int) -> bool:
    import win32gui
    return win32gui.GetForegroundWindow() == hwnd


PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2


def print_window(hwnd: int) -> np.ndarray | None:
    """ウィンドウの中身だけを撮る（PrintWindow + PW_RENDERFULLCONTENT）。

    画面上の領域ではなくウィンドウ自身の描画内容なので、上に重なった別のウィンドウ
    （このツールのオーバーレイも含む）は写らない。VRChat（Unity の DirectX 描画）でも撮れることを確認済み
    （1690x1122 前後で 1 枚 約 34ms）。最小化中や失敗したときは None。
    """
    import win32gui
    import win32ui
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    if w < 8 or h < 8:
        return None
    hdc = win32gui.GetWindowDC(hwnd)
    src = win32ui.CreateDCFromHandle(hdc)
    mem = src.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    try:
        bmp.CreateCompatibleBitmap(src, w, h)
        mem.SelectObject(bmp)
        ok = ctypes.windll.user32.PrintWindow(hwnd, mem.GetSafeHdc(), PW_CLIENTONLY | PW_RENDERFULLCONTENT)
        if not ok:
            return None
        img = np.frombuffer(bmp.GetBitmapBits(True), np.uint8).reshape(h, w, 4)[:, :, :3]
        return np.ascontiguousarray(img)
    finally:
        win32gui.DeleteObject(bmp.GetHandle())
        mem.DeleteDC()
        src.DeleteDC()
        win32gui.ReleaseDC(hwnd, hdc)


class WindowCapture:
    """ウィンドウのクライアント領域を撮る。

    既定はウィンドウの中身だけを撮る PrintWindow。オーバーレイ（overlay.py / relate_overlay.py）を
    OBS や Win+Shift+R の録画に写せるよう「キャプチャから除外」をやめたので、証拠用の画面や推論用の画面に
    オーバーレイが写り込まないよう、画面上の領域を撮る mss ではなくこちらを使う。
    PrintWindow が真っ黒や失敗を返すウィンドウでは method="mss" にする（その場合オーバーレイは写り込む）。
    """

    def __init__(self, hwnd: int, method: str = "printwindow"):
        self.hwnd = hwnd
        self.method = method
        self._sct = None
        if method == "mss":
            import mss
            self._sct = getattr(mss, "MSS", mss.mss)()

    def read(self) -> np.ndarray | None:
        if self.method == "printwindow":
            return print_window(self.hwnd)
        r = client_rect(self.hwnd)
        if r.w < 8 or r.h < 8:   # 最小化中など
            return None
        raw = self._sct.grab({"left": r.x, "top": r.y, "width": r.w, "height": r.h})
        return np.ascontiguousarray(np.asarray(raw)[:, :, :3])   # BGRA -> BGR


def imwrite(path: str, img: np.ndarray) -> None:
    """日本語パスでも書けるように cv2.imencode 経由で保存する。"""
    ok, buf = cv2.imencode(".png" if path.lower().endswith(".png") else ".jpg", img)
    if ok:
        buf.tofile(path)
