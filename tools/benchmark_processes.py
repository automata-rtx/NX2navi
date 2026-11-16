"""Lightweight process-name check for the replacement benchmark runs."""

import ctypes as c
from ctypes import wintypes as w


class Entry(c.Structure):
    _fields_ = [
        ("size", w.DWORD),
        ("usage", w.DWORD),
        ("pid", w.DWORD),
        ("heap", c.c_size_t),
        ("module", w.DWORD),
        ("threads", w.DWORD),
        ("parent", w.DWORD),
        ("priority", w.LONG),
        ("flags", w.DWORD),
        ("name", w.WCHAR * 260),
    ]


api = c.windll.kernel32
api.CreateToolhelp32Snapshot.restype = w.HANDLE
api.Process32FirstW.argtypes = [w.HANDLE, c.POINTER(Entry)]
api.Process32NextW.argtypes = api.Process32FirstW.argtypes
api.CloseHandle.argtypes = [w.HANDLE]


def competing(allowed=0):
    handle = api.CreateToolhelp32Snapshot(2, 0)
    if handle == c.c_void_p(-1).value:
        raise OSError("process snapshot failed")
    entry = Entry()
    entry.size = c.sizeof(entry)
    found = []
    try:
        ok = api.Process32FirstW(handle, c.byref(entry))
        if not ok:
            raise OSError("process enumeration failed")
        while ok:
            name = entry.name.lower()
            if entry.pid != allowed and (
                name.startswith(("2nx2navi", "2x2navi", "ocarina", "nx2navi.exe"))
                or name
                in {
                    "ocarina.exe",
                    "nx2navi.exe",
                    "ninja.exe",
                    "clang.exe",
                    "clang++.exe",
                    "clang-20.exe",
                    "clang-21.exe",
                    "clang-22.exe",
                    "ld.lld.exe",
                    "lld-link.exe",
                }
            ):
                found.append([entry.pid, name])
            ok = api.Process32NextW(handle, c.byref(entry))
    finally:
        api.CloseHandle(handle)
    return found
