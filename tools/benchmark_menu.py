"""Hold the first menu input until the driver's background library pass finishes."""

import ctypes as c
from ctypes import wintypes as w
import os, threading, time
from benchmark_processes import competing

k = c.WinDLL("kernel32", use_last_error=True)
k.CreateNamedPipeW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, w.DWORD, c.c_void_p]
k.CreateNamedPipeW.restype = w.HANDLE
k.ConnectNamedPipe.argtypes = [w.HANDLE, c.c_void_p]
k.WriteFile.argtypes = [w.HANDLE, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p]
k.FlushFileBuffers.argtypes = [w.HANDLE]
k.DisconnectNamedPipe.argtypes = [w.HANDLE]
k.CloseHandle.argtypes = [w.HANDLE]


class MenuGate:
    def __init__(self, route, log):
        self.path = rf"\\.\pipe\nx2-benchmark-{os.getpid()}-{time.monotonic_ns()}"
        self.route = route.read_bytes()
        self.log = log
        self.note = {}
        self.cancel = threading.Event()
        self.pid = 0
        self.handle = k.CreateNamedPipeW(self.path, 2, 0, 1, 65536, 65536, 0, None)
        if self.handle == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        start = time.monotonic()
        try:
            if not k.ConnectNamedPipe(self.handle, None) and c.get_last_error() != 535:
                raise c.WinError(c.get_last_error())
            quiet_since = None
            while not self.cancel.is_set():
                msg = self.log.read_text(errors="replace") if self.log.exists() else ""
                conditions = (
                    "driver's cache has the prepared libraries",
                    "library warm: done,",
                    "library warm: ends for this run",
                    "library warm: the prepared libraries are cold again",
                )
                reason = next((s for s in conditions if s in msg), None)
                if self.pid and competing(self.pid):
                    quiet_since = time.monotonic()
                quiet = quiet_since is None or time.monotonic() - quiet_since >= 15
                if quiet and (reason or time.monotonic() - start > 1200):
                    self.note = dict(
                        wait_seconds=round(time.monotonic() - start, 3), reason=reason or "wait limit"
                    )
                    written = w.DWORD()
                    if not k.WriteFile(
                        self.handle, self.route, len(self.route), c.byref(written), None
                    ) or written.value != len(self.route):
                        raise c.WinError(c.get_last_error())
                    k.FlushFileBuffers(self.handle)
                    break
                self.cancel.wait(0.5)
        except OSError as ex:
            self.note["error"] = str(ex)
        finally:
            # Closing the writer gives the stdio reader EOF after the last script line.
            k.CloseHandle(self.handle)
