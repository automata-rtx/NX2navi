"""Replay opt-in EFFCAP01 immutable GPU-binding captures without launching the game.

Captures remain in ignored build directories. Output contains identities, timings and derived ranges, never shader
or uniform bytes. Python callbacks add overhead: use the native microprobe for renderer-cost measurements.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
READ = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint64, C.c_size_t, C.c_void_p)
BUILTIN = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64))


class Inputs(C.Structure):
    _fields_ = [("read", READ), ("context", C.c_void_p), ("direct", C.c_uint32),
                ("groups", C.c_uint32 * 3), ("local", C.c_uint32 * 3),
                ("workspace", C.c_void_p), ("workspace_bytes", C.c_size_t), ("builtin", BUILTIN)]


class Range(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("set", C.c_uint32), ("binding", C.c_uint32),
                ("begin", C.c_uint64), ("size", C.c_uint64)]


def capsule(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) > 64 << 20 or len(raw) < 104 or raw[:8] != b"EFFCAP01":
        raise ValueError(f"Invalid capture header or size: {path}")
    meta = struct.unpack_from("<13I", raw, 40)
    words, count = struct.unpack_from("<QI", raw, 92)
    offset = 104
    if words > (len(raw) - offset) // 4 or count > 256:
        raise ValueError(f"Invalid capture counts: {path}")
    spv = raw[offset:offset + words * 4]
    offset += words * 4
    buffers = {}
    for _ in range(count):
        if offset + 20 > len(raw):
            raise ValueError(f"Truncated buffer header: {path}")
        kind, set_, binding, size = struct.unpack_from("<IIIQ", raw, offset)
        offset += 20
        if kind not in (0, 1) or size > len(raw) - offset or (set_, binding) in buffers:
            raise ValueError(f"Invalid or duplicate capture binding: {path}")
        buffers[set_, binding] = (kind, raw[offset:offset + size])
        offset += size
    if offset != len(raw):
        raise ValueError(f"Unexpected capture trailing bytes: {path}")
    return {"path": str(path), "key": raw[8:40].hex(), "sha256": hashlib.sha256(raw).hexdigest(),
            "spirv_sha256": hashlib.sha256(spv).hexdigest(), "meta": meta, "spv": spv, "buffers": buffers}


def configure(lib: C.CDLL) -> None:
    lib.rc_effect_compile.argtypes = [C.POINTER(C.c_uint32), C.c_size_t]
    lib.rc_effect_compile.restype = C.c_void_p
    lib.rc_effect_free.argtypes = [C.c_void_p]
    for name in ("rc_effect_supported", "rc_effect_write_count", "rc_effect_workspace_size"):
        getattr(lib, name).argtypes = [C.c_void_p]
        getattr(lib, name).restype = C.c_size_t
    lib.rc_effect_evaluate.argtypes = [C.c_void_p, C.POINTER(Inputs), C.POINTER(Range), C.c_size_t, C.POINTER(C.c_size_t)]
    lib.rc_effect_read_ranges.argtypes = [C.c_void_p, C.POINTER(Inputs), C.c_uint32, C.c_uint32,
                                         C.POINTER(Range), C.c_size_t, C.POINTER(C.c_size_t)]


def replay(lib: C.CDLL, cap: dict) -> dict:
    spv = (C.c_uint32 * (len(cap["spv"]) // 4)).from_buffer_copy(cap["spv"])
    started = time.perf_counter()
    program = lib.rc_effect_compile(spv, len(spv))
    compile_ms = (time.perf_counter() - started) * 1000
    meta, buffers = cap["meta"], cap["buffers"]
    read_calls: dict[tuple, int] = {}

    @READ
    def read(_context, set_, binding, offset, size, destination):
        key = (set_, binding)
        found = buffers.get(key)
        good = bool(found is not None and offset <= len(found[1]) and size <= len(found[1]) - offset)
        dependency = (set_, binding, offset, size, good)
        read_calls[dependency] = read_calls.get(dependency, 0) + 1
        if good:
            C.memmove(destination, found[1][offset:offset + size], size)
        return int(good)

    @BUILTIN
    def builtin(_context, code, component, minimum, maximum):
        if not meta[8] or component or code not in (42, 43):
            return 0
        at = 9 if code == 42 else 11
        minimum[0], maximum[0] = meta[at:at + 2]
        return int(minimum[0] <= maximum[0])

    result = {k: cap[k] for k in ("path", "key", "sha256", "spirv_sha256")}
    result.update(stage=meta[0], direct_dispatch=bool(meta[1]), groups=list(meta[2:5]), local=list(meta[5:8]),
                  vertices_known=bool(meta[8]), compile_ms=compile_ms,
                  buffers=[{"kind": kind, "set": key[0], "binding": key[1], "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()} for key, (kind, data) in sorted(buffers.items())])
    try:
        result["supported"] = bool(lib.rc_effect_supported(program))
        result["writes"] = lib.rc_effect_write_count(program)
        size = lib.rc_effect_workspace_size(program)
        result["workspace_bytes"] = size
        if not program or not result["supported"]:
            result["status"] = "unsupported"
            return result
        scratch = C.create_string_buffer(size)
        inputs = Inputs(read, None, meta[1], (C.c_uint32 * 3)(*meta[2:5]), (C.c_uint32 * 3)(*meta[5:8]),
                        C.cast(scratch, C.c_void_p), size, builtin)
        ranges, count = (Range * 256)(), C.c_size_t()
        started = time.perf_counter()
        status = lib.rc_effect_evaluate(program, C.byref(inputs), ranges, len(ranges), C.byref(count))
        result["evaluate_ms_with_python_callbacks"] = (time.perf_counter() - started) * 1000
        result["status"] = ("complete", "unknown", "overflow")[status]
        result["ranges"] = [{"kind": ("gpa", "ssbo")[r.kind], "set": r.set, "binding": r.binding,
                             "begin": r.begin, "size": r.size} for r in ranges[:count.value]]
        result["input_reads"] = [{"set": k[0], "binding": k[1], "offset": k[2], "size": k[3],
                                  "available": k[4], "calls": n} for k, n in sorted(read_calls.items())]
        return result
    finally:
        lib.rc_effect_free(program)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Capture files or directories (recursively scanned)")
    parser.add_argument("--out", type=Path, help="Write compact JSON evidence; defaults to stdout")
    parser.add_argument("--steps", type=int, help="Offline-only abstract-step budget override")
    args = parser.parse_args()
    if args.steps is not None and not 1 <= args.steps <= 100_000_000:
        parser.error("--steps must be between 1 and 100000000")
    files = sorted({file.resolve() for p in args.paths for file in (p.rglob("*.effect") if p.is_dir() else [p])})
    if not files:
        parser.error("No .effect captures found")
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        parser.error("Set CC to the UCRT64 Clang executable")
    work = ROOT / "build" / "test"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="effect-capsules-", dir=work) as temporary:
        library = Path(temporary) / ("effects.dll" if os.name == "nt" else "effects.so")
        flags = ["-Wl,--export-all-symbols"] if os.name == "nt" else ["-fPIC"]
        if args.steps is not None:
            flags.append(f"-DRC_EFFECT_STEP_LIMIT={args.steps}")
        subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", *flags,
                        str(ROOT / "runtime" / "resource_core_effects.c"), "-o", str(library)], check=True)
        lib = C.CDLL(str(library))
        configure(lib)
        try:
            results = [replay(lib, capsule(path)) for path in files]
        finally:
            if os.name == "nt":
                import _ctypes
                _ctypes.FreeLibrary(lib._handle)
                lib._handle = 0
    evidence = {"schema": 1, "acceptance_boundary": "Offline immutable input proof replay; no route, visual or performance acceptance.",
                "step_budget_override": args.steps, "capsules": results}
    text = json.dumps(evidence, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Analyzed {len(results)} captures: " + ", ".join(f"{s}={sum(r['status'] == s for r in results)}"
                                                              for s in ("complete", "unknown", "unsupported", "overflow")))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
