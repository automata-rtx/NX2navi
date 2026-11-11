"""Add archive pipeline coverage to a completed installation without rewarming its prepared stages.

Uses the installed translator and renderer. Existing translated shaders and the Vulkan cache stay in place.
Run with the game's bundled Python: extend_shader_preparation.py <game folder> --workers 4.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

HEADER = struct.pack('<8sII', b'NX2NSPRP', 1, 36)
SEEDS = ('seed-programs.bin', 'seed-answers.bin', 'seed-mesh.bin', 'seed-vertex.bin', 'seed-passes.bin')


def read_manifest(data):
    if data[:16] != HEADER or (len(data) - 16) % 36:
        raise ValueError('Malformed prepared shader inventory')
    rows = {}
    for offset in range(16, len(data), 36):
        row = data[offset:offset + 36]
        if row[0] > 7 or row[1] > 2 or row[2:4] != bytes(2) or row[4:] == bytes(32):
            raise ValueError('Invalid prepared shader record')
        if row[4:] in rows and rows[row[4:]] != row:
            raise ValueError('Conflicting prepared shader records')
        rows[row[4:]] = row
    return rows


def merge_manifests(baseline, prepared):
    old, new = read_manifest(baseline), read_manifest(prepared)
    if not old or not new:
        raise ValueError('Additive preparation requires nonempty inventories')
    # Current metadata corrects older mesh/task rows mislabeled as vertex. IDs still identify the same input.
    union = old | new
    delta = {key: value for key, value in new.items() if key not in old}
    return HEADER + b''.join(union.values()), HEADER + b''.join(delta.values()), len(old), len(delta)


def atomic_write(path, data):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def profile_lock(game):
    path = game / 'profile' / 'instance.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
        kernel.CreateFileW.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.CreateFileW(str(path), 0xc0000000, 0, None, 4, 0, None)
        if handle == w.HANDLE(-1).value:
            raise RuntimeError('The game profile is in use; close the game before extending preparation')
        try:
            yield
        finally:
            kernel.CloseHandle(handle)
    else:
        import fcntl
        with path.open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def run(command, game, env, log):
    with log.open('wb') as stream:
        result = subprocess.run(command, cwd=game, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name} failed ({result.returncode}); see {log}')


def extend(game, workers, exe_name):
    game = game.resolve()
    exe = game / exe_name
    cache = game / 'pipeline-cache.bin'
    manifest = game / 'shader-prepared.bin'
    if not exe.is_file() or not manifest.is_file() or not cache.is_file() or cache.stat().st_size <= 32:
        raise ValueError('A completed installation and its saved Vulkan cache are required')
    work = game / 'build-cache' / 'additional-preparation'
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(NX2NAVI_PIPELINE_CACHE=str(cache), NX2NAVI_SHADER_CACHE=str(game / 'shader-cache'),
               NX2NAVI_XLATE=str(game / 'nx2navi_xlate.exe'), NX2NAVI_PREWARM_XLATE=str(workers),
               NX2NAVI_PIPELINE_CACHE_MAX_MB='0', NX2NAVI_PIPE_BINARIES='0', NX2NAVI_DISCORD_RPC='0',
               NX2NAVI_LIB_WARM='0', NX2NAVI_LIB_PREBUILD='0', NX2NAVI_PIPE_FEEDBACK='1')
    worker_file = work / 'workers.bin'
    worker_file.write_bytes(bytes([workers]))
    env['NX2NAVI_SHADER_WORKER_FILE'] = str(worker_file)
    with profile_lock(game):
        identity_log = work / 'identity.log'
        run([str(exe), '--shader-cache-id'], game, env, identity_log)
        lines = identity_log.read_text(errors='replace').splitlines()
        identity = next(line for line in lines if line.startswith('shader-cache-id '))
        if 'shader-driver-cache-ready 1' not in lines or 'shader-translations-ready 1' not in lines:
            raise RuntimeError(f'The completed cache is not reusable; refusing a full rebuild. See {identity_log}')
        state_path = work / 'state.json'
        baseline_path = work / 'baseline.bin'
        stamp = hashlib.sha256(exe.read_bytes()).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['identity'] != identity or state['executable'] != stamp:
                raise RuntimeError('Renderer or driver changed during additive preparation; review before resuming')
            if state['phase'] == 'complete':
                print('Additional preparation already completed for this renderer and driver.', flush=True)
                return
        else:
            baseline = manifest.read_bytes()
            read_manifest(baseline)
            atomic_write(baseline_path, baseline)
            state = dict(identity=identity, executable=stamp, phase='inventory', cache_bytes_before=cache.stat().st_size)
            atomic_write(state_path, json.dumps(state, indent=2).encode())
        baseline = baseline_path.read_bytes()
        if state['phase'] == 'inventory':
            print('Reading shader combinations from this installation...', flush=True)
            run([sys.executable, str(Path(__file__).with_name('prepare_shaders.py')), str(game), '--romfs',
                 str(game / 'romfs'), '--exefs', str(game / 'exefs'), '--no-list', '--no-run'], game, env, work / 'inventory.log')
            print('Reusing existing translations and translating missing variants...', flush=True)
            try:
                run([str(exe), '--prepare-shader-inputs'], game, env, work / 'translations.log')
                union, delta, reused, added = merge_manifests(baseline, manifest.read_bytes())
                atomic_write(game / 'shader-additional.bin', delta)
                atomic_write(manifest, union)
            except BaseException:
                atomic_write(manifest, baseline)
                raise
            state.update(phase='pipelines', baseline_stages=reused, additional_stages=added)
            atomic_write(state_path, json.dumps(state, indent=2).encode())
        print(f"Preserving {state['baseline_stages']} prepared stages; warming {state['additional_stages']} additional "
              'stages and archive graphics combinations.', flush=True)
        run([str(exe), '--warm-additional-pipelines'], game, env, work / 'pipelines.log')
        state.update(phase='complete', cache_bytes_after=cache.stat().st_size)
        atomic_write(state_path, json.dumps(state, indent=2).encode())
        for name in SEEDS:
            (game / name).unlink(missing_ok=True)
        print(f'Additional preparation saved. Report: {state_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game', type=Path)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 65), metavar='1..64')
    parser.add_argument('--exe', default='ocarina.exe' if os.name == 'nt' else 'ocarina')
    args = parser.parse_args()
    extend(args.game, args.workers, args.exe)


if __name__ == '__main__':
    main()
