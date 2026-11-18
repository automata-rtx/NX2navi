"""Check rendered samples and voice accounting without game files or a playback device.

Windows: python tests/audio_voice_check.py --toolchain C:/msys64/ucrt64
Linux: python tests/audio_voice_check.py (Clang and SDL3 development files)
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--toolchain', type=Path)
args = parser.parse_args()
env = os.environ.copy()
out = root / 'build' / 'audio-voice-check'
out.mkdir(parents=True, exist_ok=True)
if args.toolchain:
    env['PATH'] = str(args.toolchain / 'bin') + os.pathsep + env['PATH']
    cc = args.toolchain / 'bin' / 'clang.exe'
    libs = ['-L', str(args.toolchain / 'lib'), '-lSDL3']
else:
    cc = 'clang'
    libs = shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'sdl3'], text=True))
exe = out / ('audio-voice-check.exe' if os.name == 'nt' else 'audio-voice-check')
subprocess.run([cc, '-std=gnu11', '-O2', '-fno-strict-aliasing', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                '-I', root / 'runtime', root / 'tests/audio_voice_check.c',
                root / 'runtime/audio_effects.c', root / 'runtime/audio_reverb.c', '-o', exe, *libs, '-lm'],
               env=env, check=True)
subprocess.run([exe], env=env, check=True, timeout=15)
