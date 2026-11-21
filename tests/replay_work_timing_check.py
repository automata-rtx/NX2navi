"""Check production replay diagnostic accounting with controlled clocks, without a game or GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Set CC to MSYS2 UCRT64 Clang")
    env = dict(os.environ)
    env["PATH"] = str(Path(cc).resolve().parent) + os.pathsep + env.get("PATH", "")
    parent = ROOT / "build/test"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replay-work-", dir=parent) as directory:
        for enabled,chain in ((0,0),(1,0),(1,1)):
            exe = Path(directory) / (f"check-{enabled}-{chain}" + (".exe" if os.name == "nt" else ""))
            subprocess.run([cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fsanitize-trap=all",
                            f"-DNX2NAVI_REPLAY_WORK={enabled}", f"-DNX2NAVI_REPLAY_WORK_POOL={enabled}",
                            f"-DNX2NAVI_REPLAY_WORK_CHAIN={chain}",
                            "-I", str(ROOT / "runtime"), str(ROOT / "tests/replay_work_timing_check.c"),
                            "-o", str(exe)], check=True, env=env)
            subprocess.run([str(exe)], check=True, env=env)
        if os.name == "nt":
            # Use the production Windows clock conversion, and time the actual region functions. This is host
            # calibration evidence with explicit pair counts, not a correction applied to renderer timings.
            platform=(ROOT/'runtime/platform/win32.c').read_text()
            calibration=r'''
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <assert.h>
#include <stdio.h>
#include "platform/platform.h"
#include "replay_work_trace.h"
#define TL_CLOSED (1u << 31)
static PlatScale qpc_ns;
'''
            for name in ('qpc_freq','qpc_ns_init','plat_now_ns'):
                calibration+=function(platform,name).replace('plat_now_ns(', 'raw_now_ns(')+'\n'
            calibration+=r'''
static uint64_t clock_calls;
uint64_t plat_now_ns(void){clock_calls++;return raw_now_ns();}
uint32_t plat_thread_id(void){return 7;}
uint64_t plat_thread_cpu_ns(uint32_t tid){(void)tid;return 1;}
static int tl_recording(void){return 1;}
static void rt_log(const char *format,...){(void)format;}
#include "replay_work_impl.h"
int main(void){
 qpc_ns_init();enum {PAIRS=100000,TRIALS=5};
 uint64_t raw_min=UINT64_MAX,raw_max=0,region_min=UINT64_MAX,region_max=0;
 volatile uint64_t sink=0;
 for(unsigned trial=0;trial<TRIALS;trial++){
  uint64_t start=raw_now_ns();clock_calls=0;
  for(unsigned i=0;i<PAIRS;i++){uint64_t a=plat_now_ns(),b=plat_now_ns();sink+=b-a;}
  uint64_t elapsed=raw_now_ns()-start;assert(clock_calls==2*PAIRS);
  if(elapsed<raw_min)raw_min=elapsed;if(elapsed>raw_max)raw_max=elapsed;
  RwtUnit u={0};u.last=raw_now_ns();rwt_current=&u;clock_calls=0;start=raw_now_ns();
  for(unsigned i=0;i<PAIRS;i++){
   RwtRegionScope s=gfx_rwt_region_begin(RWT_REGION_DRAW_BIND);gfx_rwt_region_end(&s);
  }
  elapsed=raw_now_ns()-start;rwt_current=NULL;
  assert(clock_calls==2*PAIRS&&!u.invalid&&!u.region_depth&&u.region_calls[RWT_REGION_DRAW_BIND]==PAIRS);
  if(elapsed<region_min)region_min=elapsed;if(elapsed>region_max)region_max=elapsed;
 }
 printf("replay calibration: %u trials x %u pairs, %u clock calls/trial; clock pair %.1f..%.1f ns, region pair %.1f..%.1f ns; host microprobe only\n",
  TRIALS,PAIRS,2*PAIRS,(double)raw_min/PAIRS,(double)raw_max/PAIRS,(double)region_min/PAIRS,(double)region_max/PAIRS);
 return sink==UINT64_MAX;
}
'''
            path=Path(directory)/'clock.c';path.write_text(calibration)
            exe=path.with_suffix('.exe')
            subprocess.run([cc,'-std=gnu11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-function',
                '-DNX2NAVI_REPLAY_WORK=1','-DNX2NAVI_REPLAY_WORK_CHAIN=1','-I',str(ROOT/'runtime'),
                str(path),'-o',str(exe)],check=True,env=env)
            subprocess.run([str(exe)],check=True,env=env)


if __name__ == "__main__":
    main()
