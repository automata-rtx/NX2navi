"""Production lazy full/chunk release forwards the exact guest address to its visibility adapter."""
from pathlib import Path
import os
import subprocess
import tempfile
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/memory.c").read_text()
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#define FRZ_SHIFT 16
#define LZ_CHUNK (1ull<<12)
enum {PLAT_PROT_NONE,PLAT_PROT_RW};
typedef struct {uint64_t lo,hi,h0,h1;int64_t delta;uint8_t stale;void *provider;} LzSeg;
static LzSeg lzs[8];static int lz_n;
static uint64_t lz_fetch_now,lz_fetch_total,lz_fetches,lz_fetch_bytes,lz_fetch_ns,lz_other,lz_faults[2],lz_pieces,lz_piece_bytes;
static uint32_t lz_fetch_tid,lz_fetch_ms,lz_hot[16];static uint8_t lz_back[16];
static uint64_t guest_seen,offset_seen,size_seen;static int calls,first,second;
static void *provider_seen;
static void fetch(uint64_t a,uint64_t b,uint64_t off,uint64_t size,uint64_t guest,void *provider) {
    assert(a==11&&b==22);calls++;offset_seen=off;size_seen=size;guest_seen=guest;provider_seen=provider;
}
static void (*lz_fetch)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,void *)=fetch;
static uint64_t plat_now_ns(void) {return 10;}
static uint32_t plat_now_ms(void) {return 1;}
static uint32_t plat_thread_id(void) {return 1;}
static int plat_mem_state(uint64_t a,uint64_t end,uint64_t *next) {(void)a;*next=end;return PLAT_PROT_NONE;}
static int plat_mem_protect(uint64_t a,uint64_t n,int p) {(void)a;(void)n;(void)p;return 1;}
static void rt_fatal(const char *s,...) {(void)s;abort();}
static void lazy_flags_locked(uint64_t a,uint64_t b) {(void)a;(void)b;}
static void lazy_hold_piece_locked(uint64_t a,uint32_t now) {(void)a;(void)now;}
static void lazy_push_locked(LzSeg s) {assert(lz_n<8);lzs[lz_n++]=s;}
'''
fixture += function(source, "lazy_drop_locked") + function(source, "lazy_piece_locked")
fixture += r'''
int main(void) {
    lzs[0]=(LzSeg){0x10000,0x14000,11,22,-0x10000,1,&first};lz_n=1;
    lazy_drop_locked(0,0,0);
    assert(calls==1&&guest_seen==0x10000&&!offset_seen&&size_seen==0x4000&&!lz_n&&provider_seen==&first);
    lzs[0]=(LzSeg){0x20000,0x24000,11,22,-0x20000,1,&second};lz_n=1;
    assert(lazy_piece_locked(0,0x21500,1)==0x1000);
    assert(calls==2&&guest_seen==0x21000&&offset_seen==0x1000&&size_seen==0x1000&&lz_n==2&&provider_seen==&second);
    assert(lzs[0].provider==&second && lzs[1].provider==&second);
    assert(lzs[0].lo==0x20000&&lzs[0].hi==0x21000&&lzs[1].lo==0x22000&&lzs[1].hi==0x24000);
    lzs[0].stale=0;lazy_drop_locked(0,2,0);assert(calls==2); // No fabricated copy/visibility proof for an already-current segment.
    assert(lz_fetch_total==2&&lz_fetches==2&&lz_fetch_bytes==0x5000&&lz_piece_bytes==0x1000&&lz_pieces==1);
    assert(lz_other==1&&lz_faults[0]==1&&lz_faults[1]==1&&!lz_fetch_tid&&lz_fetch_ms&&lz_fetch_now==0x1000&&!lz_fetch_ns);
    puts("lazy address handoff: exact whole/chunk guest span and no-copy path passed");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="lazy-address-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "lazy.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)
