"""Check exact lazy writer custody against the production locking and page ownership code."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

runtime=Path(os.environ.get("NX2_LAZY_CUSTODY_SOURCE",Path(__file__).resolve().parents[1]/"runtime"))
source=(runtime/"memory.c").read_text(encoding="utf-8")
header=(runtime/"host.h").read_text(encoding="utf-8")
defs=header[header.index("typedef struct MemLazyWriteRange {"):header.index("int mem_lazy_provider_transition_try(void);")]
parts=[source[source.index("static void lazy_write_arm_clip("):source.index("// Arms the whole pages of [a, a + n)")],
       source[source.index("static int lazy_write_fetch_locked("):source.index("int mem_lazy_provider_transition_try(void) {")]]
prefix=r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define PAGE 4096ULL
#define RT_ADDRESS_LIMIT (1ULL<<40)
enum {PLAT_MUTEX_INIT=0};
typedef int PlatMutex;
static void plat_mutex_lock(PlatMutex *p) { assert(!*p);*p=1; }
static void plat_mutex_unlock(PlatMutex *p) { assert(*p);*p=0; }
static void rt_fatal(const char *s){fputs(s,stderr);abort();}
static uint32_t lz_win_tid;
static int lz_want=1, lz_n, busy, fault_lock, drops, real_fetches, refreshes, fail_call, fetch_calls;
static uint64_t lz_fetch_now,lz_fetch_total,lz_fetches,lz_fetch_bytes,lz_fetch_ns;
static uint32_t lz_fetch_tid,lz_fetch_ms;
static uint64_t last_lo,last_hi,last_native,last_fetch,last_fetch_lo,last_fetch_offset;
static int first_owner, foreign_owner;
typedef struct { uint64_t lo,hi,h0,h1; int64_t delta; uint8_t stale; void *provider; } LzSeg;
static LzSeg lzs[16];
static PlatMutex lazy_write_lock = PLAT_MUTEX_INIT;
static MemLazyWriteRange *lazy_writes;
static uint64_t plat_now_ns(void) {return 10;}
static uint32_t plat_now_ms(void) {return 1;}
static uint32_t plat_thread_id(void) {return 2;}
static int fetch(uint64_t mirror,uint64_t import_buffer,uint64_t offset,uint64_t size,uint64_t guest,void *provider) {
    assert(mirror==11 && import_buffer==22 && provider==&first_owner);
    if(++fetch_calls==fail_call) return 0;
    real_fetches++;last_fetch=size;last_fetch_lo=guest;last_fetch_offset=offset;
    return 1;
}
static int (*lz_fetch_verified)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,void *)=fetch;
static void lazy_push_locked(LzSeg seg) { assert(lz_n<16);lzs[lz_n++]=seg; }
static void lazy_flags_locked(uint64_t lo, uint64_t hi) { assert(lo<hi);refreshes++; }
static int mem_lazy_provider_transition_try(void) { if(busy)return 0;assert(!fault_lock); return fault_lock=1; }
static void mem_lazy_provider_transition_end(void) { assert(fault_lock);fault_lock=0; }
static void lazy_drop_locked(int i,int why,uint64_t touch) {
    assert(fault_lock && why==2 && !touch);
    LzSeg s=lzs[i]; drops++;
    last_lo=s.lo;last_hi=s.hi;last_native=s.lo+(uint64_t)s.delta;
    assert(!s.stale); // The production direct drop cannot use the unverified legacy fetch.
    lzs[i]=lzs[--lz_n];
}
"""
cases=r"""
static const uint64_t base=0x26618d000ULL;
static const uint64_t count=0x5b40;
static void reset(void) {
    assert(!lazy_writes && !fault_lock && !lazy_write_lock);
    lz_n=busy=lz_win_tid=drops=real_fetches=refreshes=fail_call=fetch_calls=0; lz_fetch_verified=fetch;lz_want=1;lz_fetches=lz_fetch_total=lz_fetch_bytes=lz_fetch_now=lz_fetch_ns=lz_fetch_ms=lz_fetch_tid=0;
}
static LzSeg segment(uint64_t lo,uint64_t hi,int stale) {
    return (LzSeg){base+lo,base+hi,11,22,-(int64_t)base,stale,&first_owner};
}
static MemLazyWriteResult acquire(MemLazyWriteRange *lease,int parked) {
    return mem_lazy_write_acquire(lease,base,count,11,22,0,&first_owner,parked);
}
static void same_source(int stale) {
    reset();
    MemLazyWriteRange lease={0};
    lazy_push_locked(segment(0x2000,0x9000,stale));
    assert(!lazy_writes);
    if(stale) {
        assert(acquire(&lease,0)==MEM_LAZY_WRITE_NEEDS_FETCH);
        assert(!drops && !real_fetches && lz_n==1 && !lazy_writes);
    }
    assert(acquire(&lease,stale)==MEM_LAZY_WRITE_READY);
    assert(lazy_writes==&lease && !fault_lock && drops==1 && refreshes==1 && lz_n==1);
    assert(last_lo==base+0x2000 && last_hi==base+0x6000 && last_native==0x2000);
    assert(lzs[0].lo==base+0x6000 && lzs[0].hi==base+0x9000 && lzs[0].stale==stale);
    assert(real_fetches==stale && (!stale || last_fetch==0x4000));
    uint64_t end=base+0x20000;
    assert(lazy_write_arm_limit(base,&end,11,0) && end==base+0x6000);
    end=base+0x20000;
    assert(!lazy_write_arm_limit(base+0x6000,&end,11,0x6000) && end==base+0x20000);
    end=0x80010000;
    assert(lazy_write_arm_limit(0x80000000,&end,11,0) && end==0x80006000); // native alias
    end=base;
    assert(!lazy_write_arm_limit(base-0x1000,&end,99,0) && end==base);
    uint64_t cross_end=base+0xa000;
    assert(!lazy_write_arm_limit(base-0x1000,&cross_end,99,0) && cross_end==base);
    mem_lazy_write_release(&lease);
    mem_lazy_write_release(&lease); // completion/cancel cleanup is single-owner, the empty token is safe
    end=base+0x20000;
    assert(!lazy_write_arm_limit(base,&end,11,0) && !lazy_writes);
}
static void controls(void) {
    reset();
    MemLazyWriteRange lease={0},peer={0};
    busy=1; assert(acquire(&lease,0)==MEM_LAZY_WRITE_BUSY && !lazy_writes);
    busy=0;lz_win_tid=5; assert(acquire(&lease,1)==MEM_LAZY_WRITE_BUSY && !lazy_writes);
    lz_win_tid=0;
    lazy_push_locked(segment(0,0x1000,1));
    LzSeg foreign=segment(0x1000,0x2000,0); foreign.h1=23;foreign.provider=&foreign_owner;
    lazy_push_locked(foreign);
    assert(acquire(&lease,1)==MEM_LAZY_WRITE_FOREIGN && !drops && !real_fetches && lz_n==2 && !fault_lock);
    lz_n=0;
    foreign=(LzSeg){0x80000000,0x80001000,11,22,-INT64_C(0x80000000),1,&foreign_owner};
    lazy_push_locked(foreign);
    assert(acquire(&lease,1)==MEM_LAZY_WRITE_FOREIGN && !drops && !real_fetches);
    lz_n=0;
    lazy_push_locked(segment(0,0x1000,1));lz_fetch_verified=NULL;
    assert(acquire(&lease,1)==MEM_LAZY_WRITE_NEEDS_FETCH && !drops && !lazy_writes);
    lz_fetch_verified=fetch;lz_n=0;
    assert(acquire(&lease,0)==MEM_LAZY_WRITE_READY && acquire(&peer,0)==MEM_LAZY_WRITE_READY);
    mem_lazy_write_release(&lease);
    uint64_t end=base+0x20000;
    assert(lazy_write_arm_limit(base,&end,11,0) && lazy_writes==&peer);
    mem_lazy_write_release(&peer);
    assert(!lazy_writes && !drops);
    lz_want=0;busy=1;
    assert(acquire(&lease,0)==MEM_LAZY_WRITE_READY && !lease.size && !lazy_writes);
}
static void failed_fetch_and_untouched_pages(void) {
    reset();
    MemLazyWriteRange lease={0};
    lazy_push_locked(segment(0,0x2000,1));
    lazy_push_locked(segment(0x3000,0x7000,1));
    fail_call=2;
    assert(acquire(&lease,1)==MEM_LAZY_WRITE_FETCH_FAILED);
    assert(fetch_calls==2 && real_fetches==1 && lz_fetches==1 && lz_fetch_total==1 && lz_fetch_bytes==0x2000 && !drops && !refreshes && lz_n==2);
    assert(lzs[0].stale && lzs[1].stale && !lazy_writes && !lease.size && !fault_lock);
    assert(!lz_fetch_tid);
    reset();
    // Start/finish inside the original protected segment. Fetch whole touched pages and leave both
    // untouched sides protected. Native offsets are kept relative to the original provider.
    lazy_push_locked(segment(0,0x9000,1));
    assert(mem_lazy_write_acquire(&lease,base+0x1024,0x4030,11,22,0x1024,&first_owner,1)==MEM_LAZY_WRITE_READY);
    assert(fetch_calls==1 && real_fetches==1 && drops==1 && lz_n==2);
    assert(last_fetch_lo==base+0x1000 && last_fetch_offset==0x1000 && last_fetch==0x5000);
    assert(lzs[0].lo==base && lzs[0].hi==base+0x1000 && lzs[0].stale && lzs[0].provider==&first_owner);
    assert(lzs[1].lo==base+0x6000 && lzs[1].hi==base+0x9000 && lzs[1].stale && lzs[1].provider==&first_owner);
    mem_lazy_write_release(&lease);
}
int main(void) {
    same_source(0);same_source(1);controls();failed_fetch_and_untouched_pages();
    puts("Exact lazy custody: clean/stale affine overlaps, page clip, late guest/native rearm, foreign owner, busy, failed fetch, untouched pages, collision/cancel passed");
}
"""
cc=os.environ.get("CC") or shutil.which("clang")
if not cc:
    raise SystemExit("Set CC to the Clang executable")
cc=Path(cc).resolve()
env=dict(os.environ,PATH=str(cc.parent)+os.pathsep+os.environ.get("PATH",""))
with tempfile.TemporaryDirectory(prefix="lazy-write-custody-") as directory:
    c=Path(directory)/"check.c"
    exe=c.with_suffix(".exe")
    c.write_text("#include <stdint.h>\n"+defs+prefix+"\n".join(parts)+cases,encoding="utf-8")
    subprocess.run([str(cc),"-std=c11","-O1","-Wall","-Wextra","-Werror","-fsanitize=undefined","-fsanitize-trap=all",
                    str(c),"-o",str(exe)],check=True,env=env)
    subprocess.run([str(exe)],check=True,env=env)
