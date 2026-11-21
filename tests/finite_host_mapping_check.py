"""Verify finite guest mapping validation against a waiting Windows native-pool writer."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
from bind_recipe_check import function

if os.name != 'nt':
    raise SystemExit('Windows SRW contention fixture')

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'runtime/memory.c').read_text()
name='mem_mapping_validate_read_ranges'
start=re.search(r'^uint64_t '+name+r'\([^;{}]+\)\s*\{', source,re.M).start()
validation=function(source[start:],name)
setup=r"""
#include <assert.h>
#include <windows.h>
#include <stdint.h>
#include "mem_mapping.h"
typedef SRWLOCK PlatMutex;
static PlatMutex lock=SRWLOCK_INIT, host_backing_topology_lock=SRWLOCK_INIT, device_pool=SRWLOCK_INIT;
static uint64_t regions[8];
static MemMapping mapping={regions,2,8,21,0,NULL};
typedef struct { uint64_t identity; } MemBackingCapture;
typedef struct {
    const MemBackingCapture *backing;
    uint64_t address, bytes;
    int needs_readable;
} MemMappingReadRange;
static uint64_t incarnation[8]={7,7,7,7,7,7,7,7};
static uint64_t unreadable;
static HANDLE ready, done;
static void plat_mutex_lock_shared(PlatMutex *lock) { AcquireSRWLockShared(lock); }
static void plat_mutex_unlock_shared(PlatMutex *lock) { ReleaseSRWLockShared(lock); }
static uint64_t mem_mapping_serial(void) { return mem_mapping_clock(&mapping); }
static int mem_mapping_range_unchanged(uint64_t a,uint64_t n,uint64_t before) {
    return mem_mapping_unchanged(&mapping,a,n,before);
}
static int host_backing_range_current_locked(const MemBackingCapture *lease,uint64_t address,uint64_t n) {
    assert(!TryAcquireSRWLockExclusive(&lock));
    assert(!TryAcquireSRWLockExclusive(&host_backing_topology_lock));
    return n==4 && incarnation[address>>21]==lease->identity;
}
static int readable_locked(uint64_t address,uint64_t n) {
    return n==4 && unreadable!=address;
}
"""
test=r"""
// Vulkan backing work owns the native pool first. A host validation must never need that lock.
static DWORD WINAPI mutate(void *unused) {
    (void)unused;
    AcquireSRWLockExclusive(&device_pool);
    AcquireSRWLockExclusive(&lock);
    AcquireSRWLockExclusive(&host_backing_topology_lock);
    assert(SetEvent(ready));
    mem_mapping_publish(&mapping,6u<<21,4,1);
    incarnation[6]++;
    mem_mapping_publish(&mapping,6u<<21,4,0);
    ReleaseSRWLockExclusive(&host_backing_topology_lock);
    ReleaseSRWLockExclusive(&lock);
    assert(WaitForSingleObject(done,5000)==WAIT_OBJECT_0);
    ReleaseSRWLockExclusive(&device_pool);
    return 0;
}

int main(void) {
    const MemBackingCapture exact={7};
    const MemMappingReadRange inputs[]={{&exact,0x200040,4,1},{0,0x400040,4,0}};
    uint64_t before=mem_mapping_serial();
    ready=CreateEventW(NULL,TRUE,FALSE,NULL);
    done=CreateEventW(NULL,TRUE,FALSE,NULL);
    HANDLE thread=CreateThread(NULL,0,mutate,NULL,0,NULL);
    assert(ready&&done&&thread&&WaitForSingleObject(ready,5000)==WAIT_OBJECT_0);
    uint64_t verified=mem_mapping_validate_read_ranges(inputs,2,before);
    assert(verified>before && !TryAcquireSRWLockExclusive(&device_pool));
    assert(SetEvent(done)&&WaitForSingleObject(thread,5000)==WAIT_OBJECT_0);
    CloseHandle(thread);
    CloseHandle(ready);
    CloseHandle(done);

    // Actual remap cannot qualify even if the cohort's legacy watermark still matches.
    incarnation[1]++;
    assert(!mem_mapping_validate_read_ranges(inputs,2,mem_mapping_serial()));
    incarnation[1]--;
    unreadable=inputs[0].address;
    assert(!mem_mapping_validate_read_ranges(inputs,2,mem_mapping_serial()));
    unreadable=0;
    // A proven absence has no backing to compare. A change in its checked region refuses.
    mem_mapping_publish(&mapping,inputs[1].address,inputs[1].bytes,1);
    mem_mapping_publish(&mapping,inputs[1].address,inputs[1].bytes,0);
    assert(!mem_mapping_validate_read_ranges(inputs,2,verified));
    assert(mem_mapping_validate_read_ranges(inputs,2,mem_mapping_serial()));
    return 0;
}
"""
cc=Path(os.environ['CC'])
includes = ROOT/'runtime'
with tempfile.TemporaryDirectory(prefix='finite-guest-map-') as temporary:
    c=Path(temporary)/'check.c'
    exe=c.with_suffix('.exe')
    c.write_text(setup+validation+test)
    env=dict(os.environ,PATH=str(cc.parent)+os.pathsep+os.environ['PATH'])
    subprocess.run([str(cc),'-std=gnu11','-Wall','-Wextra','-Werror','-fsanitize=undefined',
                    '-fsanitize-trap=all','-I',str(includes),str(c),'-o',str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True,timeout=12)
print('Finite host mapping valid with unrelated blocking native-pool writer; selected remap, protection and changed absence refused')
