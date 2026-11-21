// Link in place of runtime/tex_probe.c and run --probe-vkmem with WB_LOCAL_HOLD=1,
// WB_LAZY=1 and WB_LAZY_CHUNK=0. The normal memory initialization installs the fault handler.
#include "host.h"
#include <stdio.h>
#include <string.h>

static uint64_t calls, offset_seen, bytes_seen, guest_seen;
static int clean = 1;

static void fetch(uint64_t mirror, uint64_t destination, uint64_t offset, uint64_t bytes, uint64_t guest_addr,
                  void *provider) {
    if (mirror != 11 || destination != 22 || provider != NULL) rt_fatal("write-back test: wrong buffer");
    calls++;
    offset_seen = offset;
    bytes_seen = bytes;
    guest_seen = guest_addr;
}

static int check_clean(uint64_t lo, uint64_t hi, void *ctx) { return clean; }

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "write-back test failed: %s\n", #expr); return 1; } } while (0)

int probe_vkmem(void) {
    uint64_t base = 0, skipped[4][2];
    const uint64_t piece = 1u << 16, length = 4u << 20;
    CHECK(!mem_set_heap_size(8u << 20, &base));
    base = (base + piece - 1) & ~(piece - 1);
    memset((void *)(uintptr_t)base, 37, length);
    CHECK(mem_lazy_enable(fetch, check_clean, NULL));
    CHECK(mem_lazy_arm(base, length, 11, 22, 0, NULL) == length);
    CHECK(mem_lazy_skip(base, length, 11, -(int64_t)base, NULL, skipped, 4) == 1);
    // A real read fault has to return the full segment before the original instruction retries.
    volatile uint8_t *address = (volatile uint8_t *)(uintptr_t)(base + piece + 47);
    CHECK(*address == 37 && calls == 1 && !offset_seen && bytes_seen == length && guest_seen == base);
    CHECK(!mem_lazy_armed(base, length));
    clean = 0;
    CHECK(!mem_lazy_arm(base, length, 11, 22, 0, NULL));
    CHECK(!mem_lazy_skip(base, length, 11, -(int64_t)base, NULL, skipped, 4));
    clean = 1;
    CHECK(mem_lazy_arm(base, length, 11, 22, 0, NULL) == length - piece);
    CHECK(mem_lazy_skip(base, length, 11, -(int64_t)base, NULL, skipped, 4) == 2);
    CHECK(skipped[0][0] == base && skipped[0][1] == base + piece);
    CHECK(skipped[1][0] == base + 2 * piece && skipped[1][1] == base + length);
    *address = 89;
    CHECK(calls == 1 && *address == 89);
    // A write in the other segment fetches that entire segment and retries the store.
    address = (volatile uint8_t *)(uintptr_t)(base + 5 * piece);
    *address = 91;
    CHECK(*address == 91 && calls == 2 && offset_seen == 2 * piece && bytes_seen == length - 2 * piece && guest_seen == base + 2 * piece);
    mem_lazy_release(base, length);
    CHECK(calls == 3 && !offset_seen && bytes_seen == piece && guest_seen == base);
    CHECK(!mem_lazy_skip(base, length, 11, -(int64_t)base, NULL, skipped, 4));
    puts("write-back test passed: full fetch, address-local cooldown, dirty refusal, disjoint rearm, write retry and release");
    return 0;
}
