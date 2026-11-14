// Check only the 512 pages of the proposed 2 MiB freeze. A missing page, guard,
// different protection or unsupported query leaves discovery to the existing walker.
#pragma once
#include <stdint.h>
#include <windows.h>
#include <psapi.h>

static int freeze_resident_rw(uint64_t start) {
    enum { page_size = 4096, page_count = 512 };
    PSAPI_WORKING_SET_EX_INFORMATION pages[page_count];
    for (unsigned i = 0; i < page_count; ++i)
        pages[i].VirtualAddress = (void *)(uintptr_t)(start + i * page_size);
    if (!QueryWorkingSetEx(GetCurrentProcess(), pages, sizeof pages)) return 0;
    for (unsigned i = 0; i < page_count; ++i) {
        PSAPI_WORKING_SET_EX_BLOCK p = pages[i].VirtualAttributes;
        if (!p.Valid || p.Win32Protection != PAGE_READWRITE || p.Bad || p.LargePage) return 0;
    }
    return 1;
}
