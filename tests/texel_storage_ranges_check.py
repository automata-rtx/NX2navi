"""Compare copied bytes with the exact storage ranges for both NVN GOB layouts."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
fixture = r"""
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tex.h"
const int rt_nvn_api_version[2] = {NVN_TEST_VERSION, 0};
const char *cfg_get(const char *key) { (void)key; return NULL; }
typedef struct {
    unsigned char *seen;
    uint64_t capacity, end, bytes;
} Coverage;
static void note(void *context, uint64_t offset, uint64_t size) {
    Coverage *c = context;
    assert(size && offset >= c->end && offset <= c->capacity && size <= c->capacity - offset);
    for (uint64_t i = offset; i < offset + size; i++) assert(!c->seen[i]);
    memset(c->seen + offset, 1, (size_t)size);
    c->end = offset + size;
    c->bytes += size;
}
static void test(nvn_texdesc d, unsigned level, unsigned layer, uint32_t x, uint32_t y,
                 uint32_t z, uint32_t width, uint32_t height) {
    nvn_texlayout layout;
    nvn_tex_layout(&d, &layout);
    assert(layout.size && layout.size < 8u * 1024 * 1024);
    unsigned char *storage = malloc(layout.size);
    unsigned char *rows = malloc((size_t)width * height);
    Coverage coverage = {.seen = calloc(layout.size, 1), .capacity = layout.size};
    assert(storage && rows && coverage.seen);
    memset(storage, 0x13, layout.size);
    memset(rows, 0xa5, (size_t)width * height);
    assert(!nvn_texels_box(&d, level, layer, x, y, z, width, height, storage, rows, width, 0));
    assert(!nvn_texels_box_ranges(&d, level, layer, x, y, z, width, height, note, &coverage));
    assert(coverage.bytes == (uint64_t)width * height);
    for (uint64_t i = 0; i < layout.size; i++) assert(storage[i] == (coverage.seen[i] ? 0xa5 : 0x13));
    uint64_t transferred = coverage.bytes;
    assert(nvn_texels_box_ranges(&d, level, layer, x, y, z, UINT32_MAX, height, note, &coverage) == -1);
    assert(coverage.bytes == transferred);
    free(coverage.seen);
    free(rows);
    free(storage);
}
static void test_sample(nvn_texdesc d, unsigned level, unsigned layer, uint32_t row, uint32_t height) {
    nvn_texlayout layout;
    nvn_tex_layout(&d, &layout);
    assert(layout.size);
    size_t bytes=(size_t)row*height;
    uint8_t *storage=malloc(layout.size), *sample=calloc(layout.size,1);
    uint8_t *pixels=malloc(bytes), *decoded=malloc(bytes);
    Coverage coverage={.seen=calloc(layout.size,1),.capacity=layout.size};
    assert(storage && sample && pixels && decoded && coverage.seen);
    memset(storage,0xbd,layout.size);
    for (size_t i=0;i<bytes;i++) pixels[i]=(uint8_t)(1+i%181);
    assert(!nvn_swizzle(&d,level,layer,pixels,row,storage));
    assert(!nvn_texels_slice_ranges(&d,level,layer,0,note,&coverage));
    assert(coverage.bytes==bytes);
    for (uint64_t i=0;i<layout.size;i++) {
        assert((storage[i]!=0xbd)==(coverage.seen[i]!=0));
        if (coverage.seen[i]) sample[i]=storage[i];
    }
    assert(!nvn_deswizzle(&d,level,layer,sample,decoded,row));
    assert(!memcmp(decoded,pixels,bytes));
    free(coverage.seen);free(decoded);free(pixels);free(sample);free(storage);
}
int main(void) {
    nvn_texdesc d = {.target=1, .format=0x25, .width=71, .height=19, .levels=2};
    test(d, 0, 0, 0, 0, 0, 284, 19);
    test(d, 0, 0, 15, 7, 0, 133, 11);
    test_sample(d, 0, 0, 284, 19);
    d.flags = 0x10;
    d.stride = 320;
    test(d, 0, 0, 13, 1, 0, 42, 17);
    test_sample(d, 0, 0, 284, 19);
    d = (nvn_texdesc){.target=2, .format=0x25, .width=73, .height=34, .depth=7, .levels=3};
    test(d, 1, 0, 3, 5, 1, 127, 11);
    d = (nvn_texdesc){.target=4, .format=0x25, .width=50, .height=28, .depth=3, .levels=3};
    test(d, 1, 1, 11, 1, 0, 82, 11);
    test_sample(d, 1, 1, 100, 14);
    return 0;
}
"""
work = ROOT / "build/test"
work.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="texel-ranges-", dir=work) as temporary:
    path = Path(temporary) / "check.c"
    path.write_text(fixture)
    for api in (1, 200):
        exe = path.with_name(f"texel_{api}.exe")
        subprocess.run([str(CC), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-DNVN_TEST_VERSION={api}",
                        "-I", str(ROOT / "runtime"), "-I", str(CC.parent.parent / "include/vma"),
                        str(path), str(ROOT / "runtime/tex_deswizzle.c"), str(ROOT / "runtime/nvn_tex.c"),
                        "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)
print("texel ranges: exact copied bytes for linear pitch, NX1/NVN2, partial GOBs, mip/array and 3D slices")
