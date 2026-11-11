// File publication and large-file offsets. Pass a scratch path; a second argument exercises sizes above 4 GiB.
// Link with platform/win32.c or platform/linux.c using function sections and the linker's section garbage collection.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "platform/platform.h"

static void check_bytes(const char *path, size_t size) {
    PlatFile *f = plat_file_open(path);
    PlatFileId id;
    assert(f && plat_file_id(f, &id) && id.size == size);
    const unsigned char *p = plat_file_map(f);
    assert(p && p[0] == 0x12 && p[size - 1] == 0x34);
    if (size > UINT32_MAX) assert(p[UINT32_MAX] == 0x56 && p[(size_t)UINT32_MAX + 1] == 0x78);
    plat_file_unmap(p, size);
    plat_file_close(f);
}

int main(int argc, char **argv) {
    assert(argc >= 2);
    const char *path = argv[1];
    size_t size = argc > 2 ? (size_t)UINT32_MAX + 4097 : 8192;
    unsigned char old[] = {0x12, 0x34};
    assert(plat_file_write_atomic(path, old, sizeof old));
    void *data;
    PlatFileWrite *f = plat_file_write_begin(path, size, &data);
    assert(f && data);
    check_bytes(path, sizeof old); // The previous file stays readable until publication.
    unsigned char *p = data;
    p[0] = 0x12;
    p[size - 1] = 0x34;
    if (size > UINT32_MAX) p[UINT32_MAX] = 0x56, p[(size_t)UINT32_MAX + 1] = 0x78;
    assert(plat_file_write_finish(f, size));
    check_bytes(path, size);
    f = plat_file_write_begin(path, 8192, &data);
    assert(f && data);
    memset(data, 0xff, 8192);
    assert(!plat_file_write_finish(f, 0)); // Abort must preserve the previous complete cache.
    check_bytes(path, size);
    f = plat_file_write_begin(path, 8192, &data);
    assert(f && data);
    memcpy(data, old, sizeof old);
    assert(plat_file_write_finish(f, sizeof old)); // Exact driver-returned size, not the initial capacity.
    check_bytes(path, sizeof old);
    assert(!plat_file_write_begin(path, 0, &data) && !data);
    // Exercise the ordinary writer above DWORD range too, using the mapped source as its input.
    if (argc > 2) {
        char source[4096];
        snprintf(source, sizeof source, "%s.source", path);
        f = plat_file_write_begin(source, size, &data);
        assert(f && data);
        p = data;
        p[0] = 0x12, p[size - 1] = 0x34, p[UINT32_MAX] = 0x56, p[(size_t)UINT32_MAX + 1] = 0x78;
        assert(plat_file_write_atomic(path, data, size));
        assert(!plat_file_write_finish(f, 0));
        check_bytes(path, size);
        assert(plat_file_set_readonly(path, 1));
        f = plat_file_write_begin(path, 4096, &data);
        if (f) assert(!plat_file_write_finish(f, 4096));
        assert(plat_file_set_readonly(path, 0));
        check_bytes(path, size);
    }
    plat_file_delete(path);
    puts("File mapping, atomic publication, abort, exact size and boundary reads passed");
    return 0;
}
