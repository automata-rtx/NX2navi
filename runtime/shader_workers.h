#pragma once

#include <stdint.h>
#include "platform/platform.h"

// A preparation phase owns this queue until it has joined every worker. The launcher changes one
// shared byte. Zero restores the phase default. In-flight variants hold their place until completion.
typedef struct {
    PlatMutex lock;
    PlatCond changed;
    const volatile uint8_t *limit;
    size_t next, count;
    int automatic, active, peak;
} ShaderWork;

static inline int shader_work_open(ShaderWork *w, size_t count, int automatic, int installer) {
    w->count = count;
    w->automatic = automatic > 0 ? automatic : 1;
    char path[PLAT_PATH_MAX * 2];
    if (!installer || !plat_getenv("NX2NAVI_SHADER_WORKER_FILE", path, sizeof path)) return 1;
    PlatFile *f = plat_fs_open(path, 0, NULL);
    if (f) {
        PlatFileId id;
        if (plat_file_id(f, &id) && id.size == 1) w->limit = plat_file_map(f);
        plat_file_close(f);
    }
    return w->limit != NULL;
}

static inline void shader_work_close(ShaderWork *w) {
    if (w->limit) plat_file_unmap((const void *)w->limit, 1);
    w->limit = NULL;
}

static inline size_t shader_work_next(ShaderWork *w, size_t previous) {
    plat_mutex_lock(&w->lock);
    if (previous != SIZE_MAX) w->active--;
    size_t index = SIZE_MAX;
    while (w->next < w->count) {
        int limit = w->limit ? *w->limit : 0;
        if (w->active < (limit ? limit : w->automatic)) {
            index = w->next++;
            w->active++;
            if (w->active > w->peak) w->peak = w->active;
            if (w->next == w->count) plat_cond_broadcast(&w->changed);
            break;
        }
        // A higher limit can arrive while all active workers are inside a long driver call.
        plat_cond_wait_ms(&w->changed, &w->lock, 100);
    }
    plat_mutex_unlock(&w->lock);
    return index;
}
