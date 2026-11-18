// Freshness observations over bounded texture subresources. The memory tracker owns generation and mapping facts;
// callers publish a stamp only after preparing contents from the observation and validating the mapping again.
#pragma once
#include <stdint.h>
#include <string.h>

enum { TEXTURE_SOURCE_MAX_SPANS = 256, TEXTURE_SOURCE_MASK_WORDS = 4 };
enum TextureSourceState { TEX_SOURCE_UNKNOWN = -1, TEX_SOURCE_CLEAN = 0, TEX_SOURCE_DIRTY = 1 };

typedef struct { uint64_t offset, size; } TextureSourceSpan;
typedef struct {
    uint64_t mapping, protection, lineage;
    uint32_t generation;
} TextureSourceStamp;
typedef struct {
    TextureSourceStamp stamp;
    uint64_t dirty[TEXTURE_SOURCE_MASK_WORDS];
} TextureSourceObservation;

static inline int texture_source_stamp_valid(const TextureSourceStamp *s) {
    return s && s->mapping && s->protection && s->lineage && s->generation;
}

static inline int texture_source_spans_valid(uint64_t size, const TextureSourceSpan *spans, uint32_t count) {
    if (!size || !spans || !count || count > TEXTURE_SOURCE_MAX_SPANS) return 0;
    for (uint32_t i = 0; i < count; i++)
        if (!spans[i].size || spans[i].offset >= size || spans[i].size > size - spans[i].offset) return 0;
    return 1;
}

static inline void texture_source_all_dirty(TextureSourceObservation *o, uint32_t count) {
    memset(o->dirty, 0, sizeof o->dirty);
    if (count > TEXTURE_SOURCE_MAX_SPANS) count = TEXTURE_SOURCE_MAX_SPANS;
    for (uint32_t i = 0; i < count; i++) o->dirty[i >> 6] |= UINT64_C(1) << (i & 63);
}

static inline int texture_source_span_dirty(const TextureSourceObservation *o, uint32_t index) {
    return index < TEXTURE_SOURCE_MAX_SPANS && ((o->dirty[index >> 6] >> (index & 63)) & 1);
}

// Relative byte bounds of an observed dirty page, clipped to the texture. Overlapping source regions both become
// dirty. A write outside every declared region conservatively invalidates them all: incomplete layout coverage
// must not silently become proof that a source-content identity is still current.
static inline void texture_source_mark_range(TextureSourceObservation *o, const TextureSourceSpan *spans,
                                              uint32_t count, uint64_t lo, uint64_t hi) {
    int covered = 0;
    for (uint32_t i = 0; i < count; i++)
        if (lo < spans[i].offset + spans[i].size && spans[i].offset < hi) {
            o->dirty[i >> 6] |= UINT64_C(1) << (i & 63);
            covered = 1;
        }
    if (!covered) texture_source_all_dirty(o, count);
}
