#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t load_word, image_word, query_word;
    uint32_t result_type;
} RcSampledImageQuerySite;

typedef struct {
    uint32_t variable_id;
    size_t site_count;
} RcSampledImageQueryProof;

typedef enum {
    RC_SAMPLED_IMAGE_QUERY_NONE,
    RC_SAMPLED_IMAGE_QUERY_PROVEN,
    RC_SAMPLED_IMAGE_QUERY_CAPACITY,
} RcSampledImageQueryResult;

typedef enum {
    RC_SAMPLED_IMAGE_QUERY_VERTEX = 0,
    RC_SAMPLED_IMAGE_QUERY_FRAGMENT = 4,
    RC_SAMPLED_IMAGE_QUERY_COMPUTE = 5,
} RcSampledImageQueryStage;

#define RC_SAMPLED_IMAGE_QUERY_ALL_MEMBERS UINT32_MAX

typedef struct {
    RcSampledImageQueryStage stage;
    uint32_t set, binding;
    uint32_t table_count, member;
} RcSampledImageQueryTarget;

// Prove LOD-zero size queries in these exact, validated SPIR-V words. A direct
// combined descriptor has table_count == 0 and member == 0. A table is a bare
// sampled-image array with the supplied descriptor count. A literal member
// requires every table access to have an in-range literal index. ALL_MEMBERS
// allows dynamic indices and proves every access is query-only; the caller must
// separately prove the complete runtime domain and its descriptor bounds.
// A bare-image query without OpImage reports image_word == load_word.
RcSampledImageQueryResult rc_sampled_image_query_size_lod0_for(const uint32_t *words, size_t word_count,
    RcSampledImageQueryTarget target, RcSampledImageQueryProof *proof,
    RcSampledImageQuerySite *sites, size_t site_capacity);

// Prove all uses of one exact combined 2D binding in these SPIR-V words. The
// input is a validated module. This bounded recognizer accepts a single vertex
// main, non-depth, non-array, non-MS float images and direct LOD-zero size queries.
// A NULL sites pointer with zero capacity requests the count only. CAPACITY
// reports the required count without writing any sites. NONE clears the proof.
// Offsets and SPIR-V IDs refer to these exact words, including binding remaps.
RcSampledImageQueryResult rc_sampled_image_query_size_lod0(const uint32_t *words, size_t word_count,
    uint32_t set, uint32_t binding, RcSampledImageQueryProof *proof,
    RcSampledImageQuerySite *sites, size_t site_capacity);
