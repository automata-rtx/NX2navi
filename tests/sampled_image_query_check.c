#include "rc_sampled_image_query.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    V = 1, BOOL, I32, F32, I2, F2, F4, IMAGE, SAMPLED, POINTER, MAIN_TYPE, HELPER_TYPE,
    TRUE_ID, ZERO_ID, ONE_ID, FLOAT_ZERO, ICOORD, FCOORD, MAIN, HELPER, HELPER_ARG,
    ENTRY_LABEL, THEN_LABEL, ELSE_LABEL, MERGE_LABEL, HELPER_LABEL,
    TEX_A = 60, TEX_B, A_LOAD_THEN, A_IMAGE_THEN, A_QUERY_THEN, B_LOAD_THEN, B_IMAGE_THEN, B_QUERY_THEN,
    A_LOAD_ELSE, A_IMAGE_ELSE, A_QUERY_ELSE, B_LOAD_ELSE, B_IMAGE_ELSE, B_QUERY_ELSE,
    EXTRA, EXTRACT_THEN, EXTRACT_ELSE, NOISE, HELPER_LOAD, HELPER_IMAGE, HELPER_FETCH, BOUND = 102,
};

typedef enum {
    MATCHED, FETCH, SAMPLE, CALL, COPY, NONZERO_LOD, DYNAMIC_LOD, SPEC_LOD, DEPTH, ARRAY,
    MSAA, DUPLICATE, SECOND_ENTRY, FOREIGN_LITERAL, BAD_RESULT, ARRAY_DESCRIPTOR, HELPER_ESCAPE,
    MISSING_RECONVERGENCE_EXTENSION, FOREIGN_RECONVERGENCE_EXTENSION, FOREIGN_RECONVERGENCE_ENTRY,
    MALFORMED_RECONVERGENCE_MODE, DUPLICATE_RECONVERGENCE_MODE, RECONVERGENCE_IMAGE_ESCAPE, TRUNCATED,
    RECONVERGENCE,
} Variant;

typedef struct {uint32_t words[640]; size_t count;} Module;

static void emit(Module *m, uint32_t opcode, const uint32_t *operands, size_t count) {
    assert(m->count + count + 1 <= sizeof m->words / sizeof m->words[0]);
    m->words[m->count++] = (uint32_t)((count + 1) << 16) | opcode;
    if (count) memcpy(m->words + m->count, operands, count * sizeof *operands);
    m->count += count;
}

#define EMIT(op, ...) emit(&m, op, (const uint32_t[]){__VA_ARGS__}, sizeof((const uint32_t[]){__VA_ARGS__}) / sizeof(uint32_t))
#define BARE(op) emit(&m, op, NULL, 0)

static void add_reconvergence_extension(Module *m, int foreign) {
    const uint32_t words[] = {0x5f565053, 0x5f52484b, 0x6978616d, 0x5f6c616d, 0x6f636572, 0x7265766e,
                              0x636e6567, foreign ? 0x66 : 0x65};
    emit(m, 10, words, sizeof words / sizeof *words);
}

static Module stage_module(Variant variant, uint32_t set, uint32_t a_binding, uint32_t b_binding,
                           RcSampledImageQueryStage stage) {
    Module m = {{0x07230203, 0x00010300, 0, BOUND, 0}, 5};
    EMIT(17, 1);  // Shader
    EMIT(17, 50); // ImageQuery
    const int reconvergence = variant >= MISSING_RECONVERGENCE_EXTENSION && variant != TRUNCATED;
    if (reconvergence && variant != MISSING_RECONVERGENCE_EXTENSION)
        add_reconvergence_extension(&m, variant == FOREIGN_RECONVERGENCE_EXTENSION);
    EMIT(14, 0, 1);  // Logical GLSL450
    EMIT(15, stage, MAIN, 0x6e69616d, 0);
    if (variant == SECOND_ENTRY) EMIT(15, stage, MAIN, 0x7268746f, 0);
    if (reconvergence) {
        const uint32_t entry = variant == FOREIGN_RECONVERGENCE_ENTRY ? HELPER : MAIN;
        if (variant == MALFORMED_RECONVERGENCE_MODE) EMIT(16, entry, 6023, TEX_A);
        else EMIT(16, entry, 6023);
        if (variant == DUPLICATE_RECONVERGENCE_MODE) EMIT(16, entry, 6023);
    }
    if (stage == RC_SAMPLED_IMAGE_QUERY_FRAGMENT) EMIT(16, MAIN, 7); // OriginUpperLeft
    if (stage == RC_SAMPLED_IMAGE_QUERY_COMPUTE) EMIT(16, MAIN, 17, 1, 1, 1); // LocalSize
    EMIT(5, TEX_A, 0x00000061); // OpName "a"
    EMIT(71, TEX_A, 34, set);
    EMIT(71, TEX_A, 33, a_binding);
    EMIT(71, TEX_B, 34, set);
    EMIT(71, TEX_B, 33, variant == DUPLICATE ? a_binding : b_binding);
    EMIT(19, V);
    EMIT(20, BOOL);
    EMIT(21, I32, 32, 1);
    EMIT(22, F32, 32);
    EMIT(23, I2, I32, 2);
    EMIT(23, F2, F32, 2);
    EMIT(23, F4, F32, 4);
    EMIT(25, IMAGE, F32, 1, variant == DEPTH, variant == ARRAY, variant == MSAA, 1, 0);
    EMIT(27, SAMPLED, IMAGE);
    EMIT(41, BOOL, TRUE_ID);
    EMIT(variant == SPEC_LOD ? 50 : 43, I32, ZERO_ID, 0);
    EMIT(43, I32, ONE_ID, 1);
    EMIT(43, F32, FLOAT_ZERO, 0);
    EMIT(44, I2, ICOORD, ZERO_ID, ZERO_ID);
    EMIT(44, F2, FCOORD, FLOAT_ZERO, FLOAT_ZERO);
    if (variant == FOREIGN_LITERAL) EMIT(43, I32, NOISE, TEX_A);
    if (variant == ARRAY_DESCRIPTOR) EMIT(28, NOISE, SAMPLED, ONE_ID);
    EMIT(32, POINTER, 0, variant == ARRAY_DESCRIPTOR ? NOISE : SAMPLED);
    EMIT(33, MAIN_TYPE, V);
    EMIT(33, HELPER_TYPE, V, POINTER);
    EMIT(59, POINTER, TEX_A, 0);
    EMIT(59, POINTER, TEX_B, 0);
    EMIT(54, V, MAIN, 0, MAIN_TYPE);
    EMIT(248, ENTRY_LABEL);
    EMIT(247, MERGE_LABEL, 0);
    EMIT(250, TRUE_ID, THEN_LABEL, ELSE_LABEL);
    EMIT(248, THEN_LABEL);
    EMIT(61, SAMPLED, A_LOAD_THEN, TEX_A);
    EMIT(100, IMAGE, A_IMAGE_THEN, A_LOAD_THEN);
    EMIT(103, I2, A_QUERY_THEN, A_IMAGE_THEN, ZERO_ID);
    EMIT(81, I32, EXTRACT_THEN, A_QUERY_THEN, 0);
    EMIT(61, SAMPLED, B_LOAD_THEN, TEX_B);
    EMIT(100, IMAGE, B_IMAGE_THEN, B_LOAD_THEN);
    EMIT(103, I2, B_QUERY_THEN, B_IMAGE_THEN, ZERO_ID);
    EMIT(249, MERGE_LABEL);
    EMIT(248, ELSE_LABEL);
    EMIT(61, SAMPLED, A_LOAD_ELSE, TEX_A);
    const uint32_t image_else = reconvergence ? 101 : A_IMAGE_ELSE;
    EMIT(100, IMAGE, image_else, A_LOAD_ELSE);
    if (variant == DYNAMIC_LOD) EMIT(128, I32, EXTRA, ZERO_ID, ZERO_ID);
    EMIT(103, variant == BAD_RESULT ? F2 : I2, A_QUERY_ELSE, image_else,
         variant == NONZERO_LOD ? ONE_ID : variant == DYNAMIC_LOD ? EXTRA : ZERO_ID);
    EMIT(81, I32, EXTRACT_ELSE, A_QUERY_ELSE, 1);
    EMIT(61, SAMPLED, B_LOAD_ELSE, TEX_B);
    EMIT(100, IMAGE, B_IMAGE_ELSE, B_LOAD_ELSE);
    EMIT(103, I2, B_QUERY_ELSE, B_IMAGE_ELSE, ZERO_ID);
    if (variant == FETCH || variant == RECONVERGENCE_IMAGE_ESCAPE)
        EMIT(95, F4, EXTRA, image_else, ICOORD, 2, ZERO_ID);
    if (variant == SAMPLE) EMIT(88, F4, EXTRA, A_LOAD_ELSE, FCOORD, 2, FLOAT_ZERO);
    if (variant == CALL) EMIT(57, V, EXTRA, HELPER, TEX_A);
    if (variant == COPY) EMIT(83, IMAGE, EXTRA, A_IMAGE_ELSE);
    EMIT(249, MERGE_LABEL);
    EMIT(248, MERGE_LABEL);
    BARE(253);
    BARE(56);
    EMIT(54, V, HELPER, 0, HELPER_TYPE);
    EMIT(55, POINTER, HELPER_ARG);
    EMIT(248, HELPER_LABEL);
    if (variant == HELPER_ESCAPE) {
        EMIT(61, SAMPLED, HELPER_LOAD, TEX_A);
        EMIT(100, IMAGE, HELPER_IMAGE, HELPER_LOAD);
        EMIT(95, F4, HELPER_FETCH, HELPER_IMAGE, ICOORD, 2, ZERO_ID);
    }
    BARE(253);
    BARE(56);
    if (variant == TRUNCATED) m.count--;
    return m;
}

static Module module(Variant variant, uint32_t set, uint32_t a_binding, uint32_t b_binding) {
    return stage_module(variant, set, a_binding, b_binding, RC_SAMPLED_IMAGE_QUERY_VERTEX);
}

static void check_site(const uint32_t *words, RcSampledImageQuerySite site, uint32_t var,
    uint32_t load, uint32_t image, uint32_t result, uint32_t result_type) {
    const uint32_t *l = words + site.load_word;
    const uint32_t *i = words + site.image_word;
    const uint32_t *q = words + site.query_word;
    assert(l[0] == (4u << 16 | 61u) && l[2] == load && l[3] == var);
    assert(i[0] == (4u << 16 | 100u) && i[2] == image && i[3] == load);
    assert(q[0] == (5u << 16 | 103u) && q[2] == result && q[3] == image);
    assert(site.result_type == result_type && q[1] == result_type);
}

static void fixture(void) {
    RcSampledImageQueryProof proof;
    RcSampledImageQuerySite sites[2];
    Module m = module(MATCHED, 0, 1, 2);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 1, &proof, NULL, 0) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN);
    assert(proof.variable_id == TEX_A && proof.site_count == 2);
    sites[0] = (RcSampledImageQuerySite){777, 888, 999, 555};
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 1, &proof, sites, 1) ==
           RC_SAMPLED_IMAGE_QUERY_CAPACITY);
    assert(proof.variable_id == TEX_A && proof.site_count == 2 && sites[0].query_word == 999);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 1, &proof, sites, 2) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN);
    check_site(m.words, sites[0], TEX_A, A_LOAD_THEN, A_IMAGE_THEN, A_QUERY_THEN, I2);
    check_site(m.words, sites[1], TEX_A, A_LOAD_ELSE, A_IMAGE_ELSE, A_QUERY_ELSE, I2);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 2, &proof, sites, 2) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN);
    check_site(m.words, sites[0], TEX_B, B_LOAD_THEN, B_IMAGE_THEN, B_QUERY_THEN, I2);
    check_site(m.words, sites[1], TEX_B, B_LOAD_ELSE, B_IMAGE_ELSE, B_QUERY_ELSE, I2);

    m = module(MATCHED, 3, 6, 7);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 3, 6, &proof, sites, 2) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 3, 7, &proof, sites, 2) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 1, &proof, sites, 2) ==
           RC_SAMPLED_IMAGE_QUERY_NONE);

    for (Variant v = FETCH; v <= TRUNCATED; ++v) {
        m = module(v, 0, 1, 2);
        sites[0].query_word = 999;
        assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 1, &proof, sites, 2) ==
               RC_SAMPLED_IMAGE_QUERY_NONE);
        assert(!proof.variable_id && !proof.site_count && sites[0].query_word == 999);
    }
    m = module(FETCH, 0, 1, 2);
    assert(rc_sampled_image_query_size_lod0(m.words, m.count, 0, 2, &proof, sites, 2) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN); // Mixed use in another descriptor does not taint this one.
    for (unsigned stage = RC_SAMPLED_IMAGE_QUERY_VERTEX; stage <= RC_SAMPLED_IMAGE_QUERY_COMPUTE; ++stage) {
        if (stage && stage < RC_SAMPLED_IMAGE_QUERY_FRAGMENT) continue;
        m = stage_module(RECONVERGENCE, 0, 1, 2, (RcSampledImageQueryStage)stage);
        const RcSampledImageQueryTarget target = {.stage = stage, .binding = 1};
        assert(rc_sampled_image_query_size_lod0_for(m.words, m.count, target, &proof, sites, 2) ==
               RC_SAMPLED_IMAGE_QUERY_PROVEN);
        check_site(m.words, sites[1], TEX_A, A_LOAD_ELSE, 101, A_QUERY_ELSE, I2);
    }
    puts("sampled image query proof: both sites and bindings, reconvergence metadata, and refusal controls");
}

enum {
    TABLE_VAR = 28, // Exercise a table variable ID equal to the GlobalInvocationId literal.
    TABLE_TYPE = BOUND, TABLE_PTR, ELEM_PTR, TABLE_LENGTH, MEMBER_7, MEMBER_8, ELEM_7, ELEM_8,
    LOAD_7, LOAD_8, QUERY_7, QUERY_8, TABLE_VALUE, OUT_PTR, OUT_VAR, IN_PTR, IN_VAR, INPUT_VALUE,
    SAMPLER_TYPE, SAMPLER_PTR, SAMPLER_VAR, SAMPLER_LOAD, COMBINED, EXTRACTED, U32, U3, TABLE_BOUND,
};

typedef enum {
    TABLE_LITERAL, TABLE_WRAP_IMAGE, TABLE_SIBLING_FETCH, TABLE_SELECTED_FETCH, TABLE_DYNAMIC,
    TABLE_DYNAMIC_FETCH, TABLE_RUNTIME_DYNAMIC, TABLE_RANGE, TABLE_INDEX_SPEC, TABLE_POINTER_ESCAPE,
    TABLE_WRONG_DIM, TABLE_WRONG_ELEMENT, TABLE_WRONG_CHAIN, TABLE_ARRAY_IMAGE, TABLE_INTEGER, TABLE_COMPUTE,
    TABLE_RECONVERGENCE, TABLE_RECONVERGENCE_FETCH,
} TableVariant;

static Module table_module(TableVariant variant) {
    const int reconvergence = variant == TABLE_RECONVERGENCE || variant == TABLE_RECONVERGENCE_FETCH;
    const int dynamic = variant == TABLE_DYNAMIC || variant == TABLE_DYNAMIC_FETCH || variant == TABLE_RUNTIME_DYNAMIC ||
        reconvergence;
    const int runtime = variant == TABLE_RUNTIME_DYNAMIC;
    const int compute = variant == TABLE_COMPUTE;
    Module m = {{0x07230203, 0x00010300, 0, TABLE_BOUND, 0}, 5};
    EMIT(17, 1);  // Shader
    EMIT(17, 50); // ImageQuery
    if (dynamic) {
        EMIT(17, 5301); // ShaderNonUniform
        if (runtime) EMIT(17, 5302); // RuntimeDescriptorArray
        EMIT(17, 5307); // SampledImageArrayNonUniformIndexing
        EMIT(10, 0x5f565053, 0x5f545845, 0x63736564, 0x74706972, 0x695f726f, 0x7865646e, 0x00676e69);
    }
    if (reconvergence) add_reconvergence_extension(&m, 0);
    EMIT(14, 0, 1); // Logical GLSL450
    if (compute) {
        EMIT(15, RC_SAMPLED_IMAGE_QUERY_COMPUTE, MAIN, 0x6e69616d, 0, IN_VAR);
        EMIT(16, MAIN, 17, 16, 1, 1); // LocalSize
        EMIT(71, IN_VAR, 11, 28); // BuiltIn GlobalInvocationId
    } else {
        if (dynamic) EMIT(15, RC_SAMPLED_IMAGE_QUERY_FRAGMENT, MAIN, 0x6e69616d, 0, OUT_VAR, IN_VAR);
        else EMIT(15, RC_SAMPLED_IMAGE_QUERY_FRAGMENT, MAIN, 0x6e69616d, 0, OUT_VAR);
        EMIT(16, MAIN, 7); // OriginUpperLeft
        if (reconvergence) EMIT(16, MAIN, 6023);
        EMIT(71, OUT_VAR, 30, 0); // Location 0
    }
    EMIT(71, TABLE_VAR, 34, 2);
    EMIT(71, TABLE_VAR, 33, 2);
    if (dynamic) {
        EMIT(71, IN_VAR, 30, 0);
        EMIT(71, IN_VAR, 14); // Flat
        EMIT(71, INPUT_VALUE, 5300); // NonUniform
        EMIT(71, ELEM_7, 5300);
        EMIT(71, LOAD_7, 5300);
    }
    if (variant == TABLE_WRAP_IMAGE) {
        EMIT(71, SAMPLER_VAR, 34, 2);
        EMIT(71, SAMPLER_VAR, 33, 16);
    }
    EMIT(19, V);
    EMIT(21, I32, 32, 1);
    EMIT(22, F32, 32);
    EMIT(23, I2, I32, 2);
    EMIT(23, F4, F32, 4);
    EMIT(25, IMAGE, variant == TABLE_INTEGER ? I32 : F32, variant == TABLE_WRONG_DIM ? 0 : 1,
         0, variant == TABLE_ARRAY_IMAGE, 0, 1, 0);
    EMIT(43, I32, ZERO_ID, 0);
    EMIT(variant == TABLE_INDEX_SPEC ? 50 : 43, I32, MEMBER_7, variant == TABLE_RANGE ? 32768 : 7);
    EMIT(43, I32, MEMBER_8, 8);
    EMIT(43, I32, TABLE_LENGTH, 32768);
    EMIT(44, I2, ICOORD, ZERO_ID, ZERO_ID);
    if (runtime) EMIT(29, TABLE_TYPE, IMAGE);
    else EMIT(28, TABLE_TYPE, IMAGE, TABLE_LENGTH);
    EMIT(32, TABLE_PTR, 0, TABLE_TYPE);
    EMIT(32, ELEM_PTR, 0, variant == TABLE_WRONG_ELEMENT ? F32 : IMAGE);
    if (compute) {
        EMIT(21, U32, 32, 0);
        EMIT(23, U3, U32, 3);
        EMIT(32, IN_PTR, 1, U3);
    } else {
        EMIT(32, OUT_PTR, 3, I2);
        if (dynamic) EMIT(32, IN_PTR, 1, I32);
    }
    if (variant == TABLE_WRAP_IMAGE) {
        EMIT(26, SAMPLER_TYPE);
        EMIT(27, SAMPLED, IMAGE);
        EMIT(32, SAMPLER_PTR, 0, SAMPLER_TYPE);
        EMIT(59, SAMPLER_PTR, SAMPLER_VAR, 0);
    }
    EMIT(33, MAIN_TYPE, V);
    EMIT(59, TABLE_PTR, TABLE_VAR, 0);
    if (!compute) EMIT(59, OUT_PTR, OUT_VAR, 3);
    if (dynamic || compute) EMIT(59, IN_PTR, IN_VAR, 1);
    EMIT(54, V, MAIN, 0, MAIN_TYPE);
    EMIT(248, ENTRY_LABEL);
    if (dynamic) EMIT(61, I32, INPUT_VALUE, IN_VAR);
    if (variant == TABLE_WRONG_CHAIN) EMIT(65, ELEM_PTR, ELEM_7, TABLE_VAR, MEMBER_7, ZERO_ID);
    else EMIT(65, ELEM_PTR, ELEM_7, TABLE_VAR, dynamic ? INPUT_VALUE : MEMBER_7);
    EMIT(61, IMAGE, LOAD_7, ELEM_7);
    if (variant == TABLE_WRAP_IMAGE) {
        EMIT(61, SAMPLER_TYPE, SAMPLER_LOAD, SAMPLER_VAR);
        EMIT(86, SAMPLED, COMBINED, LOAD_7, SAMPLER_LOAD);
        EMIT(100, IMAGE, EXTRACTED, COMBINED);
    }
    EMIT(103, I2, QUERY_7, variant == TABLE_WRAP_IMAGE ? EXTRACTED : LOAD_7, ZERO_ID);
    if (variant == TABLE_SELECTED_FETCH || variant == TABLE_DYNAMIC_FETCH || variant == TABLE_RECONVERGENCE_FETCH)
        EMIT(95, F4, TABLE_VALUE, LOAD_7, ICOORD, 2, ZERO_ID);
    if (variant == TABLE_SIBLING_FETCH || variant == TABLE_POINTER_ESCAPE) {
        EMIT(66, ELEM_PTR, ELEM_8, TABLE_VAR, MEMBER_8);
        if (variant == TABLE_POINTER_ESCAPE) EMIT(83, ELEM_PTR, EXTRA, ELEM_8);
        EMIT(61, IMAGE, LOAD_8, ELEM_8);
        EMIT(95, F4, TABLE_VALUE, LOAD_8, ICOORD, 2, ZERO_ID);
    }
    if (!compute) EMIT(62, OUT_VAR, QUERY_7);
    BARE(253);
    BARE(56);
    return m;
}

static void check_table_site(const Module *m, RcSampledImageQuerySite site, uint32_t image_op) {
    const uint32_t *load = m->words + site.load_word;
    const uint32_t *image = m->words + site.image_word;
    const uint32_t *query = m->words + site.query_word;
    assert((load[0] & 65535) == 61 && load[2] == LOAD_7 && load[3] == ELEM_7);
    assert((image[0] & 65535) == image_op);
    assert((query[0] & 65535) == 103 && query[3] == image[2] && query[4] == ZERO_ID);
    assert(site.result_type == I2);
}

static void generalized_fixture(void) {
    RcSampledImageQueryProof proof;
    RcSampledImageQuerySite site;
    const RcSampledImageQueryStage stages[] = {RC_SAMPLED_IMAGE_QUERY_FRAGMENT, RC_SAMPLED_IMAGE_QUERY_COMPUTE};
    for (unsigned i = 0; i < sizeof stages / sizeof stages[0]; ++i) {
        RcSampledImageQueryTarget target = {.stage = stages[i], .set = 1, .binding = 6};
        RcSampledImageQuerySite sites[2];
        Module m = stage_module(MATCHED, 1, 6, 7, target.stage);
        assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,sites,2) ==
               RC_SAMPLED_IMAGE_QUERY_PROVEN && proof.variable_id == TEX_A && proof.site_count == 2);
        check_site(m.words, sites[0], TEX_A, A_LOAD_THEN, A_IMAGE_THEN, A_QUERY_THEN, I2);
        assert(rc_sampled_image_query_size_lod0(m.words,m.count,1,6,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
        m = stage_module(SAMPLE, 1, 6, 7, target.stage);
        assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
        target.binding = 7;
        assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_PROVEN);
        target.stage = stages[1-i];
        assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    }

    RcSampledImageQueryTarget target = {
        .stage = RC_SAMPLED_IMAGE_QUERY_FRAGMENT, .set = 2, .binding = 2, .table_count = 32768, .member = 7,
    };
    Module m = table_module(TABLE_LITERAL);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,&site,1) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN && proof.variable_id == TABLE_VAR && proof.site_count == 1);
    assert(site.image_word == site.load_word);
    check_table_site(&m,site,61);
    m = table_module(TABLE_WRAP_IMAGE);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,&site,1) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN && proof.site_count == 1);
    assert(site.image_word != site.load_word);
    check_table_site(&m,site,100);

    m = table_module(TABLE_SIBLING_FETCH);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_PROVEN);
    target.member = 8;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    target.member = RC_SAMPLED_IMAGE_QUERY_ALL_MEMBERS;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    m = table_module(TABLE_RECONVERGENCE);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,&site,1) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN && proof.site_count == 1);
    check_table_site(&m,site,61);
    m = table_module(TABLE_RECONVERGENCE_FETCH);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    m = table_module(TABLE_DYNAMIC);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,&site,1) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN && proof.site_count == 1);
    check_table_site(&m,site,61);
    target.member = 7;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    m = table_module(TABLE_RUNTIME_DYNAMIC);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    target.member = RC_SAMPLED_IMAGE_QUERY_ALL_MEMBERS;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_PROVEN);
    m = table_module(TABLE_DYNAMIC_FETCH);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    m = table_module(TABLE_RANGE);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);

    const TableVariant refused[] = {TABLE_SELECTED_FETCH, TABLE_RANGE, TABLE_INDEX_SPEC, TABLE_POINTER_ESCAPE,
        TABLE_WRONG_DIM, TABLE_WRONG_ELEMENT, TABLE_WRONG_CHAIN, TABLE_ARRAY_IMAGE, TABLE_INTEGER};
    target.member = 7;
    for (unsigned i = 0; i < sizeof refused / sizeof refused[0]; ++i) {
        m = table_module(refused[i]);
        site.query_word = 999;
        assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,&site,1) == RC_SAMPLED_IMAGE_QUERY_NONE);
        assert(!proof.variable_id && !proof.site_count && site.query_word == 999);
    }
    m = table_module(TABLE_LITERAL);
    target.table_count--;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    target.table_count++;
    target.set = 1;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    target.set = 2;
    target.binding = 3;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    target.binding = 2;
    target.member = 32768;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    target.member = 7;
    target.stage = RC_SAMPLED_IMAGE_QUERY_COMPUTE;
    m = table_module(TABLE_COMPUTE);
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,&site,1) ==
           RC_SAMPLED_IMAGE_QUERY_PROVEN && proof.site_count == 1);
    target.stage = (RcSampledImageQueryStage)1;
    assert(rc_sampled_image_query_size_lod0_for(m.words,m.count,target,&proof,NULL,0) == RC_SAMPLED_IMAGE_QUERY_NONE);
    puts("sampled image shape: fragment/compute, exact pool members, dynamic binding-wide scope and conservative refusals");
}

static int inspect(const char *path, uint32_t set, uint32_t binding) {
    FILE *file = fopen(path, "rb");
    if (!file) return 1;
    int status = 1;
    uint32_t *words = NULL;
    RcSampledImageQuerySite *sites = NULL;
    if (fseek(file, 0, SEEK_END)) goto done;
    const long bytes = ftell(file);
    if (bytes < 20 || bytes % 4 || fseek(file, 0, SEEK_SET)) goto done;
    words = malloc((size_t)bytes);
    if (!words) goto done;
    const size_t count = (size_t)bytes / 4;
    for (size_t i = 0; i < count; ++i) {
        unsigned char b[4];
        if (fread(b, 1, sizeof b, file) != sizeof b) goto done;
        words[i] = b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    }
    RcSampledImageQueryProof proof;
    if (rc_sampled_image_query_size_lod0(words, count, set, binding, &proof, NULL, 0) !=
        RC_SAMPLED_IMAGE_QUERY_PROVEN) goto done;
    sites = calloc(proof.site_count, sizeof *sites);
    if (!sites || rc_sampled_image_query_size_lod0(words, count, set, binding, &proof, sites, proof.site_count) !=
        RC_SAMPLED_IMAGE_QUERY_PROVEN) goto done;
    printf("set=%u binding=%u variable=%%%u sites=%zu\n", set, binding, proof.variable_id, proof.site_count);
    for (size_t i = 0; i < proof.site_count; ++i) {
        const RcSampledImageQuerySite s = sites[i];
        printf("  load=%zu (%%%u) image=%zu (%%%u) query=%zu (%%%u, type %%%u, lod %%%u)\n",
            s.load_word, words[s.load_word + 2], s.image_word, words[s.image_word + 2], s.query_word,
            words[s.query_word + 2], s.result_type, words[s.query_word + 4]);
    }
    status = 0;
done:
    free(sites);
    free(words);
    fclose(file);
    return status;
}

int main(int argc, char **argv) {
    if (argc == 4) return inspect(argv[1], (uint32_t)strtoul(argv[2], NULL, 0), (uint32_t)strtoul(argv[3], NULL, 0));
    if (argc != 1) return 1;
    fixture();
    generalized_fixture();
    return 0;
}
