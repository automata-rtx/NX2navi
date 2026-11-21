#include "rc_native_rmw_word.h"
#include "resource_core_inventory.h"

#include <stdlib.h>

enum {
    SpvEntryPoint = 15, SpvTypeVoid = 19, SpvTypeBool = 20, SpvTypeInt = 21, SpvTypeFloat = 22,
    SpvTypeVector = 23, SpvTypeArray = 28,
    SpvTypeRuntimeArray = 29, SpvTypeStruct = 30, SpvTypePointer = 32, SpvTypeForwardPointer = 39,
    SpvConstant = 43,
    SpvFunction = 54, SpvFunctionEnd = 56, SpvVariable = 59, SpvLoad = 61, SpvAccessChain = 65,
    SpvArrayLength = 68,
    SpvDecorate = 71, SpvMemberDecorate = 72, SpvDecorationGroup = 73, SpvGroupMemberDecorate = 75,
    SpvBitcast = 124, SpvIAdd = 128, SpvISub = 130,
    SpvSelect = 169, SpvULessThan = 176, SpvLogicalAnd = 167,
    SpvShiftRightLogical = 194, SpvShiftLeftLogical = 196, SpvBitwiseAnd = 199,
    SpvAtomicIAdd = 234, SpvAtomicUMin = 237, SpvAtomicOr = 241,
    SpvUniform = 2, SpvStorageBuffer = 12,
    SpvBlock = 2, SpvBufferBlock = 3, SpvArrayStride = 6, SpvBinding = 33, SpvDescriptorSet = 34, SpvOffset = 35,
    SpvFragment = 4, SpvMeshEXT = 5365,
};

typedef struct {
    const uint32_t *words;
    const uint32_t **definitions;
    size_t count;
    uint32_t bound;
} NativeWordModule;

static uint32_t opcode(const uint32_t *instruction) {
    return *instruction & UINT16_MAX;
}

static uint32_t length(const uint32_t *instruction) {
    return *instruction >> 16;
}

static unsigned result_slot(uint32_t op) {
    switch (op) {
    case SpvTypeInt: case SpvTypeFloat: case SpvTypeVector: case SpvTypeArray:
    case SpvTypeRuntimeArray: case SpvTypeStruct: case SpvTypePointer:
        return 1;
    case SpvConstant: case SpvVariable: case SpvLoad: case SpvAccessChain: case SpvBitcast:
    case SpvIAdd: case SpvISub: case SpvShiftRightLogical: case SpvBitwiseAnd:
        return 2;
    default:
        return 0;
    }
}

static const uint32_t *definition(const NativeWordModule *module, uint32_t id, uint32_t op, uint32_t len) {
    const uint32_t *instruction = id < module->bound ? module->definitions[id] : NULL;
    return instruction && opcode(instruction) == op && length(instruction) == len ? instruction : NULL;
}

static int int32(const NativeWordModule *module, uint32_t id) {
    const uint32_t *type = definition(module, id, SpvTypeInt, 4);
    return type && type[2] == 32;
}

static int constant32(const NativeWordModule *module, uint32_t id, uint32_t *value) {
    const uint32_t *instruction = definition(module, id, SpvConstant, 4);
    if (!instruction || !int32(module, instruction[1])) return 0;
    *value = instruction[3];
    return 1;
}

static uint32_t pointee(const NativeWordModule *module, uint32_t type, uint32_t storage) {
    const uint32_t *ptr = definition(module, type, SpvTypePointer, 4);
    return ptr && ptr[2] == storage ? ptr[3] : 0;
}

static uint32_t object_type(const NativeWordModule *module, uint32_t id, uint32_t storage) {
    const uint32_t *var = definition(module, id, SpvVariable, 4);
    return var && var[3] == storage ? pointee(module, var[1], storage) : 0;
}

static int decoration(const NativeWordModule *module, uint32_t id, uint32_t member, uint32_t kind, uint32_t *value) {
    unsigned hits = 0;
    const int is_member = member != UINT32_MAX;
    const uint32_t op = is_member ? SpvMemberDecorate : SpvDecorate;
    const unsigned tag = is_member ? 3 : 2;
    for (size_t at = 5; at < module->count; at += length(module->words + at)) {
        const uint32_t *p = module->words + at;
        if (opcode(p) != op || length(p) <= tag || p[1] != id || p[tag] != kind ||
            (is_member && p[2] != member)) continue;
        if (length(p) != tag + 1 + (value != NULL) || ++hits != 1) return 0;
        if (value) *value = p[tag + 1];
    }
    return hits == 1;
}

static int block_layout(const NativeWordModule *module, uint32_t structure, uint32_t kind) {
    uint32_t base;
    return decoration(module, structure, UINT32_MAX, kind, NULL) &&
           decoration(module, structure, 0, SpvOffset, &base) && base == 0;
}

static int descriptor(const NativeWordModule *module, uint32_t id, uint32_t *set, uint32_t *binding) {
    if (!decoration(module, id, UINT32_MAX, SpvDescriptorSet, set) ||
        !decoration(module, id, UINT32_MAX, SpvBinding, binding)) return 0;
    for (size_t at = 5; at < module->count; at += length(module->words + at)) {
        const uint32_t *p = module->words + at;
        uint32_t other_set;
        if (opcode(p) == SpvDecorate && length(p) == 4 && p[1] != id &&
            p[2] == SpvBinding && p[3] == *binding &&
            decoration(module, p[1], UINT32_MAX, SpvDescriptorSet, &other_set) && other_set == *set) return 0;
    }
    return 1;
}

static int c0_source(const NativeWordModule *module, uint32_t id, uint32_t *variable, uint32_t *byte) {
    const uint32_t *cast = definition(module, id, SpvBitcast, 4);
    const uint32_t *load = cast ? definition(module, cast[3], SpvLoad, 4) : NULL;
    const uint32_t *source = load ? definition(module, load[3], SpvAccessChain, 7) : NULL;
    uint32_t member, row, component;
    if (!cast || !int32(module, cast[1]) || !source ||
        pointee(module, source[1], SpvUniform) != load[1] ||
        !constant32(module, source[4], &member) || member != 0 ||
        !constant32(module, source[5], &row) || !constant32(module, source[6], &component) ||
        component >= 4 || row > (UINT32_MAX - 12) / 16) return 0;
    const uint32_t var_struct = object_type(module, source[3], SpvUniform);
    const uint32_t *structure = definition(module, var_struct, SpvTypeStruct, 3);
    const uint32_t *array = structure ? definition(module, structure[2], SpvTypeArray, 4) : NULL;
    const uint32_t *vector = array ? definition(module, array[2], SpvTypeVector, 4) : NULL;
    const uint32_t *scalar = vector ? definition(module, vector[2], SpvTypeFloat, 3) : NULL;
    uint32_t stride, rows;
    if (!scalar || scalar[2] != 32 || vector[2] != load[1] || vector[3] != 4 ||
        !constant32(module, array[3], &rows) || row >= rows || !block_layout(module, var_struct, SpvBlock) ||
        !decoration(module, structure[2], UINT32_MAX, SpvArrayStride, &stride) || stride != 16) return 0;
    *variable = source[3];
    *byte = row * 16 + component * 4;
    return 1;
}

static int address_expression(const NativeWordModule *module, uint32_t id, RcNativeRmwWord *word) {
    const uint32_t *shift = definition(module, id, SpvShiftRightLogical, 5);
    const uint32_t *delta = shift ? definition(module, shift[3], SpvISub, 5) : NULL;
    const uint32_t *sum = delta ? definition(module, delta[3], SpvIAdd, 5) : NULL;
    const uint32_t *align = delta ? definition(module, delta[4], SpvBitwiseAnd, 5) : NULL;
    uint32_t amount, var, other, other_byte;
    if (!sum || !align || !int32(module, shift[1]) || !int32(module, delta[1]) ||
        !int32(module, sum[1]) || !int32(module, align[1]) ||
        !constant32(module, shift[4], &amount) || amount != 2 ||
        !constant32(module, sum[4], &word->add) ||
        !constant32(module, align[4], &word->align_mask) ||
        !c0_source(module, sum[3], &var, &word->c0_byte) ||
        !c0_source(module, align[3], &other, &other_byte) || var != other || word->c0_byte != other_byte)
        return 0;
    const uint32_t low_bits = ~word->align_mask;
    return word->align_mask && !(low_bits & (low_bits + 1)) &&
        word->add <= UINT32_MAX - low_bits && descriptor(module, var, &word->c0_set, &word->c0_binding);
}

static int storage_layout(const NativeWordModule *module, const uint32_t *chain, uint32_t atomic_type,
                          uint32_t storage) {
    const uint32_t var_struct = object_type(module, chain[3], storage);
    const uint32_t *structure = definition(module, var_struct, SpvTypeStruct, 3);
    const uint32_t *array = structure ? definition(module, structure[2], SpvTypeRuntimeArray, 3) : NULL;
    uint32_t member, stride;
    return array && int32(module, atomic_type) && array[2] == atomic_type &&
        pointee(module, chain[1], storage) == atomic_type &&
        constant32(module, chain[4], &member) && member == 0 &&
        block_layout(module, var_struct, storage == SpvStorageBuffer ? SpvBlock : SpvBufferBlock) &&
        decoration(module, structure[2], UINT32_MAX, SpvArrayStride, &stride) && stride == 4;
}

// Conservative raw-operand scan confined to function bodies. A coincident literal can cause refusal.
static int only_storage_access(const NativeWordModule *module, const uint32_t *chain, const uint32_t *atomic) {
    int function = 0;
    for (size_t i = 5; i < module->count; i += length(module->words + i)) {
        const uint32_t *p = module->words + i;
        if (opcode(p) == SpvFunction) function = 1;
        if (opcode(p) == SpvFunctionEnd) function = 0;
        if (!function) continue;
        for (uint32_t k = 1; k < length(p); ++k) {
            if (p[k] == chain[3] && (p != chain || k != 3)) return 0;
            if (p[k] == chain[2] && (p != chain || k != 2) && (p != atomic || k != 3)) return 0;
        }
    }
    return 1;
}

static int matches_output(const RcNativeRmwWord *word, const RcWriteBinding *binding) {
    return binding->kind == RC_INVENTORY_BUFFER && binding->count == 1 &&
        binding->set == word->storage_set && binding->binding == word->storage_binding;
}

static int only_output(void *context, const RcWriteBinding *binding) {
    return matches_output(context, binding);
}

static int is_graphics_model(uint32_t model) {
    return model <= 4 || model == 5267 || model == 5268 || model == 5364 || model == 5365;
}

static int native_module_parse(NativeWordModule *module, uint32_t operation, RcNativeRmwWord *word) {
    int function = 0;
    for (size_t i = 5; i < module->count;) {
        const uint32_t *p = module->words + i;
        const uint32_t op = opcode(p), len = length(p), slot = result_slot(op);
        if (!len || len > module->count - i || slot >= len) return 0;
        if (slot) {
            uint32_t id = p[slot];
            if (!id || id >= module->bound || module->definitions[id]) return 0;
            module->definitions[id] = p;
        }
        if (op >= SpvDecorationGroup && op <= SpvGroupMemberDecorate) return 0;
        if (op == SpvFunction) {
            if (function || len != 5) return 0;
            function = 1;
        } else if (op == SpvFunctionEnd) {
            if (!function || len != 1) return 0;
            function = 0;
        } else if (op == operation && !function) return 0;
        if (op == SpvEntryPoint) {
            if (len < 4 || word->entry || !p[2] || p[2] >= module->bound || !is_graphics_model(p[1])) return 0;
            word->execution_model = p[1];
            word->entry = p[2];
        }
        i += len;
    }
    return !function && word->entry;
}

static int native_rmw_match(const uint32_t *words, size_t count, RcNativeRmwWord *out, int fragment) {
    if (!out) return 0;
    *out = (RcNativeRmwWord){0};
    if (!words || count < 5 || words[0] != 0x07230203 || !words[3] || words[3] > (1u << 20)) return 0;
    NativeWordModule module = {.words = words, .count = count, .bound = words[3]};
    module.definitions = calloc(module.bound, sizeof *module.definitions);
    if (!module.definitions) return 0;
    RcNativeRmwWord word = {0};
    int result = 0;
    const uint32_t *atomic = NULL;
    const uint32_t operation = fragment ? SpvAtomicUMin : SpvAtomicIAdd;
    if (!native_module_parse(&module, operation, &word)) goto done;
    for (size_t i = 5; i < count; i += length(words + i)) {
        const uint32_t *p = words + i;
        if (opcode(p) != operation) continue;
        if (length(p) != 7) goto done;
        const uint32_t *chain = definition(&module, p[3], SpvAccessChain, 6);
        if (!chain) continue;
        uint32_t storage = SpvStorageBuffer;
        if (fragment && !object_type(&module, chain[3], storage)) storage = SpvUniform;
        if (!object_type(&module, chain[3], storage)) continue;
        if (atomic || !storage_layout(&module, chain, p[1], storage) ||
            !descriptor(&module, chain[3], &word.storage_set, &word.storage_binding) ||
            !only_storage_access(&module, chain, p)) goto done;
        if (fragment) {
            const uint32_t *index = chain[5] < module.bound ? module.definitions[chain[5]] : NULL;
            const uint32_t *type = definition(&module, p[1], SpvTypeInt, 4);
            uint32_t scope, semantics;
            if (!index || result_slot(opcode(index)) != 2 || !int32(&module, index[1]) ||
                !type || type[3] || !constant32(&module, p[4], &scope) || scope != 1 ||
                !constant32(&module, p[5], &semantics) || semantics != 0)
                goto done;
            word.index = chain[5];
            word.storage_class = storage;
        } else if (!address_expression(&module, chain[5], &word)) goto done;
        atomic = p;
        word.atomic_word = i;
    }
    if (!word.entry || !atomic || (fragment && word.execution_model != SpvFragment) ||
        !rc_write_inventory_covered(words, count, only_output, &word)) goto done;
    word.bitmap_sites = fragment;
    *out = word;
    result = 1;
done:
    free(module.definitions);
    return result;
}

int rc_native_rmw_word(const uint32_t *words, size_t count, RcNativeRmwWord *out) {
    return native_rmw_match(words, count, out, 0);
}

int rc_native_rmw_fragment(const uint32_t *words, size_t count, RcNativeRmwWord *out) {
    return native_rmw_match(words, count, out, 1);
}

enum { BitmapSiteLimit = 16 };
typedef struct { size_t instruction; uint32_t index; } NativeBitmapSite;
typedef struct {
    RcNativeRmwWord receipt;
    NativeBitmapSite sites[BitmapSiteLimit];
} NativeBitmapPlan;

static int mesh_bitmap_write_covered(void *opaque, size_t instruction, const RcWriteBinding *binding) {
    const NativeBitmapPlan *plan = opaque;
    if (!matches_output(&plan->receipt, binding)) return 0;
    for (uint32_t i = 0; i < plan->receipt.bitmap_sites; ++i)
        if (plan->sites[i].instruction == instruction) return 1;
    return 0;
}

static int native_mesh_bitmap_plan(const uint32_t *words, size_t count, NativeBitmapPlan *plan) {
    if (!words || count < 5 || words[0] != 0x07230203 || !words[3] || words[3] > (1u << 20)) return 0;
    NativeWordModule module = {.words = words, .count = count, .bound = words[3]};
    module.definitions = calloc(module.bound, sizeof *module.definitions);
    if (!module.definitions) return 0;
    RcNativeRmwWord *receipt = &plan->receipt;
    int result = 0;
    uint32_t base = 0, atomic_type = 0, scalar_pointer = 0;
    if (!native_module_parse(&module, SpvAtomicIAdd, receipt) || receipt->execution_model != SpvMeshEXT) goto done;
    for (size_t i = 5; i < count; i += length(words + i)) {
        const uint32_t *p = words + i;
        if (opcode(p) != SpvAtomicIAdd || length(p) != 7) continue;
        const uint32_t *chain = definition(&module, p[3], SpvAccessChain, 6);
        if (!chain) continue;
        const uint32_t storage = object_type(&module, chain[3], SpvStorageBuffer) ?
            SpvStorageBuffer : SpvUniform;
        if (!object_type(&module, chain[3], storage)) continue;
        uint32_t set, binding, scope, semantics;
        const uint32_t *index = chain[5] < module.bound ? module.definitions[chain[5]] : NULL;
        const uint32_t *type = definition(&module, p[1], SpvTypeInt, 4);
        if (receipt->bitmap_sites == BitmapSiteLimit || !storage_layout(&module, chain, p[1], storage) ||
            !descriptor(&module, chain[3], &set, &binding) ||
            !index || result_slot(opcode(index)) != 2 || !int32(&module, index[1]) || !type || type[3] ||
            !constant32(&module, p[4], &scope) || scope != 1 ||
            !constant32(&module, p[5], &semantics) || semantics != 0) goto done;
        if (!base) {
            base = chain[3];
            atomic_type = p[1];
            scalar_pointer = chain[1];
            receipt->storage_set = set;
            receipt->storage_binding = binding;
            receipt->storage_class = storage;
            receipt->atomic_word = i;
            receipt->index = chain[5];
        } else if (chain[3] != base || p[1] != atomic_type || chain[1] != scalar_pointer) goto done;
        plan->sites[receipt->bitmap_sites++] = (NativeBitmapSite){i, chain[5]};
    }
    result = receipt->bitmap_sites &&
        rc_write_inventory_sites_covered(words, count, mesh_bitmap_write_covered, plan);
done:
    free(module.definitions);
    return result;
}

int rc_native_rmw_mesh_bitmap(const uint32_t *words, size_t count, RcNativeRmwWord *out) {
    if (!out) return 0;
    NativeBitmapPlan plan = {0};
    const int valid = native_mesh_bitmap_plan(words, count, &plan);
    *out = valid ? plan.receipt : (RcNativeRmwWord){0};
    return valid;
}

int rc_native_rmw_word_address(const RcNativeRmwWord *plan, uint32_t c0_value, uint64_t descriptor_address,
                               uint64_t descriptor_size, uint64_t *word) {
    if (!word) return 0;
    *word = 0;
    if (!plan || !plan->atomic_word || !descriptor_address || (descriptor_address & 3) ||
        descriptor_size < 4 || descriptor_size > UINT64_MAX - descriptor_address) return 0;
    const uint32_t relative = ((c0_value + plan->add) - (c0_value & plan->align_mask)) & ~(uint32_t)3;
    if (relative > descriptor_size - 4) return 0;
    *word = descriptor_address + relative;
    return 1;
}

static void marker_emit(uint32_t **out, const uint32_t *words, size_t count) {
    for (size_t i = 0; i < count; ++i) *(*out)++ = words[i];
}

uint32_t *rc_native_rmw_word_mark(const uint32_t *words, size_t count, uint32_t binding,
                                  RcNativeRmwWord *original, size_t *marked_count) {
    enum { MarkerWordLimit = 56 };
    if (original) *original = (RcNativeRmwWord){0};
    if (marked_count) *marked_count = 0;
    if (!binding || !original || !marked_count || count > SIZE_MAX / sizeof(uint32_t) - MarkerWordLimit)
        return NULL;
    RcNativeRmwWord proof;
    if (!rc_native_rmw_word(words, count, &proof) || proof.execution_model != SpvMeshEXT || proof.storage_set != 0 ||
        words[1] < 0x00010400)
        return NULL;
    const uint32_t *atomic = words + proof.atomic_word;
    const uint32_t uint_type = atomic[1];
    uint32_t scalar_pointer = 0;
    size_t annotations_end = 0, functions_begin = 0, entry = 0;
    for (size_t i = 5; i < proof.atomic_word; i += length(words + i)) {
        const uint32_t *p = words + i;
        const uint32_t op = opcode(p);
        if (op == SpvEntryPoint) {
            if (length(p) == UINT16_MAX || p[2] != proof.entry) return NULL;
            entry = i;
        } else if (op == SpvDecorate && length(p) == 4 && p[2] == SpvBinding && p[3] == binding) {
            return NULL;
        } else if (op == SpvTypePointer && length(p) == 4 && p[2] == SpvStorageBuffer && p[3] == uint_type) {
            scalar_pointer = p[1];
        }
        if (!annotations_end && op >= SpvTypeVoid && op <= SpvTypeForwardPointer) annotations_end = i;
        if (op == SpvFunction) {
            functions_begin = i;
            break;
        }
    }
    if (!scalar_pointer || !entry || !annotations_end || !functions_begin || proof.atomic_word < functions_begin)
        return NULL;
    const uint32_t entry_size = length(words + entry);
    const size_t after_entry = entry + entry_size;
    if (after_entry > annotations_end || annotations_end > functions_begin) return NULL;

    uint32_t next = words[3];
    const uint32_t block = next++, block_pointer = next++, marker = next++, zero = next++, one = next++;
    const uint32_t marker_ptr = next++, marker_result = next++;
    uint32_t *module = malloc((count + MarkerWordLimit) * sizeof *module);
    if (!module) return NULL;
    uint32_t *out = module;
    marker_emit(&out, words, entry);
    *out++ = ((entry_size + 1) << 16) | SpvEntryPoint;
    marker_emit(&out, words + entry + 1, entry_size - 1);
    *out++ = marker; // Vulkan SPIR-V 1.4+ lists each statically used global in the entry-point interface.
    marker_emit(&out, words + after_entry, annotations_end - after_entry);
    const uint32_t decorations[] = {
        (3u << 16) | SpvDecorate, block, SpvBlock,
        (5u << 16) | SpvMemberDecorate, block, 0, SpvOffset, 0,
        (4u << 16) | SpvDecorate, marker, SpvDescriptorSet, 0,
        (4u << 16) | SpvDecorate, marker, SpvBinding, binding,
    };
    marker_emit(&out, decorations, sizeof decorations / sizeof *decorations);
    marker_emit(&out, words + annotations_end, functions_begin - annotations_end);
    const uint32_t declarations[] = {
        (3u << 16) | SpvTypeStruct, block, uint_type,
        (4u << 16) | SpvTypePointer, block_pointer, SpvStorageBuffer, block,
        (4u << 16) | SpvConstant, uint_type, zero, 0,
        (4u << 16) | SpvConstant, uint_type, one, 1,
        (4u << 16) | SpvVariable, block_pointer, marker, SpvStorageBuffer,
    };
    marker_emit(&out, declarations, sizeof declarations / sizeof *declarations);
    const size_t after_atomic = proof.atomic_word + length(atomic);
    marker_emit(&out, words + functions_begin, after_atomic - functions_begin);
    const uint32_t instructions[] = {
        (5u << 16) | SpvAccessChain, scalar_pointer, marker_ptr, marker, zero,
        (7u << 16) | SpvAtomicOr, uint_type, marker_result, marker_ptr, atomic[4], atomic[5], one,
    };
    marker_emit(&out, instructions, sizeof instructions / sizeof *instructions);
    marker_emit(&out, words + after_atomic, count - after_atomic);
    module[3] = next;
    *original = proof;
    *marked_count = (size_t)(out - module);
    return module;
}

static uint32_t *native_bitmap_mark(const uint32_t *words, size_t count, uint32_t binding,
    const NativeBitmapPlan *plan, RcNativeRmwWord *original, size_t *marked_count) {
    enum { HeaderWords = 31, SiteWords = 80, HeaderIds = 6, SiteIds = 14 };
    const RcNativeRmwWord proof = plan->receipt;
    const size_t extra = HeaderWords + (size_t)SiteWords * proof.bitmap_sites;
    if (count > SIZE_MAX / sizeof(uint32_t) - extra ||
        words[3] > UINT32_MAX - HeaderIds - SiteIds * proof.bitmap_sites) return NULL;
    const uint32_t *atomic = words + proof.atomic_word;
    const uint32_t uint_type = atomic[1];
    uint32_t bool_type = 0, source = 0, block_pointer = 0, scalar_pointer = 0;
    size_t annotations_end = 0, functions_begin = 0, entry = 0;
    const int interface_global = words[1] >= 0x00010400;
    for (size_t i = 5; i < proof.atomic_word; i += length(words + i)) {
        const uint32_t *p = words + i;
        const uint32_t op = opcode(p);
        if (op == SpvEntryPoint) {
            if (length(p) == UINT16_MAX || p[2] != proof.entry) return NULL;
            entry = i;
        } else if (op == SpvDecorate && length(p) == 4 && p[2] == SpvBinding && p[3] == binding) {
            return NULL;
        } else if (op == SpvTypeBool) {
            bool_type = p[1];
        } else if (op == SpvFunction && !functions_begin) {
            functions_begin = i;
        } else if (op == SpvAccessChain && length(p) == 6 && p[2] == atomic[3]) {
            scalar_pointer = p[1];
            source = p[3];
        }
        if (!annotations_end && op >= SpvTypeVoid && op <= SpvTypeForwardPointer) annotations_end = i;
    }
    if (!scalar_pointer || !entry || !annotations_end || !functions_begin || proof.atomic_word < functions_begin)
        return NULL;
    for (size_t i = annotations_end; i < functions_begin; i += length(words + i)) {
        const uint32_t *p = words + i;
        if (opcode(p) == SpvVariable && length(p) == 4 && p[2] == source) block_pointer = p[1];
    }
    const uint32_t entry_size = length(words + entry);
    const size_t after_entry = entry + entry_size;
    if (!block_pointer || after_entry > annotations_end || annotations_end > functions_begin) return NULL;

    uint32_t next = words[3];
    const uint32_t marker = next++, zero = next++, one = next++, five = next++, thirty_one = next++;
    const int add_bool = !bool_type;
    if (add_bool) bool_type = next++;

    uint32_t *module = malloc((count + extra) * sizeof *module);
    if (!module) return NULL;
    uint32_t *out = module;
    marker_emit(&out, words, entry);
    *out++ = ((entry_size + interface_global) << 16) | SpvEntryPoint;
    marker_emit(&out, words + entry + 1, entry_size - 1);
    if (interface_global) *out++ = marker;
    marker_emit(&out, words + after_entry, annotations_end - after_entry);
    const uint32_t decorations[] = {
        (4u << 16) | SpvDecorate, marker, SpvDescriptorSet, proof.storage_set,
        (4u << 16) | SpvDecorate, marker, SpvBinding, binding,
    };
    marker_emit(&out, decorations, sizeof decorations / sizeof *decorations);
    marker_emit(&out, words + annotations_end, functions_begin - annotations_end);
    if (add_bool) {
        *out++ = (2u << 16) | SpvTypeBool;
        *out++ = bool_type;
    }
    const uint32_t declarations[] = {
        (4u << 16) | SpvConstant, uint_type, zero, 0,
        (4u << 16) | SpvConstant, uint_type, one, 1,
        (4u << 16) | SpvConstant, uint_type, five, 5,
        (4u << 16) | SpvConstant, uint_type, thirty_one, 31,
        (4u << 16) | SpvVariable, block_pointer, marker, proof.storage_class,
    };
    marker_emit(&out, declarations, sizeof declarations / sizeof *declarations);
    size_t cursor = functions_begin;
    for (uint32_t i = 0; i < proof.bitmap_sites; ++i) {
        const NativeBitmapSite *site = &plan->sites[i];
        const uint32_t *issued = words + site->instruction;
        const size_t after_atomic = site->instruction + length(issued);
        marker_emit(&out, words + cursor, after_atomic - cursor);
        cursor = after_atomic;
        // Keep each original atomic and result intact. An out-of-range operation routes zero to the dummy word.
        const uint32_t index = next++, lane = next++, word = next++, slot = next++, bit = next++;
        const uint32_t source_length = next++, marker_length = next++, source_in = next++, marker_in = next++;
        const uint32_t in = next++, safe_slot = next++, safe_bit = next++, marker_ptr = next++, marker_result = next++;
        const uint32_t instructions[] = {
            (4u << 16) | SpvBitcast, uint_type, index, site->index,
            (5u << 16) | SpvBitwiseAnd, uint_type, lane, index, thirty_one,
            (5u << 16) | SpvShiftRightLogical, uint_type, word, index, five,
            (5u << 16) | SpvIAdd, uint_type, slot, word, one,
            (5u << 16) | SpvShiftLeftLogical, uint_type, bit, one, lane,
            (5u << 16) | SpvArrayLength, uint_type, source_length, source, 0,
            (5u << 16) | SpvArrayLength, uint_type, marker_length, marker, 0,
            (5u << 16) | SpvULessThan, bool_type, source_in, index, source_length,
            (5u << 16) | SpvULessThan, bool_type, marker_in, slot, marker_length,
            (5u << 16) | SpvLogicalAnd, bool_type, in, source_in, marker_in,
            (6u << 16) | SpvSelect, uint_type, safe_slot, in, slot, zero,
            (6u << 16) | SpvSelect, uint_type, safe_bit, in, bit, zero,
            (6u << 16) | SpvAccessChain, scalar_pointer, marker_ptr, marker, zero, safe_slot,
            (7u << 16) | SpvAtomicOr, uint_type, marker_result, marker_ptr, issued[4], issued[5], safe_bit,
        };
        marker_emit(&out, instructions, sizeof instructions / sizeof *instructions);
    }
    marker_emit(&out, words + cursor, count - cursor);
    module[3] = next;
    *original = proof;
    *marked_count = (size_t)(out - module);
    return module;
}

uint32_t *rc_native_rmw_fragment_mark(const uint32_t *words, size_t count, uint32_t binding,
                                       RcNativeRmwWord *original, size_t *marked_count) {
    if (original) *original = (RcNativeRmwWord){0};
    if (marked_count) *marked_count = 0;
    NativeBitmapPlan plan = {0};
    if (!binding || !original || !marked_count || !rc_native_rmw_fragment(words, count, &plan.receipt)) return NULL;
    plan.sites[0] = (NativeBitmapSite){plan.receipt.atomic_word, plan.receipt.index};
    return native_bitmap_mark(words, count, binding, &plan, original, marked_count);
}

uint32_t *rc_native_rmw_mesh_bitmap_mark(const uint32_t *words, size_t count, uint32_t binding,
                                         RcNativeRmwWord *original, size_t *marked_count) {
    if (original) *original = (RcNativeRmwWord){0};
    if (marked_count) *marked_count = 0;
    NativeBitmapPlan plan = {0};
    if (!binding || !original || !marked_count || !native_mesh_bitmap_plan(words, count, &plan)) return NULL;
    return native_bitmap_mark(words, count, binding, &plan, original, marked_count);
}
