#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t atomic_word;
    uint32_t execution_model;
    uint32_t entry;
    uint32_t storage_set;
    uint32_t storage_binding;
    uint32_t c0_set;
    uint32_t c0_binding;
    uint32_t c0_byte;
    uint32_t add;
    uint32_t align_mask;
    uint32_t index;
    uint32_t storage_class;
    uint32_t bitmap_sites;
} RcNativeRmwWord;

// Match the only storage access to this descriptor and require complete output coverage of the selected module.
// The input is already-validated SPIR-V from the actual selected stage. This never proves execution.
int rc_native_rmw_word(const uint32_t *, size_t count, RcNativeRmwWord *);

// c0_value is from the guarded, actual C0 bound at the original read position.
// Return the exact guest word only if it is aligned and entirely inside the descriptor.
int rc_native_rmw_word_address(const RcNativeRmwWord *, uint32_t c0_value, uint64_t descriptor_address,
                               uint64_t descriptor_size, uint64_t *word);

// Inject one separate StorageBuffer atomic OR of 1 on the exact original atomic's executed path.
// The original attestation describes the unmodified input. Binding is a reserved, nonzero set-0 binding.
// The result is a malloc-owned module. No shader or execution receipt is created on failure.
uint32_t *rc_native_rmw_word_mark(const uint32_t *, size_t count, uint32_t binding,
                                  RcNativeRmwWord *, size_t *marked_count);

// Qualify a fragment shader's sole device-scope u32 atomic-min storage access.
// On execution the marked variant sets one bit corresponding to the exact original indexed word.
int rc_native_rmw_fragment(const uint32_t *, size_t count, RcNativeRmwWord *);
uint32_t *rc_native_rmw_fragment_mark(const uint32_t *, size_t count, uint32_t binding,
                                       RcNativeRmwWord *, size_t *marked_count);

// A mesh bitmap attests every guest write in a single module and storage descriptor. Site
// indices are checked at runtime against the actual source and marker descriptor lengths.
int rc_native_rmw_mesh_bitmap(const uint32_t *, size_t count, RcNativeRmwWord *);
uint32_t *rc_native_rmw_mesh_bitmap_mark(const uint32_t *, size_t count, uint32_t binding,
                                         RcNativeRmwWord *, size_t *marked_count);
