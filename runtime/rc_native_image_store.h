#pragma once
#include <stddef.h>
#include <stdint.h>

// A compute entry point accesses a single direct 2D storage descriptor.
// A nonzero local size also proves that its only store is unconditional and
// writes one texel per invocation at global XY. Z is checked at dispatch.
typedef enum {
    RC_NATIVE_IMAGE_SCALAR_NONE,
    RC_NATIVE_IMAGE_FLOAT32,
    RC_NATIVE_IMAGE_SINT32,
    RC_NATIVE_IMAGE_UINT32,
} RcNativeImageScalar;

typedef struct {
    uint32_t set, binding, local[3];
    uint32_t image_format; // SPIR-V Image Format, including Unknown (0)
    RcNativeImageScalar scalar;
    int reads_storage;
} RcNativeImageStore;

typedef enum {
    RC_NATIVE_IMAGE_STORE_NONE,
    RC_NATIVE_IMAGE_STORE_BINDING,
    RC_NATIVE_IMAGE_STORE_XY,
} RcNativeImageStoreResult;

RcNativeImageStoreResult rc_native_image_store(const uint32_t *words, size_t count, RcNativeImageStore *out);
int rc_native_image_store_dispatch_safe(const RcNativeImageStore *, const uint32_t groups[3],
                                       uint32_t width, uint32_t height, int robust_image_access);
int rc_native_image_store_covers(const RcNativeImageStore *, const uint32_t groups[3],
                                uint32_t width, uint32_t height, int robust_image_access);
// The caller translates its actual view format using Vulkan's compatibility table
// and reports whether that format supports storage writes with Unknown format.
int rc_native_image_store_format(const RcNativeImageStore *, uint32_t compatible_format,
                                 int formatless_writes, int formatless_reads);

enum {
    RC_NATIVE_IMAGE_SEED_MAX_BINDINGS = 8,
    RC_NATIVE_IMAGE_SEED_READ = 1,
    RC_NATIVE_IMAGE_SEED_WRITE = 2,
    RC_NATIVE_IMAGE_SEED_ATOMIC = 4,
    RC_NATIVE_IMAGE_SEED_2D = 0,
    RC_NATIVE_IMAGE_SEED_3D = 1,
};

typedef struct {
    uint8_t binding, image_format, scalar, access, dimension;
} RcNativeImageSeedBinding;

typedef struct {
    RcNativeImageSeedBinding entries[RC_NATIVE_IMAGE_SEED_MAX_BINDINGS];
    uint8_t count;
} RcNativeImageSeedBindings;

// Exact storage accesses of the final compute module. Callers must establish
// defined prior texels and robust image access separately.
int rc_native_image_seed_bindings(const uint32_t *words, size_t count, RcNativeImageSeedBindings *out);
// Separate opt-in for an exact direct 2D R32ui image read-modify-write.
// No texel coverage or overwrite proof is returned.
int rc_native_image_seed_rmw_bindings(const uint32_t *words, size_t count, RcNativeImageSeedBindings *out);
int rc_native_image_seed_store(const RcNativeImageSeedBinding *, RcNativeImageStore *out);
int rc_native_image_seed_binding_format(const RcNativeImageSeedBinding *, uint32_t compatible_format,
                                        int formatless_writes, int formatless_reads);
