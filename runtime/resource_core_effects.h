// Conservative write ranges derived from the loaded shader, including carried
// shader caches. The program borrows immutable SPIR-V until destruction.
// Evaluation reads only immutable bytes matching the actual GPU descriptor;
// unavailable bytes and unsupported semantics preserve the caller's broad write
// authority.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct RcEffectProgram RcEffectProgram;
typedef struct RcEffectMemo RcEffectMemo;
typedef enum {
  RC_EFFECT_COMPLETE,
  RC_EFFECT_UNKNOWN,
  RC_EFFECT_OVERFLOW
} RcEffectResult;
// WHOLE names an ordinary descriptor root with unknown offset; begin and size are zero. Accepting its physical
// extent requires the consumer's actual robustness and descriptor/backing proof. It never denotes a GPA table.
typedef enum { RC_EFFECT_GPA, RC_EFFECT_SSBO, RC_EFFECT_SSBO_WHOLE, RC_EFFECT_GPA_TWIN } RcEffectKind;
enum {
  RC_EFFECT_WRITES_BUFFER = 1u,
  RC_EFFECT_WRITES_GPA = 2u,
  RC_EFFECT_WRITES_IMAGE = 4u,
  RC_EFFECT_WRITES_UNKNOWN = 8u
};
// One linear preparation pass. Zero proves no guest-addressable shader writes;
// framebuffer outputs are separate.
uint32_t rc_effect_write_class(const uint32_t *words, size_t word_count);
typedef struct {
  RcEffectKind kind;
  uint32_t set, binding;
  uint64_t begin, size; // GPA: guest bytes. SSBO: bytes relative to the
                        // unchanged bound descriptor.
} RcEffectRange;
typedef int (*RcEffectRead)(void *context, uint32_t set, uint32_t binding,
                            uint64_t offset, size_t size, void *destination);
typedef int (*RcEffectBuiltin)(void *context, uint32_t builtin,
                               uint32_t component, uint64_t *minimum,
                               uint64_t *maximum);
typedef struct {
  RcEffectRead read;
  void *context;
  uint32_t direct_dispatch, groups[3], local[3];
  void *workspace;
  size_t workspace_bytes;
  RcEffectBuiltin builtin; // Optional exact draw index bounds. Unavailable
                           // builtins retain their full domain.
} RcEffectInputs;

RcEffectProgram *rc_effect_compile(const uint32_t *words, size_t word_count);
void rc_effect_free(RcEffectProgram *program);
size_t rc_effect_workspace_size(const RcEffectProgram *program);
// Number of writes found; zero is a proved absence only when
// rc_effect_supported also returns nonzero.
size_t rc_effect_write_count(const RcEffectProgram *program);
int rc_effect_supported(const RcEffectProgram *program);
// UNKNOWN/OVERFLOW publishes no partial result. No allocation, mutation of
// program, or guest reads occur here.
RcEffectResult rc_effect_evaluate(const RcEffectProgram *program,
                                  const RcEffectInputs *inputs,
                                  RcEffectRange *ranges, size_t capacity,
                                  size_t *count);
// Report bytes definitely stored by the direct dispatch of the selected native module. Stores must be
// unconditional. Single-invocation stores can use the exact immutable inputs of that invocation. Multiple
// invocations are supported only for ordinary SSBO arrays with the module's verified literal local size
// and nonwrapping affine addresses that tile each written span without gaps or duplicate stores.
// RC_EFFECT_GPA and RC_EFFECT_GPA_TWIN remain single-invocation proofs for the primary and twin table routes.
// All reported offsets still need checked native descriptor bounds or a real GPA projection.
RcEffectResult rc_effect_evaluate_definite_stores(const RcEffectProgram *program, const RcEffectInputs *inputs,
    RcEffectRange *ranges, size_t capacity, size_t *count);
int rc_effect_has_definite_store_plan(const RcEffectProgram *program);
// Every guest-visible store is an ordered primary/twin GPA pair with the same address and value SSA inputs.
int rc_effect_has_mirrored_gpa_store_plan(const RcEffectProgram *program);
// A bounded caller-owned memo shares the caller's interpreter workspace.
// Initialize malloc-aligned storage once. Reads must be pure for an evaluation
// and return only bytes matching the actual immutable GPU binding. Hits re-read
// every dependency (including unavailable inputs); the caller still validates
// its captured input leases.
size_t rc_effect_memo_size(void);
RcEffectMemo *rc_effect_memo_init(void *storage, size_t bytes);
RcEffectResult rc_effect_evaluate_memo(const RcEffectProgram *program,
                                       const RcEffectInputs *inputs,
                                       RcEffectMemo *memo,
                                       RcEffectRange *ranges, size_t capacity,
                                       size_t *count, int *reused);
// Prove the complete read footprint of one ordinary storage binding. This
// permits the runtime to capture/rebind a sufficient prefix instead of copying
// the descriptor's potentially much larger accessible backing span.
RcEffectResult rc_effect_read_ranges(const RcEffectProgram *program,
                                     const RcEffectInputs *inputs, uint32_t set,
                                     uint32_t binding, RcEffectRange *ranges,
                                     size_t capacity, size_t *count);
