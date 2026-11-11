#include <assert.h>
#include <stdio.h>
#include "nvn_obj.h"

static struct { const op_hdr *at[8]; int depth; } AH;
typedef struct { uint64_t target; } p_jump;
static const op_hdr *nvn_op_next(const op_hdr *h) { return (const op_hdr *)((const char *)h + h->size); }
static const void *nvn_op_payload(const op_hdr *h) { return h + 1; }
#include "mesh_run.inc"

typedef struct { op_hdr h; uint64_t payload[4]; } Record;
static Record rec[8];
static void set(int i, uint32_t op) { rec[i].h = (op_hdr){op, sizeof(Record)}; }

int main(void) {
    set(0, OP_DRAW_MESH_IND); set(1, OP_BIND_PROGRAM); set(2, OP_DRAW_MESH_IND);
    set(3, OP_REPORT_COUNTER); set(4, OP_DRAW_MESH_IND); set(5, OP_END);
    AH.at[0] = &rec[0].h;
    const void *out[8];
    const void *cur = nvn_op_payload(AH.at[0]);
    assert(vk_exec_mesh_run(cur, out, 8) == 2);
    assert(out[0] == nvn_op_payload(&rec[2].h) && out[1] == nvn_op_payload(&rec[4].h));
    assert(vk_exec_mesh_run(cur, out, 1) == 1);
    assert(vk_exec_mesh_run(NULL, out, 8) == 0);
    const uint32_t stops[] = {OP_END, OP_CALL, OP_DISPATCH, OP_DISPATCH_IND, OP_UPDATE_UB,
        OP_COPY_B2B, OP_COPY_B2T, OP_COPY_T2B, OP_COPY_T2T, OP_CLEAR_BUF, OP_CLEAR_TEX,
        OP_SET_RT, OP_BARRIER, OP_SIGNAL_EVENT, OP_WAIT_EVENT};
    for (unsigned i = 0; i < sizeof stops / sizeof *stops; i++) {
        set(1, stops[i]);
        assert(vk_exec_mesh_run(cur, out, 8) == 0);
    }
    set(1, OP_JUMP); rec[1].payload[0] = (uint64_t)(uintptr_t)&rec[4];
    assert(vk_exec_mesh_run(cur, out, 8) == 1 && out[0] == nvn_op_payload(&rec[4].h));
    rec[1].payload[0] = (uint64_t)(uintptr_t)&rec[1];
    assert(vk_exec_mesh_run(cur, out, 8) == 0); // Cycles are bounded by MESH_RUN_OPS.
    rec[1].h.size = 7;
    assert(vk_exec_mesh_run(cur, out, 8) == 0);
    puts("Mesh lookahead boundary checks passed");
}
