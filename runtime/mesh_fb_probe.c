// nx2navi.exe --probe-meshfb: the mesh shader fallback (mesh_fb.c, vk_mesh_fb.c, vk_exec_draw.c "mesh fallback").
//
// Draw part, no game files: two mesh programs of the runtime's own in the shape the translator gives one
// (shaders/meshfb_probe_*.mesh, .task), one without a task stage and one fed by a task program, drawn through the
// replay into an RGBA8 target with direct and with indirect workgroup counts, and both in one pass after each other.
// Each picture is drawn twice, through VK_EXT_mesh_shader and through the fallback, and the two are compared texel by
// texel; without mesh shaders on the device only the fallback draws, and the probe says that nothing was compared.
//
// Module part, with NX2NAVI_PROBE_MESHFB_DIR=<dir>: every "<name>" of <dir>/list.txt names <dir>/<name>.in (a stored
// shader input, gfx_shader.c in_store: control section and code) and <dir>/<name>.spv (its translation). The compute
// module and, for a mesh program, a vertex module are written to <dir>/out/<name>.cs.spv and .vs.spv for spirv-val
// (work the caller does; tests of game shaders stay outside the repository). Where <dir>/<name>.res.json (the
// translation's reflection) is there too, the compute pipeline of the module is built on this device, as its first
// draw would build it.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "mesh_fb.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_mesh_fb.h"
#include "platform/platform.h"

#include "meshfb_probe_f_spv.h"
#include "meshfb_probe_m_spv.h"
#include "meshfb_probe_t_spv.h"
#include "meshfb_probe_tm_spv.h"
#include "meshfb_storage_probe_f_spv.h"
#include "meshfb_storage_probe_t_spv.h"
#include "meshfb_storage_probe_tm_spv.h"
#include "meshfb_task_pointer_probe_task_spv.h"
#include "meshfb_task_pointer_probe_mesh_spv.h"

#define TEX_W 256
#define TEX_H 144
#define CHECK(cond, ...)                                                                                                \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("probe-meshfb: FAILED: ");                                                                          \
            printf(__VA_ARGS__);                                                                                       \
            printf("\n");                                                                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

void vk_exec_probe_shader(const uint8_t *ctl, const GfxShader *s);   // vk_exec_draw.c
void vk_exec_draw_shutdown(void);
void vk_exec_xfer_shutdown(void);

static void meshfb_finish_probe(const GfxBackend *b) {
    b->finish();
    vk_pipeline_stop_preparation();
    vk_exec_draw_mfb_stop();
    // A recorded but unsubmitted scene can still own a pipeline revision after the last issued fence.
    vk_sync_shutdown();
    if (!vk_device_is_lost()) {
        vk_pipeline_shutdown();
        if (!vk_device_is_lost()) {
            vk_exec_draw_shutdown();
            vk_exec_xfer_shutdown();
        }
    }
    vk_backend_shutdown();
}

// ---- module part

static uint8_t *read_all(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) free(b), b = NULL;
    fclose(f);
    if (b) b[n] = 0, *size = (size_t)n;
    return b;
}

static int write_all(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok;
}

static int modules_part(const char *dir) {
    char path[1024];
    size_t list_n = 0;
    snprintf(path, sizeof path, "%s/list.txt", dir);
    char *list = (char *)read_all(path, &list_n);
    CHECK(list, "no %s", path);
    int n_task = 0, n_mesh = 0, n_fail = 0, n_pipe = 0, n_pipe_fail = 0;
    double pipe_ms = 0, pipe_worst = 0;
    uint64_t pipe_worst_words = 0;
    const GfxBackend *b = &gfx_vulkan_backend;
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22};
    int device = 0;
    for (char *name = strtok(list, "\r\n"); name; name = strtok(NULL, "\r\n")) {
        size_t in_n = 0, spv_n = 0;
        snprintf(path, sizeof path, "%s/%s.in", dir, name);
        uint8_t *in = read_all(path, &in_n);
        snprintf(path, sizeof path, "%s/%s.spv", dir, name);
        uint8_t *spv = read_all(path, &spv_n);
        uint32_t h[6];   // magic, version, stage, code length, c1 length, size of the answers (gfx_shader.c InHeader)
        if (!in || !spv || in_n < sizeof h) {
            printf("probe-meshfb: %s: input or translation missing\n", name), n_fail++;
            free(in), free(spv);
            continue;
        }
        memcpy(h, in, sizeof h);
        const uint8_t *control = in + sizeof h + h[5], *code = control + 0x880;
        int stage = (int)h[2];   // GFX_STAGE_TS / GFX_STAGE_MS are the NVN stage numbers 6 and 7
        char err[200] = "";
        MeshFbInfo m;
        uint64_t on = 0, vn = 0;
        uint32_t *cs = NULL, *vs = NULL;
        if (in_n < sizeof h + h[5] + 0x880 + 0x80 || (stage != 6 && stage != 7)) snprintf(err, sizeof err, "not a task or mesh input");
        else if (meshfb_info(control, code, stage, &m, err, sizeof err)) {
            uint32_t top = 0;
            const uint32_t *w = (const uint32_t *)spv;
            for (size_t i = 5; i < spv_n / 4;) {
                uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (!len) break;
                if (op == 71 && len == 4 && w[i + 2] == 33 && w[i + 3] > top) top = w[i + 3];
                i += len;
            }
            cs = stage == 6 ? meshfb_task_cs(w, spv_n / 4, &m, top + 1, &on, err, sizeof err)
                            : meshfb_mesh_cs(w, spv_n / 4, &m, top + 1, m.payload_bytes / 4, &on, err, sizeof err);
            if (cs && stage == 7) {
                uint8_t type[32] = {0};
                uint32_t mask = 0;
                for (uint32_t l = 0; l < m.nlines; l++)
                    if (m.lines[l] >= 0x80) mask |= 1u << (m.lines[l] - 0x80) / 16;
                for (int g = 0; g < 32; g++) type[g] = (uint8_t)(g % 3);   // every output type gets made somewhere
                vs = meshfb_vs(&m, mask, type, &vn);
                if (!vs) snprintf(err, sizeof err, "no vertex module");
            }
        }
        if (cs && (stage == 6 || vs)) {
            snprintf(path, sizeof path, "%s/out/%s.cs.spv", dir, name);
            int ok = write_all(path, cs, on * 4);
            if (vs) snprintf(path, sizeof path, "%s/out/%s.vs.spv", dir, name), ok = ok && write_all(path, vs, vn * 4);
            if (!ok) printf("probe-meshfb: %s: cannot write under %s/out\n", name, dir), n_fail++;
            else if (stage == 6) n_task++;
            else n_mesh++;
            size_t res_n = 0;
            snprintf(path, sizeof path, "%s/%s.res.json", dir, name);
            char *res = (char *)read_all(path, &res_n);
            if (res) {   // the pipeline, with the layout the reflection gives
                if (!device) {   // the first pool brings the device up
                    uint64_t heap = 0;
                    CHECK(mem_set_heap_size(16ull << 20, &heap) == 0 && heap, "SetHeapSize 16 MiB");
                    pool.mem = pool.game_mem = (uint8_t *)(uintptr_t)heap + 0x100000, pool.size = 0x100000;
                    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool");
                    device = 1;
                }
                uint32_t top = 0;
                const uint32_t *w = (const uint32_t *)spv;
                for (size_t i = 5; i < spv_n / 4;) {
                    uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                    if (!len) break;
                    if (op == 71 && len == 4 && w[i + 2] == 33 && w[i + 3] > top) top = w[i + 3];
                    i += len;
                }
                static const int task_slots[1] = {MESHFB_CB_TASK}, mesh_slots[3] = {MESHFB_CB_TASK, MESHFB_CB_MESH, MESHFB_CB_LIST};
                char *cres = stage == 6 ? meshfb_res_json(res, top + 1, task_slots, 1)
                           : m.has_task ? meshfb_res_json(res, top + 1, mesh_slots, 3) : meshfb_res_json(res, top + 2, mesh_slots + 1, 1);
                const GfxShader *sh = cres ? gfx_shader_own(cs, on, cres) : NULL;
                const VkPipeInfo *info = NULL;
                uint64_t t0 = plat_now_ns();
                VkPipeline p = sh ? vk_pipeline_compute(sh, &info) : VK_NULL_HANDLE;
                double ms = (double)(plat_now_ns() - t0) / 1e6;
                if (p && info) {
                    n_pipe++, pipe_ms += ms;
                    if (ms > pipe_worst) pipe_worst = ms, pipe_worst_words = on;
                } else {
                    n_pipe_fail++;
                    printf("probe-meshfb: %s (stage %d, %llu words): %s\n", name, stage, (unsigned long long)on,
                           sh ? "its compute pipeline was not built" : "its reflection could not be made");
                }
                fflush(stdout);
                free(cres), free(res);
            }
        } else {
            printf("probe-meshfb: %s (stage %d): %s\n", name, stage, err), n_fail++;
        }
        free(cs), free(vs), free(in), free(spv);
    }
    free(list);
    printf("probe-meshfb: module part: %d task and %d mesh programs converted, %d failed\n", n_task, n_mesh, n_fail);
    if (device) {
        printf("probe-meshfb: module part: %d compute pipelines built on this device (%.0f ms each, the longest %.0f ms for %llu "
               "words), %d not built\n", n_pipe, n_pipe ? pipe_ms / n_pipe : 0.0, pipe_worst, (unsigned long long)pipe_worst_words,
               n_pipe_fail);
        b->pool_fini(&pool);
        b->submit(NULL, 0);   // the pool's buffers go when a submission after their release has completed
        meshfb_finish_probe(b);
    }
    return n_fail != 0 || n_pipe_fail != 0;
}

// ---- draw part

static uint8_t *put(uint8_t *p, uint32_t op, const void *payload, uint32_t n) {
    op_hdr h = {op, (uint32_t)((sizeof h + n + 7) & ~7u)};
    memset(p, 0, h.size);
    memcpy(p, &h, sizeof h);
    if (n) memcpy(p + sizeof h, payload, n);
    return p + h.size;
}

// An NVN2 control section of a task (6), mesh (7) or fragment (1) stage as vk_exec_draw.c and mesh_fb.c read it, and
// a code block whose header maps the outputs: position and generic vector 0.
static void fake_stage(NVNshaderData *sd, uint32_t stage, int fed_by_task) {
    uint8_t *ctl = calloc(1, 0x880), *code = calloc(1, 0x100);
    uint32_t magic = 0x98761234u, len = 0x100, st = stage << 16;
    uint32_t w58 = 32u | 1u << 24;                           // 32 threads, a payload of one 256-byte unit
    uint32_t w5c = 3u << 24 | (fed_by_task ? 1u << 16 : 0);   // triangles
    uint32_t w60 = 64u | 32u << 16;                           // 64 vertices, 32 primitives
    memcpy(ctl, &magic, 4), memcpy(ctl + 0x38, &len, 4), memcpy(ctl + 0x54, &st, 4);
    if (stage >= 6) memcpy(ctl + 0x58, &w58, 4);
    if (stage == 7) memcpy(ctl + 0x5c, &w5c, 4), memcpy(ctl + 0x60, &w60, 4), code[0x35] = 0xf0, code[0x36] = 0x0f;
    sd->control = ctl, sd->data = (uint64_t)(uintptr_t)code;
}

static const char RES_MS[] = "{\"key\":\"probe\",\"stage\":\"ms\",\"notes\":[],\"cbufs\":[{\"binding\":1,\"slot\":3}],"
                             "\"sbufs\":[],\"textures\":[],\"images\":[]}";
static const char RES_TS[] = "{\"key\":\"probe\",\"stage\":\"ts\",\"notes\":[],\"cbufs\":[{\"binding\":1,\"slot\":3}],"
                             "\"sbufs\":[],\"textures\":[],\"images\":[]}";
static const char RES_FS[] = "{\"key\":\"probe\",\"stage\":\"ps\",\"notes\":[],\"cbufs\":[],\"sbufs\":[],\"textures\":[],"
                             "\"images\":[]}";

// The program of a mesh stage, its task stage if any, and the fragment stage.
static prog_t *make_prog(const GfxShader *ms, const GfxShader *ts, const GfxShader *fs) {
    static uint64_t serial = 0x9000;
    prog_t *p = calloc(1, sizeof *p);
    p->magic = NVN_PROG_MAGIC, p->serial = ++serial;
    fake_stage(&p->sd[p->nstage], 7, ts != NULL), vk_exec_probe_shader(p->sd[p->nstage++].control, ms);
    if (ts) fake_stage(&p->sd[p->nstage], 6, 0), vk_exec_probe_shader(p->sd[p->nstage++].control, ts);
    fake_stage(&p->sd[p->nstage], 1, 0), vk_exec_probe_shader(p->sd[p->nstage++].control, fs);
    return p;
}

typedef struct { prog_t *prog; uint32_t count; int indirect; uint32_t grid[2]; float tint[4]; } MeshDraw;
typedef struct {
    uint64_t table, selected, verify;
    p_copy_b2b seed[3];
} MeshPointerProbe;
typedef struct {
    uint64_t task, mesh, final;
    int skip_final;
    const MeshPointerProbe *pointer;
} MeshStorageProbe;

void vk_exec_probe_mfb_storage_cancel_final(const void *);
uint32_t vk_exec_probe_mfb_storage_canceled(void);
uint32_t vk_exec_probe_mfb_pointer_phases(void);

static void cancel_mesh_picture(VkExec *ex, void *opaque) {
    (void)ex;
    unsigned *pending = opaque;
    *pending = vk_meshfb_probe_pending_reports();
    vk_exec_cancel_temporal_preparation();
}

// Uniform data in the pool, 0x1000 apart per draw: the mesh stage's block at 0x90000, the task stage's at 0x98000; the
// indirect records at 0xa0000.
static int draw_picture(const GfxBackend *b, pool_t *pool, tex_t *color, const MeshDraw *d, int n, uint32_t *px,
                        int probe_partial, const MeshStorageProbe *storage) {
    CHECK(!probe_partial || (n == 1 && d[0].indirect), "the overflow diagnostic must be one indirect draw");
    CHECK(!storage || (n == 1 && !probe_partial), "the phase diagnostic must be one draw");
    const void *partial_payload = NULL, *storage_payload = NULL;
    static _Alignas(8) uint8_t stream[16384];
    uint8_t *q = put(stream, OP_BEGIN, NULL, 0);
    const MeshPointerProbe *pointer = storage ? storage->pointer : NULL;
    if (pointer) for (unsigned i=0;i<3;++i) q=put(q,OP_COPY_B2B,&pointer->seed[i],sizeof pointer->seed[i]);
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1, rt.color[0] = color;
    q = put(q, OP_SET_RT, &rt, sizeof rt);
    const p_clear_color c0 = {0, 0xf, {0.2f, 0.4f, 0.6f, 1.0f}};
    q = put(q, OP_CLEAR_COLOR, &c0, sizeof c0);
    p_viewports vp;
    memset(&vp, 0, sizeof vp);
    vp.n = 1, vp.v[0][2] = TEX_W, vp.v[0][3] = TEX_H, vp.v[0][5] = 1.0f;
    q = put(q, OP_VIEWPORTS, &vp, sizeof vp);
    const p_scissor full = {0, 0, 1 << 20, 1 << 20};
    q = put(q, OP_SCISSOR, &full, sizeof full);
    for (int i = 0; i < n; i++) {
        uint8_t *mu = pool->mem + 0x90000 + 0x1000 * i, *tu = pool->mem + 0x98000 + 0x1000 * i;
        uint32_t *rec = (uint32_t *)(pool->mem + 0xa0000 + 0x100 * i);
        uint32_t grid[4] = {d[i].grid[0], d[i].grid[1], 0, 0}, shift[4] = {1, 0, 0, 0};
        if (pointer) {
            shift[1]=(uint32_t)pointer->selected;
            shift[2]=(uint32_t)(pointer->selected>>32);
        }
        memcpy(mu, d[i].tint, 16), memcpy(mu + 16, grid, 16), memcpy(tu, shift, 16);
        if (pointer) {
            b->guest_written((uintptr_t)mu,32);
            b->guest_written((uintptr_t)tu,16);
        }
        rec[0] = d[i].count, rec[1] = 0, rec[2] = 0;
        const p_bind_program bp = {d[i].prog, 0xc2};
        q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
        const p_bind_buf bm = {7, 0, (uint64_t)(uintptr_t)mu, 256}, bt = {6, 0, (uint64_t)(uintptr_t)tu, 256};
        q = put(q, OP_BIND_UBO, &bm, sizeof bm);
        q = put(q, OP_BIND_UBO, &bt, sizeof bt);
        if (pointer) {
            const p_bind_buf writer={6,1,pointer->table,16}; // NVN task stage
            const p_bind_buf reader={7,1,pointer->table,16}; // NVN mesh stage
            q=put(q,OP_BIND_SSBO,&writer,sizeof writer);
            q=put(q,OP_BIND_SSBO,&reader,sizeof reader);
        } else if (storage) {
            const p_bind_handles task_image = {.stage = 6, .n = 1, .h = {storage->task}};
            const p_bind_handles mesh_images = {.stage = 7, .n = 2, .h = {storage->task, storage->mesh}};
            const p_bind_handles final_image = {.stage = NVN_STAGE_FS, .n = 1, .h = {storage->final}};
            q = put(q, OP_BIND_IMG, &task_image, sizeof task_image);
            q = put(q, OP_BIND_IMG, &mesh_images, sizeof mesh_images);
            q = put(q, OP_BIND_IMG, &final_image, sizeof final_image);
            if (storage->skip_final) storage_payload = nvn_op_payload((const op_hdr *)q);
        }
        if (d[i].indirect) {
            const p_draw_mesh_ind di = {(uint64_t)(uintptr_t)rec, 0, 0, 0, 0};
            if (probe_partial) partial_payload = nvn_op_payload((const op_hdr *)q);
            q = put(q, OP_DRAW_MESH_IND, &di, sizeof di);
        } else {
            const p_draw_mesh dm = {0, d[i].count, 0};
            q = put(q, OP_DRAW_MESH, &dm, sizeof dm);
        }
    }
    if (pointer) {
        const p_copy_b2b verify={pointer->table,pointer->verify,8};
        q=put(q,OP_COPY_B2B,&verify,sizeof verify);
    }
    q = put(q, OP_END, NULL, 0);
    CHECK(q <= stream + sizeof stream, "op stream overflow");
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    if (!px) {
        // This is the last submission in the probe. Stop after recording its mesh copy and before native handoff.
        unsigned pending = 0;
        uint64_t prior = vk_sync_issued(), short_before = vk_meshfb_short();
        vk_exec_probe_pass_arm(cancel_mesh_picture, &pending);
        vk_exec_probe_mesh_partial(partial_payload);
        b->submit(&handle, 1);
        b->finish();
        vk_exec_probe_mesh_partial(NULL);
        CHECK(pending == 1 && vk_sync_issued() == prior && !vk_meshfb_probe_pending_reports() &&
              vk_meshfb_short() == short_before,
              "the canceled mesh tail left a report (recorded %u pending, issued %llu to %llu)",
              pending, (unsigned long long)prior, (unsigned long long)vk_sync_issued());
        printf("probe-meshfb: recorded and canceled one unsubmitted statistics copy without publishing a report\n");
        return 0;
    }
    if (partial_payload) vk_exec_probe_mesh_partial(partial_payload);
    if (storage_payload) vk_exec_probe_mfb_storage_cancel_final(storage_payload);
    uint64_t f = b->submit(&handle, 1);
    int finished = b->fence_wait(f, 60000000000ull);
    if (storage_payload) vk_exec_probe_mfb_storage_cancel_final(NULL);
    if (partial_payload) vk_exec_probe_mesh_partial(NULL);
    CHECK(finished, "fence_wait of a mesh draw submission timed out");
    NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
    b->read_texels(color, NULL, &r, (uint8_t *)px, TEX_W * 4, TEX_W * TEX_H * 4);
    return 0;
}

static int drawn_texels(const uint32_t *px) {
    int n = 0;
    for (int i = 0; i < TEX_W * TEX_H; i++) n += px[i] != px[0];   // texel 0 is in the margin no cell covers
    return n;
}

// Texels of a and b that differ, and the largest difference of a channel.
static int differing(const uint32_t *a, const uint32_t *b, int *worst) {
    int n = 0;
    *worst = 0;
    for (int i = 0; i < TEX_W * TEX_H; i++) {
        if (a[i] == b[i]) continue;
        n++;
        for (int c = 0; c < 4; c++) {
            int d = (int)(a[i] >> 8 * c & 0xff) - (int)(b[i] >> 8 * c & 0xff);
            if (d < 0) d = -d;
            if (d > *worst) *worst = d;
        }
    }
    return n;
}

static int storage_draw_part(const GfxBackend *b, pool_t *pool, tex_t *color) {
    static const char task_reflection[] = "{\"key\":\"probe-storage-task\",\"stage\":\"ts\","
        "\"cbufs\":[{\"binding\":1,\"slot\":3}],\"sbufs\":[],\"images\":["
        "{\"binding\":2,\"cbuf_slot\":2,\"handle\":72,\"type\":3,\"format\":\"R32Uint\",\"flags\":4}]}";
    static const char mesh_reflection[] = "{\"key\":\"probe-storage-mesh\",\"stage\":\"ms\","
        "\"cbufs\":[{\"binding\":1,\"slot\":3}],\"sbufs\":[],\"images\":["
        "{\"binding\":2,\"cbuf_slot\":2,\"handle\":72,\"type\":3,\"format\":\"R32Uint\"},"
        "{\"binding\":3,\"cbuf_slot\":2,\"handle\":74,\"type\":3,\"format\":\"R32Uint\",\"flags\":4}]}";
    static const char final_reflection[] = "{\"key\":\"probe-storage-final\",\"stage\":\"ps\",\"cbufs\":[],"
        "\"images\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":72,\"type\":3,"
        "\"format\":\"R32Uint\",\"flags\":4}]}";
    const GfxShader *ts = gfx_shader_own(meshfb_storage_probe_t_spv, sizeof meshfb_storage_probe_t_spv / 4, task_reflection);
    const GfxShader *ms = gfx_shader_own(meshfb_storage_probe_tm_spv, sizeof meshfb_storage_probe_tm_spv / 4, mesh_reflection);
    const GfxShader *fs = gfx_shader_own(meshfb_storage_probe_f_spv, sizeof meshfb_storage_probe_f_spv / 4, final_reflection);
    CHECK(ts && ms && fs && gfx_shader_write_inventory_complete(ts) && gfx_shader_write_inventory_complete(ms) &&
          gfx_shader_write_inventory_complete(fs), "real task, mesh and final storage shaders");
    prog_t *program = make_prog(ms, ts, fs);
    const MeshDraw draw = {program, 10, 0, {8, 5}, {1, 1, 1, 1}};
    enum { SIDE = 32, PIXELS = SIDE * SIDE, TARGETS = 3 };
    const NVNcopyRegion all = {0, 0, 0, SIDE, SIDE, 1};
    const uint32_t seed = 0x04020010u;
    static uint32_t picture[TEX_W * TEX_H];
    uint32_t texels[PIXELS], read[PIXELS];
    const uint32_t canceled = vk_exec_probe_mfb_storage_canceled();
    tex_t fixtures[2][TARGETS];
    for (unsigned round = 0; round < 2; ++round) {
        tex_t *storage = fixtures[round];
        for (unsigned t = 0; t < TARGETS; ++t) {
            const uint64_t off = 0x100000 + round * 0x30000 + t * 0x10000;
            storage[t] = (tex_t){.magic = NVN_TEX_MAGIC, .pool = pool, .pool_flags = pool->flags, .off = off,
                .mem = pool->mem + off, .serial = nvn_new_serial()};
            storage[t].d.target = NVN_TEXTURE_TARGET_2D;
            storage[t].d.format = NVN_FORMAT_R32UI;
            storage[t].d.flags = NVN_TEXTURE_FLAGS_IMAGE;
            storage[t].d.width = storage[t].d.height = SIDE;
            storage[t].d.depth = storage[t].d.levels = 1;
            nvn_tex_layout(&storage[t].d, &storage[t].L);
            for (unsigned i = 0; i < PIXELS; ++i) texels[i] = seed + t * PIXELS + i;
            CHECK(!nvn_swizzle(&storage[t].d, 0, 0, texels, SIDE * 4, storage[t].mem), "pack storage target %u", t);
            b->guest_written((uintptr_t)storage[t].mem, storage[t].L.size);
        }
    }
    // Both rounds share one host write-watch region. Establish every immutable native seed before submitting either draw.
    VkImage prior[2][TARGETS];
    for (unsigned round = 0; round < 2; ++round) {
        tex_t *storage = fixtures[round];
        for (unsigned t = 0; t < TARGETS; ++t) {
            b->read_texels(&storage[t], NULL, &all, (uint8_t *)read, SIDE * 4, sizeof read);
            for (unsigned i = 0; i < PIXELS; ++i)
                CHECK(read[i] == seed + t * PIXELS + i, "storage target %u original texel %u reached its native image", t, i);
            const VkTexImage *image = vk_texture_get(&storage[t]);
            CHECK(image && image->image && !image->gpu_owned && (image->usage & VK_IMAGE_USAGE_STORAGE_BIT),
                  "storage round %u target %u starts immutable with real storage usage (image %p, owned %d, usage %#x)",
                  round, t, image ? (void *)image->image : NULL, image ? image->gpu_owned : -1, image ? image->usage : 0);
            prior[round][t] = image->image;
        }
    }
    for (unsigned cancel = 0; cancel < 2; ++cancel) {
        tex_t *storage = fixtures[cancel];
        for (unsigned t = 0; t < TARGETS; ++t) vk_texture_register(30 + t, &storage[t], NULL);
        const MeshStorageProbe inputs = {.task = 30, .mesh = 31, .final = 32, .skip_final = cancel != 0};
        if (draw_picture(b, pool, color, &draw, 1, picture, 0, &inputs)) return 1;
        const uint32_t canceled_now = vk_exec_probe_mfb_storage_canceled();
        CHECK(canceled_now == canceled + cancel, "round %u exact unissued final cancellation witness", cancel);
        printf("probe-meshfb: round %u finished native commands, independently canceled %u final phase\n",
               cancel, canceled_now - canceled);
        for (unsigned t = 0; t < TARGETS; ++t) {
            const tex_t *pin = nvn_texture_pin_acquire(&storage[t]);
            const VkTexImage *current = vk_texture_get(pin);
            int detached = t < 2 || !cancel;
            const void *attachment = nvn_texture_pin_attachment(pin);
            CHECK(current && current->image && ((current->image != prior[cancel][t]) == detached) &&
                  (!!attachment == detached), "round %u target %u expected detached %d old image %p current %p pin %p owning %d",
                  cancel, t, detached, (void *)prior[cancel][t], current ? (void *)current->image : NULL,
                  attachment, current ? current->gpu_owned : -1);
            b->read_texels(pin, NULL, &all, (uint8_t *)read, SIDE * 4, sizeof read);
            nvn_texture_pin_release(pin);
            for (unsigned i = 0; i < PIXELS; ++i) {
                uint32_t expected = seed + t * PIXELS + i;
                if (t == 0 && i < draw.count) expected = 0x75000000u | i;
                if (t == 1 && i < draw.count && (i * 3u + 1u) % 5u) expected = 0x76000000u | i;
                if (t == 2 && !cancel && !i) expected |= 0x01000000u;
                CHECK(read[i] == expected, "storage operation %u target %u pixel %u got %#x expected %#x",
                      cancel, t, i, read[i], expected);
            }
            b->texture_release(&storage[t]);
        }
    }
    printf("probe-meshfb: both real task and mesh dispatches shared the prepared storage image, "
           "final-only cancellation preserved its image, and %u exact storage pixels matched\n", 2 * TARGETS * PIXELS);
    return 0;
}

static int task_pointer_draw_part(const GfxBackend *b, uint8_t *guest, pool_t *pool, tex_t *color) {
    const GfxShader *ts=gfx_shader_own(meshfb_task_pointer_probe_task_spv,sizeof meshfb_task_pointer_probe_task_spv/4,
        "{\"key\":\"probe-task-pointer-write\",\"stage\":\"ts\",\"cbufs\":[{\"binding\":1,\"slot\":3}],"
        "\"sbufs\":[{\"binding\":2,\"slot\":0,\"cb_slot\":0,\"cb_offset\":416,\"flags\":1}]}");
    const GfxShader *ms=gfx_shader_own(meshfb_task_pointer_probe_mesh_spv,sizeof meshfb_task_pointer_probe_mesh_spv/4,
        "{\"key\":\"probe-task-pointer-read\",\"stage\":\"ms\",\"cbufs\":[{\"binding\":1,\"slot\":3}],"
        "\"sbufs\":[{\"binding\":2,\"slot\":0,\"cb_slot\":0,\"cb_offset\":480,\"flags\":0,\"ptr\":0}]}");
    const GfxShader *fs=gfx_shader_own(meshfb_probe_f_spv,sizeof meshfb_probe_f_spv/4,RES_FS);
    CHECK(ts && ms && fs && gfx_shader_write_inventory_complete(ts) && gfx_shader_write_inventory_complete(ms),
          "own actual transformed pointer writer and reader shaders");
    const MeshDraw draw={make_prog(ms,ts,fs),10,0,{8,5},{1,1,1,1}};
    pool_t device={.magic=NVN_POOL_MAGIC,.flags=0x189,.mem=(uint8_t *)(uintptr_t)RT_GPU_VA_BASE,
                   .game_mem=guest+0xa00000,.size=0x30000};
    CHECK(b->pool_init(&device,&device),"create device pointer table and two device color sources");
    const uint64_t table=(uintptr_t)device.mem, target[2]={table+0x10000,table+0x20000};
    const uint32_t expected[2]={0xff0000ff,0xff00ff00};
    uint8_t *stage=pool->mem+0xb0000;
    uint64_t address=(uintptr_t)stage;
    uint32_t colors[2][4]={{expected[0],0,0,0},{expected[1],0,0,0}};
    memcpy(stage+0x10,colors,sizeof colors);
    b->guest_written(address+0x10,sizeof colors);
    static uint32_t picture[TEX_W*TEX_H];
    const uint32_t before=vk_exec_probe_mfb_pointer_phases();
    uint32_t rendered=0;
    for (unsigned i=0;i<2;++i) {
        uint64_t seed[2]={target[1-i],0};
        memcpy(stage,seed,sizeof seed);
        b->guest_written(address,sizeof seed);
        const MeshPointerProbe pointer={.table=table,.selected=target[i],.verify=address+0x40,
            .seed={{address,table,sizeof seed},{address+0x10,target[0],16},{address+0x20,target[1],16}}};
        const MeshStorageProbe inputs={.pointer=&pointer};
        if (draw_picture(b,pool,color,&draw,1,picture,0,&inputs)) return 1;
        uint64_t actual;
        memcpy(&actual,stage+0x40,sizeof actual);
        CHECK(actual==target[i],"round %u transformed task wrote device table %#llx expected %#llx",i,
              (unsigned long long)actual,(unsigned long long)target[i]);
        CHECK(vk_exec_probe_mfb_pointer_phases()==before+i+1,"round %u used one actual task-prefix pointer readback",i);
        unsigned pixels=0;
        for (unsigned j=0;j<TEX_W*TEX_H;++j) if (picture[j]!=picture[0]) {
            CHECK(picture[j]==expected[i],"round %u transformed mesh pixel %u is %08x, expected %08x",i,j,picture[j],expected[i]);
            ++pixels;
        }
        CHECK(pixels>TEX_W*TEX_H/20,"round %u drew only %u pixels",i,pixels);
        rendered+=pixels;
    }
    printf("probe-meshfb: two actual task-written device pointers selected %u exact mesh pixels through two completed private phases\n",rendered);
    b->pool_fini(&device);
    return 0;
}


static int draw_part(const GfxBackend *b, uint8_t *guest) {
    const int storage_only = rt_knob("PROBE_MFB_STORAGE_PHASES", "0") == 1;
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x100000,
                   .size = storage_only ? 0x200000 : 0xC0000, .game_mem = guest + 0x100000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool");
    VkCtx *c = vk_ctx();
    CHECK(c, "no Vulkan device");
    static tex_t color;
    memset(&color, 0, sizeof color);
    color.magic = NVN_TEX_MAGIC;
    color.d.target = NVN_TEXTURE_TARGET_2D, color.d.format = NVN_FORMAT_RGBA8, color.d.width = TEX_W, color.d.height = TEX_H;
    color.d.depth = 1, color.d.levels = 1;
    nvn_tex_layout(&color.d, &color.L);
    CHECK(color.L.size <= 0x80000, "the target exceeds its pool slot");
    color.pool = &pool, color.pool_flags = (uint16_t)pool.flags, color.off = 0, color.mem = pool.mem, color.serial = 4000;
    if (storage_only || rt_knob("PROBE_MFB_TASK_POINTER", "0") == 1) {
        int saved = c->mesh_fallback;
        c->mesh_fallback = 1;
        int failed = storage_only ? storage_draw_part(b, &pool, &color) : task_pointer_draw_part(b, guest, &pool, &color);
        c->mesh_fallback = saved;
        b->texture_release(&color);
        b->pool_fini(&pool);
        return failed;
    }

    const GfxShader *fs = gfx_shader_own(meshfb_probe_f_spv, sizeof meshfb_probe_f_spv / 4, RES_FS);
    const GfxShader *ms = gfx_shader_own(meshfb_probe_m_spv, sizeof meshfb_probe_m_spv / 4, RES_MS);
    const GfxShader *ts = gfx_shader_own(meshfb_probe_t_spv, sizeof meshfb_probe_t_spv / 4, RES_TS);
    const GfxShader *tm = gfx_shader_own(meshfb_probe_tm_spv, sizeof meshfb_probe_tm_spv / 4, RES_MS);
    CHECK(fs && ms && ts && tm, "a probe shader is malformed");
    prog_t *plain = make_prog(ms, NULL, fs), *tasked = make_prog(tm, ts, fs);

    // 24 cells of a 6 x 4 grid by 24 mesh workgroups; 40 cells of an 8 x 5 grid by 10 task workgroups that start 0 to
    // 4 mesh workgroups each
    const MeshDraw cases[][2] = {
        {{plain, 24, 0, {6, 4}, {1, 1, 1, 1}}},
        {{plain, 24, 1, {6, 4}, {1, 1, 1, 1}}},
        {{tasked, 10, 0, {8, 5}, {1, 1, 1, 1}}},
        {{tasked, 10, 1, {8, 5}, {1, 1, 1, 1}}},
        {{plain, 24, 1, {6, 4}, {1, 1, 1, 1}}, {tasked, 10, 1, {8, 5}, {0.5f, 1, 0.25f, 1}}},
        {{tasked, 10, 0, {8, 5}, {1, 1, 1, 1}}, {plain, 24, 0, {6, 4}, {1, 0.5f, 0.5f, 1}}},
    };
    static const char *const names[] = {"mesh, direct", "mesh, indirect", "task + mesh, direct", "task + mesh, indirect",
                                        "mesh then task + mesh in one pass, indirect", "task + mesh then mesh in one pass, direct"};
    enum { CASES = sizeof cases / sizeof cases[0] };
    static uint32_t px[2][CASES][TEX_W * TEX_H];
    int native = c->has_mesh_shader, was = c->mesh_fallback;
    // Submit the fallback picture before any other mesh draw can warm its conversion or pipelines.
    static uint32_t cold_px[TEX_W * TEX_H];
    vk_pipeline_async_enable();
    c->mesh_fallback = 1;
    int cold_failed = draw_picture(b, &pool, &color, cases[5], 2, cold_px, 0, NULL);
    c->mesh_fallback = was;
    if (cold_failed) return 1;
    for (int way = native ? 0 : 1; way < 2; way++) {
        c->mesh_fallback = way;
        for (int k = 0; k < CASES; k++)
            if (draw_picture(b, &pool, &color, cases[k], cases[k][1].prog ? 2 : 1, px[way][k], 0, NULL)) return 1;
    }
    c->mesh_fallback = was;
    for (int k = 0; k < CASES; k++) {
        int fb = drawn_texels(px[1][k]);
        CHECK(fb > TEX_W * TEX_H / 20, "%s: the fallback drew %d texels", names[k], fb);
        if (!native) {
            printf("probe-meshfb: %s: the fallback drew %d of %d texels (no mesh shaders here: not compared)\n", names[k], fb,
                   TEX_W * TEX_H);
            continue;
        }
        int worst, diff = differing(px[0][k], px[1][k], &worst), nat = drawn_texels(px[0][k]);
        CHECK(nat > TEX_W * TEX_H / 20, "%s: the mesh pipeline drew %d texels", names[k], nat);
        CHECK(worst <= 1, "%s: %d texels differ between the mesh pipeline and the fallback, by up to %d of 255 (the mesh "
              "pipeline drew %d texels, the fallback %d)", names[k], diff, worst, nat, fb);
        printf("probe-meshfb: %s: %d texels drawn by the mesh pipeline, %d by the fallback, %d differ (by %d of 255 at most)\n",
               names[k], nat, fb, diff, worst);
    }
    int cold_worst;
    CHECK(!differing(cold_px, px[1][5], &cold_worst),
          "the first fallback submission differs from the warmed task + mesh picture (by %d of 255 at most)", cold_worst);
    printf("probe-meshfb: the first fallback submission with background builds matched the warmed task + mesh picture\n");
    int w01, w23;
    CHECK(!differing(px[1][0], px[1][1], &w01) && !differing(px[1][2], px[1][3], &w23),
          "the fallback's direct and indirect pictures differ");
    CHECK(!vk_meshfb_short(), "%llu of the draws above did not fit the fallback's buffers", (unsigned long long)vk_meshfb_short());
    // More workgroups than the mesh buffer has slots for: what fits is drawn, the rest is dropped and counted. The
    // cells past the grid are outside the picture, so it is the 24-cell picture again.
    uint32_t slots = (uint32_t)(vk_meshfb_mesh_words() / 540), over = slots + 1000;
    const MeshDraw big = {plain, over, 1, {6, 4}, {1, 1, 1, 1}};
    static uint32_t bpx[TEX_W * TEX_H];
    c->mesh_fallback = 1;
    int bad = draw_picture(b, &pool, &color, &big, 1, bpx, 1, NULL);
    c->mesh_fallback = was;
    if (bad) return 1;
    b->finish();
    int wb, db = differing(bpx, px[1][0], &wb);
    CHECK(!db, "a draw of %u workgroups, %u more than the mesh buffer holds: %d texels differ from the 24-cell picture", over,
          over - slots, db);
    CHECK(vk_meshfb_short() == 1, "a draw of %u workgroups with room for %u was counted as %llu draws that did not fit", over, slots,
          (unsigned long long)vk_meshfb_short());
    printf("probe-meshfb: a draw of %u workgroups with room for %u: the picture of the first 24 is whole, the draw was counted\n",
           over, slots);
    c->mesh_fallback = 1;
    bad = draw_picture(b, &pool, &color, &big, 1, NULL, 1, NULL);
    c->mesh_fallback = was;
    b->texture_release(&color);
    b->pool_fini(&pool);
    return bad;
}

int probe_meshfb(void) {
    const char *dir = cfg_get("NX2NAVI_PROBE_MESHFB_DIR");
    if (dir && *dir) return modules_part(dir);
    vk_hdr_off();
    const GfxBackend *b = &gfx_vulkan_backend;
    uint64_t heap = 0;
    CHECK(mem_set_heap_size(16ull << 20, &heap) == 0 && heap, "SetHeapSize 16 MiB");
    int bad = draw_part(b, (uint8_t *)(uintptr_t)heap);
    meshfb_finish_probe(b);
    CHECK(!vk_meshfb_mesh_words() && !vk_meshfb_buffer(MESHFB_CB_MESH).buffer && !vk_meshfb_probe_pending_reports(),
          "backend shutdown left mesh fallback allocations or pending reports");
    printf("probe-meshfb: checked backend shutdown retired the mesh fallback owner\n");
    long errors = vk_validation_errors_total();
    if (bad || errors) {
        printf("probe-meshfb: FAILED (%s, %ld validation errors)\n", bad ? "a draw check failed" : "checks passed", errors);
        return 1;
    }
    printf("probe-meshfb: passed (0 validation errors)\n");
    return 0;
}
