// nx2navi.exe --probe-vkexec: the Vulkan replay, through gfx_vulkan_backend. Sync part: empty
// submissions get consecutive fence values; fence_wait and fence_done agree; completion callbacks run in submit order on
// the completion thread; a guest-memory word a callback writes is visible once fence_wait returns; finish drains every
// submission; queued presents of the same cleared vk_texture.c image (layout GENERAL) read back exactly before and
// after a swapchain resize. Clear part: a hand-built op stream (SET_RT to an RGBA8 and a D32 texture, whole-target clears,
// then scissored clears) through submit reads back exactly with read_texels. Pass marks part (test_marks): the GPU's
// buffer marks and the ring records name the last pass of a submission, and the device-lost report text says so.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "gfx_wsys.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"
#include "hdr_fullscreen_spv.h"
#include "sample_restore_probe_spv.h"
#include "fallback_probe_spv.h"
#include "storage_first_write_probe_spv.h"
#include "pool_restore_probe_spv.h"
#include "pool_finite_probe_spv.h"
#include "pool_dispatch_probe_spv.h"
#include "device_pointer_draw_spv.h"

#define SUBMITS 100
#define TAIL 10
#define TEX_W 256
#define TEX_H 144
#define CHECK(cond, ...)                                                                                                \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("probe-vkexec: FAILED: ");                                                                          \
            printf(__VA_ARGS__);                                                                                       \
            printf("\n");                                                                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static uint32_t main_tid;
static uint64_t ran[SUBMITS + TAIL + 8];
static long nran, on_main;

static void note(void *arg) {   // completion thread
    if (nran < (long)(sizeof ran / sizeof ran[0])) ran[nran] = (uint64_t)(uintptr_t)arg;
    nran++;
    if (plat_thread_id() == main_tid) on_main++;
}

typedef struct { volatile uint32_t *word; uint32_t value; } WordWrite;
static void write_word(void *arg) {
    const WordWrite *w = arg;
    *w->word = w->value;
}

static long ran_now(void) { return __atomic_load_n(&nran, __ATOMIC_ACQUIRE); }

static int test_sync(const GfxBackend *b, uint8_t *guest, uint64_t *first, uint64_t *last) {
    // An empty recording as the front end hands it over: {OP_BEGIN, OP_END}.
    static op_hdr empty[2] = {{OP_BEGIN, sizeof(op_hdr)}, {OP_END, sizeof(op_hdr)}};
    uint64_t handles[2] = {(uint64_t)(uintptr_t)empty, (uint64_t)(uintptr_t)empty};
    uint64_t prev = b->fence_done();
    *first = prev + 1;
    for (int i = 0; i < SUBMITS; i++) {
        vk_completion_add(prev + 1, note, (void *)(uintptr_t)(prev + 1));   // queued before the value exists, as vk_exec.c does
        uint64_t f = b->submit(i % 3 ? handles : NULL, i % 3);
        CHECK(f == prev + 1, "submission %d got fence %llu after %llu", i, (unsigned long long)f, (unsigned long long)prev);
        CHECK(b->fence_done() <= f, "fence_done %llu ahead of the newest submission %llu",
              (unsigned long long)b->fence_done(), (unsigned long long)f);
        CHECK(!b->fence_wait(f + 1, 0), "fence_wait of a value never submitted returned 1");
        if (i % 10 == 9) {
            CHECK(b->fence_wait(f, 5000000000ull), "fence_wait(%llu) timed out", (unsigned long long)f);
            CHECK(b->fence_done() >= f, "fence_done %llu after fence_wait(%llu)", (unsigned long long)b->fence_done(),
                  (unsigned long long)f);
            CHECK(ran_now() == i + 1, "%ld callbacks ran after fence_wait of submission %d", ran_now(), i);
        }
        prev = f;
    }
    // A guest-memory word written by a completion callback is visible to the waiter.
    volatile uint32_t *word = (volatile uint32_t *)(guest + 0x1000);
    *word = 0;
    static WordWrite ww;
    ww = (WordWrite){word, 0x5EC0DE01u};
    vk_completion_add(prev + 1, write_word, &ww);
    uint64_t f = b->submit(NULL, 0);
    CHECK(f == prev + 1, "word submission got fence %llu", (unsigned long long)f);
    CHECK(b->fence_wait(f, 5000000000ull), "fence_wait of the word submission timed out");
    CHECK(*word == 0x5EC0DE01u, "guest word is %#x after fence_wait", *word);
    prev = f;
    // finish drains everything.
    for (int i = 0; i < TAIL; i++) {
        vk_completion_add(prev + 1, note, (void *)(uintptr_t)(prev + 1));
        prev = b->submit(handles, 2);
    }
    b->finish();
    CHECK(b->fence_done() == prev, "fence_done %llu after finish, newest %llu", (unsigned long long)b->fence_done(),
          (unsigned long long)prev);
    CHECK(ran_now() == SUBMITS + TAIL, "%ld of %d callbacks ran after finish", ran_now(), SUBMITS + TAIL);
    for (int i = 0; i < SUBMITS + TAIL; i++) {
        uint64_t want = *first + (uint64_t)i + (i >= SUBMITS);   // the word submission sits between the two runs
        CHECK(ran[i] == want, "callback %d ran for fence %llu, expected %llu", i, (unsigned long long)ran[i],
              (unsigned long long)want);
    }
    CHECK(!on_main, "%ld callbacks ran on the submitting thread", on_main);
    *last = prev;
    return 0;
}

static uint8_t to_byte(float f) { return (uint8_t)(f * 255.0f + 0.5f); }

static int test_present_frame(const GfxBackend *b, const tex_t *t, VkImage *source, const float rgba[4],
                              int *w_out, int *h_out) {
    const VkTexImage *img = vk_texture_get_write(t);
    CHECK(img && img->format == VK_FORMAT_R8G8B8A8_UNORM, "vk_texture_get of an RGBA8 texture");
    CHECK(img->image && (!*source || *source == img->image), "resize replaced the probe's native source image");
    *source = img->image;
    vk_texture_gpu_written(t);   // the clear below is its content, not the storage
    VkClearColorValue cv;
    memcpy(cv.float32, rgba, sizeof cv.float32);
    VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkCommandBuffer cb = vk_upload_begin();
    vkCmdClearColorImage(cb, *source, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
    vk_upload_end();

    uint64_t frames = gfx_stats().frames;
    vk_present_capture(0);
    vk_present_capture(1);
    const int crop[4] = {0, 0, TEX_W, TEX_H};
    b->present(t, crop, 0);
    b->finish();   // the queued present is submitted and, with capture on, read back
    const uint8_t *px;
    int w, h;
    VkFormat fmt;
    CHECK(vk_present_captured(&px, &w, &h, &fmt), "the present was not captured");
    CHECK(gfx_stats().frames == frames + 1, "frame counter %llu after one present from %llu",
          (unsigned long long)gfx_stats().frames, (unsigned long long)frames);
    uint8_t want[4] = {0, 0, 0, 255};
    if (fmt == VK_FORMAT_B8G8R8A8_UNORM) want[0] = to_byte(rgba[2]), want[1] = to_byte(rgba[1]), want[2] = to_byte(rgba[0]);
    else if (fmt == VK_FORMAT_R8G8B8A8_UNORM) want[0] = to_byte(rgba[0]), want[1] = to_byte(rgba[1]), want[2] = to_byte(rgba[2]);
    else CHECK(0, "swapchain format %d is not UNORM RGBA/BGRA", (int)fmt);
    uint32_t wv, black = 0xff000000u;   // opaque black in both byte orders
    memcpy(&wv, want, 4);
    int r[4];   // the picture's rectangle; black bars around it when the window's shape differs (gfx_present.c)
    gfx_fit_rect(TEX_W, TEX_H, w, h, r);
    const uint32_t *p = (const uint32_t *)px;
    for (long k = 0, n = (long)w * h; k < n; k++) {
        int x = (int)(k % w), y = (int)(k / w);
        uint32_t e = x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3] ? wv : black;
        CHECK(p[k] == e, "presented pixel %ld (%d, %d) is %08x, expected %08x", k, x, y, p[k], e);
    }
    *w_out = w, *h_out = h;
    return 0;
}

static int test_present(const GfxBackend *b, uint8_t *guest, int *w_out, int *h_out) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x100000, .size = 0x100000,
                   .game_mem = guest + 0x100000};
    tex_t t = {.magic = NVN_TEX_MAGIC, .pool = &pool, .mem = pool.mem, .serial = 1};
    t.d.target = NVN_TEXTURE_TARGET_2D, t.d.format = NVN_FORMAT_RGBA8, t.d.width = TEX_W, t.d.height = TEX_H;
    t.d.depth = 1, t.d.levels = 1;
    nvn_tex_layout(&t.d, &t.L);
    CHECK(t.L.size <= pool.size, "texture storage %#llx exceeds the pool", (unsigned long long)t.L.size);
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool");
    t.mem = pool.mem;

    // FSR's approximate reciprocals can move a flat colour by 1/255.
    cfg_setenv("NX2NAVI_UPSCALER", "bilinear");
    VkImage source = VK_NULL_HANDLE;
    int w = 0, h = 0;
    int bad = test_present_frame(b, &t, &source, (const float[]){0.2f, 0.4f, 0.6f, 1.0f}, &w, &h);
    int resized_w = 0, resized_h = 0, client_w = 0, client_h = 0;
    if (!bad) {
        int target_w = w == 1000 && h == 680 ? 900 : 1000;
        gfx_window_resize(target_w, 680);
        wsys_client_size(&client_w, &client_h);
        bad = test_present_frame(b, &t, &source, (const float[]){0.8f, 0.2f, 0.4f, 1.0f}, &resized_w, &resized_h);
    }
    vk_present_capture(0);
    b->texture_release(&t);
    b->pool_fini(&pool);
    if (bad) return bad;
    CHECK((resized_w != w || resized_h != h) && resized_w == client_w && resized_h == client_h,
          "swapchain resize produced %dx%d from %dx%d (client %dx%d)", resized_w, resized_h, w, h, client_w, client_h);
    *w_out = resized_w, *h_out = resized_h;
    printf("probe-vkexec: present %dx%d and same-source resized present %dx%d exact\n", w, h, resized_w, resized_h);
    return 0;
}

// ---- clear part: a hand-built op stream through submit, read back with read_texels.

static uint8_t *put(uint8_t *p, uint32_t op, const void *payload, uint32_t n) {
    op_hdr h = {op, (uint32_t)((sizeof h + n + 7) & ~7u)};
    memset(p, 0, h.size);
    memcpy(p, &h, sizeof h);
    if (n) memcpy(p + sizeof h, payload, n);
    return p + h.size;
}

static void init_tex(tex_t *t, pool_t *pool, int64_t off, int format) {
    memset(t, 0, sizeof *t);
    t->magic = NVN_TEX_MAGIC;
    t->d.target = NVN_TEXTURE_TARGET_2D, t->d.format = format, t->d.width = TEX_W, t->d.height = TEX_H;
    t->d.depth = 1, t->d.levels = 1;
    nvn_tex_layout(&t->d, &t->L);
    t->pool = pool, t->pool_flags = (uint16_t)pool->flags, t->off = off, t->mem = pool->mem + off, t->serial = 100 + (uint64_t)off;
}

// Pixel of the RGBA8 target after the stream: the scissored second clear inside the rectangle, the first elsewhere.
#define SC_X 40
#define SC_Y 24
#define SC_W 64
#define SC_H 32

static int test_clear(const GfxBackend *b, uint8_t *guest, int *n_inside) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x400000, .size = 0x200000,
                   .game_mem = guest + 0x400000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool for the clear part");
    static tex_t color, depth;   // payload texture pointers live as long as the submission
    init_tex(&color, &pool, 0, NVN_FORMAT_RGBA8);
    init_tex(&depth, &pool, 0x100000, NVN_FORMAT_DEPTH32F);
    CHECK(color.L.size <= 0x100000 && depth.L.size <= 0x100000, "clear-part textures exceed their pool halves");

    static _Alignas(8) uint8_t stream[4096];
    uint8_t *p = stream;
    p = put(p, OP_BEGIN, NULL, 0);
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1, rt.color[0] = &color, rt.depth = &depth;
    p = put(p, OP_SET_RT, &rt, sizeof rt);
    const p_clear_color c0 = {0, 0xf, {0.2f, 0.4f, 0.6f, 1.0f}};
    p = put(p, OP_CLEAR_COLOR, &c0, sizeof c0);   // whole target outside a pass: vkCmdClearColorImage
    const p_clear_ds d0 = {0.5f, 1, 0, 0xff};
    p = put(p, OP_CLEAR_DS, &d0, sizeof d0);      // vkCmdClearDepthStencilImage
    const p_scissor sc = {SC_X, SC_Y, SC_W, SC_H};
    p = put(p, OP_SCISSOR, &sc, sizeof sc);
    const p_clear_color c1 = {0, 0xf, {1.0f, 0.0f, 0.0f, 1.0f}};
    p = put(p, OP_CLEAR_COLOR, &c1, sizeof c1);   // scissored: a pass and vkCmdClearAttachments
    const p_clear_ds d1 = {0.25f, 1, 0, 0xff};
    p = put(p, OP_CLEAR_DS, &d1, sizeof d1);
    const p_scissor full = {0, 0, 1 << 20, 1 << 20};
    p = put(p, OP_SCISSOR, &full, sizeof full);   // leaves the replay's scissor unbounded for later submissions
    p = put(p, OP_END, NULL, 0);
    CHECK(p <= stream + sizeof stream, "op stream overflow");

    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    CHECK(b->fence_wait(f, 5000000000ull), "fence_wait of the clear submission timed out");

    static uint32_t px[TEX_W * TEX_H];
    static float dpx[TEX_W * TEX_H];
    NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
    b->read_texels(&color, NULL, &r, (uint8_t *)px, TEX_W * 4, TEX_W * TEX_H * 4);
    b->read_texels(&depth, NULL, &r, (uint8_t *)dpx, TEX_W * 4, TEX_W * TEX_H * 4);
    const uint8_t out_b[4] = {to_byte(0.2f), to_byte(0.4f), to_byte(0.6f), 255}, in_b[4] = {255, 0, 0, 255};
    uint32_t out_v, in_v;
    memcpy(&out_v, out_b, 4), memcpy(&in_v, in_b, 4);
    int inside = 0;
    for (int y = 0; y < TEX_H; y++)
        for (int x = 0; x < TEX_W; x++) {
            int in = x >= SC_X && x < SC_X + SC_W && y >= SC_Y && y < SC_Y + SC_H;
            inside += in;
            uint32_t want = in ? in_v : out_v;
            float dwant = in ? 0.25f : 0.5f;
            CHECK(px[y * TEX_W + x] == want, "colour texel (%d,%d) is %08x, expected %08x", x, y, px[y * TEX_W + x], want);
            CHECK(dpx[y * TEX_W + x] == dwant, "depth texel (%d,%d) is %g, expected %g", x, y, dpx[y * TEX_W + x], dwant);
        }
    b->texture_release(&color);
    b->texture_release(&depth);
    b->pool_fini(&pool);
    *n_inside = inside;
    return 0;
}

// ---- draw part: the pipeline self-test, then a hand-built stream that clears an RGBA8 target, binds a census
// vertex + fragment pair and draws one triangle; a census compute shader is dispatched. The census (probe_census_dir,
// tools/shader_census.py) is read the way vk_pipeline_selftest reads it: a pair is a fragment shader and a vertex
// shader of the same source file.

void vk_exec_draw_shutdown(void);   // vk_exec_draw.c: the stand-in resources; the device must be idle
void vk_exec_xfer_shutdown(void);   // vk_exec_xfer.c: the occlusion query pool; the device must be idle

typedef struct { const char *key, *c1, *src; size_t c1_len, src_len; int stage; uint32_t local[3]; } DrawEnt;

static const char *dr_field(const char *from, const char *to, const char *name, size_t *len) {
    char pat[32];
    snprintf(pat, sizeof pat, "\"%s\": \"", name);
    const char *p = strstr(from, pat);
    if (!p || p >= to) return NULL;
    p += strlen(pat);
    const char *e = memchr(p, '"', (size_t)(to - p));
    if (!e) return NULL;
    *len = (size_t)(e - p);
    return p;
}

static int dr_cmp(const void *x, const void *y) {
    const DrawEnt *a = x, *b = y;
    size_t n = a->src_len < b->src_len ? a->src_len : b->src_len;
    int c = memcmp(a->src, b->src, n);
    if (c) return c;
    if (a->src_len != b->src_len) return a->src_len < b->src_len ? -1 : 1;
    return a->stage - b->stage;
}

static int dr_parse(const char *jobs, DrawEnt *out, int max) {
    int n = 0;
    for (const char *p = jobs; n < max && (p = strstr(p, "\"job\": {")); p += 8) {
        const char *end = strstr(p, "\"origin\": ");
        if (!end) break;
        size_t kl, sl, cl, rl;
        const char *k = dr_field(p, end, "key", &kl), *s = dr_field(p, end, "stage", &sl), *c = dr_field(p, end, "c1", &cl),
                   *r = dr_field(p, end, "source", &rl);
        if (!k || !s || !c || !r || kl != 64) continue;
        DrawEnt *e = &out[n++];
        e->key = k, e->c1 = c, e->c1_len = cl / 2, e->src = r, e->src_len = rl;
        e->stage = !strncmp(s, "vs", sl) ? GFX_STAGE_VS : !strncmp(s, "ps", sl) ? GFX_STAGE_FS : GFX_STAGE_CS;
        e->local[0] = e->local[1] = e->local[2] = 1;
        const char *l = strstr(p, "\"local\": [");
        if (l && l < end) {
            l += 10;
            for (int i = 0; i < 3; i++) {
                while (*l == ' ' || *l == '\n' || *l == '\r' || *l == ',') l++;
                e->local[i] = (uint32_t)strtoul(l, (char **)&l, 10);
            }
        }
    }
    return n;
}

static char *dr_read(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) free(b), b = NULL;
    fclose(f);
    if (b) b[n] = 0, *size = n;
    return b;
}

static int dr_hex(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

// A program of one census shader as the game hands it to NVN: control section, code with its 0x30-byte header.
static prog_t *dr_prog(const char *root, const DrawEnt *e[], int n) {
    static uint64_t serial = 0x7000;
    prog_t *p = calloc(1, sizeof *p);
    p->magic = NVN_PROG_MAGIC, p->serial = ++serial;
    for (int i = 0; i < n; i++) {
        char path[PLAT_PATH_MAX * 2 + 128];
        snprintf(path, sizeof path, "%s/code/%.64s.code", root, e[i]->key);
        long body;
        char *raw = dr_read(path, &body);
        if (!raw) return free(p), NULL;
        size_t skip = e[i]->stage == GFX_STAGE_CS ? 0 : 0x30, total = skip + (size_t)body;
        uint8_t *code = calloc(1, total), *ctl = calloc(1, 0x880);
        memcpy(code + skip, raw, (size_t)body);
        free(raw);
        uint32_t c1_off = (uint32_t)(total - e[i]->c1_len), length = (uint32_t)total;
        uint32_t nvn_stage = e[i]->stage == GFX_STAGE_VS ? 0 : e[i]->stage == GFX_STAGE_FS ? 1 : 5;
        memcpy(ctl + 0x700, &c1_off, 4), memcpy(ctl + 0x704, &length, 4), memcpy(ctl + 0x714, &nvn_stage, 4);
        if (nvn_stage == 5) memcpy(ctl + 0x748, e[i]->local, 12);
        p->sd[i].data = (uint64_t)(uintptr_t)code, p->sd[i].control = ctl;
        p->nstage++;
    }
    return p;
}

#define DR_ATTRS 8
#define DR_STRIDE (DR_ATTRS * 16)

static int layouts_equal;   // texels compared equal between the two layouts (0: not compared)
static int ind_draw_texels, ind_dispatches;   // indirect draws' texels compared, indirect dispatches run

// The draw part's stream up to the draw: target cleared, viewport, program, three RGBA32F vertices at 0x80000 of the
// pool, identity uniforms at 0x90000.
static uint8_t *draw_prefix(uint8_t *q, prog_t *prog, const pool_t *pool, tex_t *color) {
    q = put(q, OP_BEGIN, NULL, 0);
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
    const p_bind_program bp = {prog, 3};
    q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
    p_bind_vattr va;
    memset(&va, 0, sizeof va);
    va.n = DR_ATTRS;
    for (int a = 0; a < DR_ATTRS; a++) {
        va_st s = {NVN_FORMAT_RGBA32F, (uint32_t)(16 * a), 0};
        memcpy(&va.st[a], &s, sizeof s);
    }
    q = put(q, OP_BIND_VATTR, &va, sizeof va);
    p_bind_vstream vs;
    memset(&vs, 0, sizeof vs);
    vs.n = 1, vs.st[0].stride = DR_STRIDE;
    q = put(q, OP_BIND_VSTREAM, &vs, sizeof vs);
    p_bind_vbuf vb;
    memset(&vb, 0, sizeof vb);
    vb.n = 1, vb.r[0].address = (uint64_t)(uintptr_t)pool->mem + 0x80000, vb.r[0].size = 3 * DR_STRIDE;
    q = put(q, OP_BIND_VBUF, &vb, sizeof vb);
    for (int stage = 0; stage < 2; stage++)
        for (int i = 0; i < 4; i++) {
            p_bind_buf bu = {stage, i, (uint64_t)(uintptr_t)pool->mem + 0x90000, 4096};
            q = put(q, OP_BIND_UBO, &bu, sizeof bu);
        }
    return q;
}

// ---- pass marks part (vk_crumbs.c): one submission with a pass of 2 draws into the draw part's target, a dispatch and
// a pass of 3 draws into a second target. Once its fence is done, the buffer marks must name the last pass as both
// started and finished, the ring records must hold what the three were, and vk_pass_marks_format (the text
// vk_device.c device_fault_report logs) must name it. Without VK_AMD_buffer_marker or with NX2NAVI_PASS_CRUMBS=0 it is skipped:
// NVIDIA's checkpoints can only be read once the device is lost.
static char marks_text[2048];
static int marks_skipped;

static int test_marks(const GfxBackend *b, pool_t *pool, tex_t *color, prog_t *gfx, prog_t *cs) {
    uint32_t s0, f0;
    if (!vk_pass_marks_read(&s0, &f0)) {
        marks_skipped = 1;
        return 0;
    }
    static tex_t second;
    init_tex(&second, pool, 0x40000, NVN_FORMAT_RGBA8);
    CHECK(0x40000 + second.L.size <= 0x80000, "pass-marks target exceeds its pool slot");
    static _Alignas(8) uint8_t stream[8192];
    uint8_t *q = draw_prefix(stream, gfx, pool, color);
    const p_draw d = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
    for (int i = 0; i < 2; i++) q = put(q, OP_DRAW, &d, sizeof d);
    const p_bind_program bc = {cs, 1 << 5};
    q = put(q, OP_BIND_PROGRAM, &bc, sizeof bc);
    const p_dispatch dp = {2, 1, 1, 0};
    q = put(q, OP_DISPATCH, &dp, sizeof dp);
    const p_bind_program bg = {gfx, 3};
    q = put(q, OP_BIND_PROGRAM, &bg, sizeof bg);
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1, rt.color[0] = &second;
    q = put(q, OP_SET_RT, &rt, sizeof rt);
    for (int i = 0; i < 3; i++) q = put(q, OP_DRAW, &d, sizeof d);
    q = put(q, OP_END, NULL, 0);
    CHECK(q <= stream + sizeof stream, "pass-marks op stream overflow");
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the pass-marks submission timed out");
    uint32_t started = 0, finished = 0;
    vk_pass_marks_read(&started, &finished);
    CHECK(started && started == finished, "after the fence the GPU last started mark %u and finished mark %u (before "
          "the submission: %u and %u)", started, finished, s0, f0);
    const VkPassMark *p1 = vk_pass_mark_record(finished), *dm = vk_pass_mark_record(finished - 1),
                     *p0 = vk_pass_mark_record(finished - 2);
    CHECK(p1 && p1->kind == VK_MARK_PASS && p1->fence == f && p1->step == 1,
          "the last finished mark %u is not pass 1 of submission %llu", finished, (unsigned long long)f);
    CHECK(p1->w == TEX_W && p1->h == TEX_H && p1->d == 1 && p1->ncolor == 1 &&
              p1->color[0] == VK_FORMAT_R8G8B8A8_UNORM && p1->depth == VK_FORMAT_UNDEFINED,
          "pass 1 has %ux%u, %u layers, %u colour targets, format %d, depth %d", p1->w, p1->h, p1->d, p1->ncolor,
          (int)p1->color[0], (int)p1->depth);
    CHECK(p1->draws == 3 && p1->first[0] && p1->first[1] && p1->first[0] == p1->last[0] && p1->first[1] == p1->last[1],
          "pass 1 has %u draws (3 expected) and shaders %p %p ... %p %p", p1->draws, (const void *)p1->first[0],
          (const void *)p1->first[1], (const void *)p1->last[0], (const void *)p1->last[1]);
    CHECK(dm && dm->kind == VK_MARK_DISPATCH && dm->fence == f && dm->step == 0 && dm->w == 2 && dm->h == 1 &&
              dm->d == 1 && dm->first[0],
          "the mark before pass 1 is not dispatch 0 of 2x1x1 groups of submission %llu", (unsigned long long)f);
    CHECK(p0 && p0->kind == VK_MARK_PASS && p0->fence == f && p0->step == 0 && p0->draws == 2 &&
              p0->first[0] == p1->first[0],
          "the mark before the dispatch is not pass 0 of submission %llu with 2 draws", (unsigned long long)f);
    vk_pass_marks_format("VK_AMD_buffer_marker", started, finished, marks_text, sizeof marks_text);
    char want[128];
    snprintf(want, sizeof want, "last finished %u: pass 1 of submission %llu: %ux%u, 1 layer, colour formats %d, ",
             finished, (unsigned long long)f, TEX_W, TEX_H, (int)VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(strstr(marks_text, want) && strstr(marks_text, ", 3 draws; first vs "),
          "the report text lacks \"%s\" or the draw count:\n%s", want, marks_text);
    b->texture_release(&second);
    return 0;
}

static int test_draw(const GfxBackend *b, uint8_t *guest, int *n_drawn, int *n_tries, int *n_cs) {
    char root[PLAT_PATH_MAX * 2], path[PLAT_PATH_MAX * 2 + 64];
    CHECK(probe_census_dir(root, sizeof root), "no build/v%d/shaders/jobs.json above the exe (run tools/shader_census.py)",
          rt_game_version);
    snprintf(path, sizeof path, "%s/jobs.json", root);
    long size;
    char *jobs = dr_read(path, &size);
    DrawEnt *e = malloc(sizeof *e * 20000);
    int n = jobs && e ? dr_parse(jobs, e, 20000) : 0;
    CHECK(n > 0, "no census entries");
    qsort(e, (size_t)n, sizeof *e, dr_cmp);
    int (*pairs)[2] = malloc(sizeof *pairs * (size_t)n), np = 0, *cs = malloc(sizeof *cs * (size_t)n), nc = 0;
    for (int i = 0; i < n;) {
        int j = i, v0 = -1, nv = 0, nf = 0;
        for (; j < n && e[j].src_len == e[i].src_len && !memcmp(e[j].src, e[i].src, e[i].src_len); j++) {
            if (e[j].stage == GFX_STAGE_VS && v0 < 0) v0 = j;
            nv += e[j].stage == GFX_STAGE_VS;
            if (e[j].stage == GFX_STAGE_CS) cs[nc++] = j;
        }
        for (int k = i; k < j && nv; k++)
            if (e[k].stage == GFX_STAGE_FS) pairs[np][0] = v0 + nf++ % nv, pairs[np][1] = k, np++;
        i = j;
    }
    CHECK(np > 0 && nc > 0, "census has %d pairs and %d compute shaders", np, nc);

    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x700000, .size = 0xC0000,
                   .game_mem = guest + 0x700000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool for the draw part");
    static tex_t color;
    init_tex(&color, &pool, 0, NVN_FORMAT_RGBA8);
    CHECK(color.L.size <= 0x80000, "draw-part target exceeds its pool slot");
    // Vertex data at 0x80000: three vertices of DR_ATTRS RGBA32F attributes, attribute 0 the position. Uniform data at
    // 0x90000: identity matrices back to back, so that a vertex shader multiplying the position by one passes it through.
    float *vtx = (float *)(pool.mem + 0x80000);
    static const float pos[3][4] = {{-0.8f, -0.8f, 0.0f, 1.0f}, {0.8f, -0.8f, 0.0f, 1.0f}, {0.0f, 0.8f, 0.0f, 1.0f}};
    for (int v = 0; v < 3; v++)
        for (int a = 0; a < DR_ATTRS; a++)
            for (int c = 0; c < 4; c++) vtx[(v * DR_ATTRS + a) * 4 + c] = a == 0 ? pos[v][c] : c == 3 ? 1.0f : 0.5f;
    float *ub = (float *)(pool.mem + 0x90000);
    for (int m = 0; m < 256; m++)
        for (int i = 0; i < 16; i++) ub[m * 16 + i] = i % 5 == 0 ? 1.0f : 0.0f;

    const uint8_t clear_b[4] = {to_byte(0.2f), to_byte(0.4f), to_byte(0.6f), 255};
    uint32_t clear_v;
    memcpy(&clear_v, clear_b, 4);
    static uint32_t px[TEX_W * TEX_H];
    static _Alignas(8) uint8_t stream[8192];
    int tries = 0, drawn = 0;
    prog_t *hit = NULL;
    for (int t = 0; t < 150 && !drawn; t++) {
        const int *pr = pairs[(long long)t * np / (np < 150 ? np : 150)];
        if (t && pr == pairs[(long long)(t - 1) * np / (np < 150 ? np : 150)]) continue;
        const DrawEnt *both[2] = {&e[pr[0]], &e[pr[1]]};
        prog_t *prog = dr_prog(root, both, 2);
        CHECK(prog, "census code file of %.16s is missing", both[0]->key);
        tries++;
        uint8_t *q = draw_prefix(stream, prog, &pool, &color);
        const p_draw d = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
        q = put(q, OP_DRAW, &d, sizeof d);
        q = put(q, OP_END, NULL, 0);
        CHECK(q <= stream + sizeof stream, "draw op stream overflow");
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t f = b->submit(&handle, 1);
        CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the draw submission timed out");
        const NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
        b->read_texels(&color, NULL, &r, (uint8_t *)px, TEX_W * 4, TEX_W * TEX_H * 4);
        int diff = 0;
        for (int k = 0; k < TEX_W * TEX_H; k++) diff += px[k] != clear_v;
        if (diff) drawn = diff, hit = prog;
        if (diff && vk_caps.gpl) {   // the same draw through the shared set 0 layout: texel-equal
            static uint32_t px2[TEX_W * TEX_H];
            int was = vk_pipeline_set_indep(0);
            f = b->submit(&handle, 1);
            CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the shared-layout draw timed out");
            b->read_texels(&color, NULL, &r, (uint8_t *)px2, TEX_W * 4, TEX_W * TEX_H * 4);
            vk_pipeline_set_indep(was);
            int ne = 0;
            for (int k = 0; k < TEX_W * TEX_H; k++) ne += px[k] != px2[k];
            CHECK(!ne, "%d of %d texels differ between independent and shared layouts", ne, TEX_W * TEX_H);
            layouts_equal = TEX_W * TEX_H;
        }
    }
    CHECK(drawn, "none of %d census pairs changed a pixel of the cleared target", tries);

    // DrawElementsIndirect (OP_DRAW_IDX_IND) with the pair that drew: indices {0, 1, 2} at 0xA0000; the arguments are
    // copied by the GPU (COPY_B2B, barrier) to 0xA0200 right before the draw, so the replay must read them on the GPU.
    // {3, 1, 0, 0, 0} draws the same triangle (texel-equal to the direct draw); count 0 draws nothing.
    uint32_t *ix = (uint32_t *)(pool.mem + 0xA0000), *args = (uint32_t *)(pool.mem + 0xA0100);
    ix[0] = 0, ix[1] = 1, ix[2] = 2;
    for (int pass = 0; pass < 2; pass++) {
        const uint32_t want[5] = {pass ? 0u : 3u, 1, 0, 0, 0};
        memcpy(args, want, sizeof want);
        memset(pool.mem + 0xA0200, 0, 20);
        uint8_t *q = draw_prefix(stream, hit, &pool, &color);
        const p_copy_b2b cp = {(uint64_t)(uintptr_t)args, (uint64_t)(uintptr_t)(pool.mem + 0xA0200), 20};
        q = put(q, OP_COPY_B2B, &cp, sizeof cp);
        const p_barrier bar = {0, 0};
        q = put(q, OP_BARRIER, &bar, sizeof bar);
        const p_draw_idx_ind di = {NVN_DRAW_PRIMITIVE_TRIANGLES, NVN_INDEX_TYPE_UNSIGNED_INT, (uint64_t)(uintptr_t)ix,
                                   (uint64_t)(uintptr_t)(pool.mem + 0xA0200)};
        q = put(q, OP_DRAW_IDX_IND, &di, sizeof di);
        q = put(q, OP_END, NULL, 0);
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t f = b->submit(&handle, 1);
        CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the indirect draw timed out");
        static uint32_t pxi[TEX_W * TEX_H];
        const NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
        b->read_texels(&color, NULL, &r, (uint8_t *)pxi, TEX_W * 4, TEX_W * TEX_H * 4);
        int ne = 0;
        for (int k = 0; k < TEX_W * TEX_H; k++) ne += pxi[k] != (pass ? clear_v : px[k]);
        CHECK(!ne, "indirect draw with count %u: %d of %d texels differ from the %s", want[0], ne, TEX_W * TEX_H,
              pass ? "cleared target" : "direct draw");
        ind_draw_texels += TEX_W * TEX_H;
    }

    // Compute: a few census compute shaders, one dispatch each, resources unbound (zero buffers and stand-in images).
    int dispatched = 0;
    prog_t *cs_prog = NULL;
    for (int t = 0; t < 4 && t < nc; t++) {
        const DrawEnt *one[1] = {&e[cs[(long long)t * nc / (nc < 4 ? nc : 4)]]};
        prog_t *prog = dr_prog(root, one, 1);
        CHECK(prog, "census code file of %.16s is missing", one[0]->key);
        cs_prog = prog;
        uint8_t *q = stream;
        q = put(q, OP_BEGIN, NULL, 0);
        const p_bind_program bp = {prog, 1 << 5};
        q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
        const p_dispatch dp = {1, 1, 1, 0};
        q = put(q, OP_DISPATCH, &dp, sizeof dp);
        // DispatchComputeIndirect: group counts {1, 1, 1} copied by the GPU to 0xA0500 first
        const uint32_t groups[3] = {1, 1, 1};
        memcpy(pool.mem + 0xA0400, groups, sizeof groups);
        memset(pool.mem + 0xA0500, 0, sizeof groups);
        const p_copy_b2b cp = {(uint64_t)(uintptr_t)(pool.mem + 0xA0400), (uint64_t)(uintptr_t)(pool.mem + 0xA0500), 12};
        q = put(q, OP_COPY_B2B, &cp, sizeof cp);
        const p_barrier bar = {0, 0};
        q = put(q, OP_BARRIER, &bar, sizeof bar);
        const p_dispatch_ind dpi = {(uint64_t)(uintptr_t)(pool.mem + 0xA0500), 0};
        q = put(q, OP_DISPATCH_IND, &dpi, sizeof dpi);
        q = put(q, OP_END, NULL, 0);
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t f = b->submit(&handle, 1);
        CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the dispatch submission timed out");
        dispatched++;
        ind_dispatches++;
    }
    if (test_marks(b, &pool, &color, hit, cs_prog)) return 1;
    b->texture_release(&color);
    b->pool_fini(&pool);
    free(pairs), free(cs), free(e), free(jobs);
    *n_drawn = drawn, *n_tries = tries, *n_cs = dispatched;
    return 0;
}

// ---- xfer part: copies, texture clear, event and counter through submit.

#define XF_W 64
#define XF_H 64
#define XF_BYTES (XF_W * XF_H * 4)

// Texture in the xfer pool at `off`.
static int xf_tex(tex_t *t, pool_t *pool, int64_t off, uint64_t serial) {
    init_tex(t, pool, off, NVN_FORMAT_RGBA8);
    t->d.width = XF_W, t->d.height = XF_H;
    nvn_tex_layout(&t->d, &t->L);
    t->serial = serial;
    return t->L.size <= 0x20000;
}

static int xf_submit(const GfxBackend *b, uint8_t *stream, uint8_t *end) {
    end = put(end, OP_END, NULL, 0);
    (void)stream;
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    return b->fence_wait(f, 5000000000ull);
}

static int test_xfer(const GfxBackend *b, uint8_t *guest, int *bytes) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000, .size = 0x100000,
                   .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool for the xfer part");
    uint8_t *m = pool.mem;
    static tex_t ta, tb, tc;
    CHECK(xf_tex(&ta, &pool, 0x40000, 201) && xf_tex(&tb, &pool, 0x60000, 202) && xf_tex(&tc, &pool, 0x80000, 203),
          "xfer textures exceed their pool slots");
    // Linear buffers: B2B source 0x0, target 0x1000 (4 KiB); texture source 0x2000, read-backs 0x6000 and 0xA000.
    for (int i = 0; i < 4096; i++) m[i] = (uint8_t)(i * 7 + 3), m[0x1000 + i] = 0;
    uint32_t *img = (uint32_t *)(m + 0x2000);
    for (int i = 0; i < XF_W * XF_H; i++) img[i] = 0x01010101u * (uint32_t)(i & 0xff) + (uint32_t)i * 0x00010000u;
    memset(m + 0x6000, 0, XF_BYTES), memset(m + 0xA000, 0, XF_BYTES);
    volatile uint32_t *event = (volatile uint32_t *)(m + 0xE000);
    NVNcounterData *ctr = (NVNcounterData *)(m + 0xE100), *sp = (NVNcounterData *)(m + 0xE200);
    *event = 0, ctr->counter = 99, ctr->timestamp = 0;
    sp->counter = 99, sp->timestamp = 0;

    static _Alignas(8) uint8_t stream[4096];
    uint8_t *p = stream;
    p = put(p, OP_BEGIN, NULL, 0);
    const p_copy_b2b bb = {(uint64_t)(uintptr_t)m, (uint64_t)(uintptr_t)(m + 0x1000), 4096};
    p = put(p, OP_COPY_B2B, &bb, sizeof bb);
    p_copy_b2t up;
    memset(&up, 0, sizeof up);
    up.src = (uint64_t)(uintptr_t)(m + 0x2000), up.tex = &ta, up.r = (NVNcopyRegion){0, 0, 0, XF_W, XF_H, 1};
    p = put(p, OP_COPY_B2T, &up, sizeof up);
    p_copy_b2t down = up;
    down.src = (uint64_t)(uintptr_t)(m + 0x6000);
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    p_copy_t2t tt;
    memset(&tt, 0, sizeof tt);
    tt.src = &ta, tt.dst = &tb, tt.sr = tt.dr = up.r;
    p = put(p, OP_COPY_T2T, &tt, sizeof tt);
    down.tex = &tb, down.src = (uint64_t)(uintptr_t)(m + 0xA000);
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    p_clear_tex ct;
    memset(&ct, 0, sizeof ct);
    ct.tex = &tc, ct.r = (NVNcopyRegion){0, 0, 0, XF_W, XF_H, 1}, ct.mask = 0xf;
    ct.c.f[0] = 0.2f, ct.c.f[1] = 0.4f, ct.c.f[2] = 0.6f, ct.c.f[3] = 1.0f;
    p = put(p, OP_CLEAR_TEX, &ct, sizeof ct);
    const p_signal_event ev1 = {event, NVN_EVENT_SIGNAL_MODE_WRITE, 7}, ev2 = {event, NVN_EVENT_SIGNAL_MODE_ADD, 5};
    p = put(p, OP_SIGNAL_EVENT, &ev1, sizeof ev1);
    p = put(p, OP_SIGNAL_EVENT, &ev2, sizeof ev2);
    const p_report_counter rc = {NVN_COUNTER_TYPE_TIMESTAMP, (uint64_t)(uintptr_t)ctr};
    p = put(p, OP_REPORT_COUNTER, &rc, sizeof rc);
    // Samples passed: a reset, no draw, a report: {0, time} through the occlusion-query path.
    const p_reset_counter rs = {NVN_COUNTER_TYPE_SAMPLES_PASSED};
    p = put(p, OP_RESET_COUNTER, &rs, sizeof rs);
    const p_report_counter rsp = {NVN_COUNTER_TYPE_SAMPLES_PASSED, (uint64_t)(uintptr_t)sp};
    p = put(p, OP_REPORT_COUNTER, &rsp, sizeof rsp);
    CHECK(p <= stream + sizeof stream - sizeof(op_hdr), "xfer op stream overflow");
    CHECK(xf_submit(b, stream, p), "fence_wait of the xfer submission timed out");

    for (int i = 0; i < 4096; i++) CHECK(m[0x1000 + i] == m[i], "B2B byte %d is %02x, expected %02x", i, m[0x1000 + i], m[i]);
    for (int i = 0; i < XF_W * XF_H; i++) {
        CHECK(((uint32_t *)(m + 0x6000))[i] == img[i], "B2T then T2B texel %d is %08x, expected %08x", i,
              ((uint32_t *)(m + 0x6000))[i], img[i]);
        CHECK(((uint32_t *)(m + 0xA000))[i] == img[i], "T2T then T2B texel %d is %08x, expected %08x", i,
              ((uint32_t *)(m + 0xA000))[i], img[i]);
    }
    static uint32_t px[XF_W * XF_H];
    const NVNcopyRegion r = {0, 0, 0, XF_W, XF_H, 1};
    b->read_texels(&ta, NULL, &r, (uint8_t *)px, XF_W * 4, XF_BYTES);
    for (int i = 0; i < XF_W * XF_H; i++) CHECK(px[i] == img[i], "read_texels of the copied texture, texel %d is %08x", i, px[i]);
    b->read_texels(&tc, NULL, &r, (uint8_t *)px, XF_W * 4, XF_BYTES);
    const uint8_t cl[4] = {to_byte(0.2f), to_byte(0.4f), to_byte(0.6f), 255};
    uint32_t clv;
    memcpy(&clv, cl, 4);
    for (int i = 0; i < XF_W * XF_H; i++) CHECK(px[i] == clv, "CLEAR_TEX texel %d is %08x, expected %08x", i, px[i], clv);
    CHECK(*event == 12, "event word is %u after WRITE 7 and ADD 5", *event);
    CHECK(ctr->counter == 0 && ctr->timestamp != 0, "counter report is {%llu, %llu}", (unsigned long long)ctr->counter,
          (unsigned long long)ctr->timestamp);
    CHECK(sp->counter == 0 && sp->timestamp != 0, "samples-passed report after a reset and no draw is {%llu, %llu}",
          (unsigned long long)sp->counter, (unsigned long long)sp->timestamp);
    b->texture_release(&ta);
    b->texture_release(&tb);
    b->texture_release(&tc);
    b->pool_fini(&pool);
    *bytes = 4096 + 2 * XF_BYTES;
    return 0;
}

// c0 as vk_c0_fill builds it (CPU only): compute storage and uniform tables, dispatch size, unbound slots zero; the
// vertex stage's point size clamp and base instance.
static int test_c0(void) {
    static p_bind_buf ssbo[16], ubo[16];
    static uint8_t bank[VK_C0_BYTES];
    ssbo[1] = (p_bind_buf){5, 1, 0x12340000, 0x400};
    ubo[4] = (p_bind_buf){5, 4, 0x55660000, 0x100};
    const uint32_t dim[3] = {3, 2, 1};
    uint32_t u[4];
    uint64_t a;
    vk_c0_fill(bank, 5, ssbo, ubo, dim, 7);
    memcpy(u, bank + 0x200, 12);
    CHECK(u[0] == 3 && u[1] == 2 && u[2] == 1, "c0 dispatch size is %u %u %u", u[0], u[1], u[2]);
    memcpy(&a, bank + 0x320, 8), memcpy(u, bank + 0x328, 4);
    CHECK(a == 0x12340000 && u[0] == 0x400, "c0 0x320 (storage slot 1) is {%#llx, %#x}", (unsigned long long)a, u[0]);
    memcpy(&a, bank + 0x250, 8), memcpy(u, bank + 0x258, 4);
    CHECK(a == 0x55660000 && u[0] == 0x100, "c0 0x250 (uniform slot 4) is {%#llx, %#x}", (unsigned long long)a, u[0]);
    for (int i = 0; i < 16; i++) CHECK(!bank[0x310 + i], "c0 0x310 (unbound storage slot 0) byte %d is %02x", i, bank[0x310 + i]);
    CHECK(!bank[0x644] && !bank[0x610], "c0 of a compute stage has graphics words");
    vk_c0_fill(bank, 0, ssbo, ubo, dim, 61);
    float ps;
    memcpy(&ps, bank + 0x610, 4), memcpy(u, bank + 0x644, 4);
    CHECK(ps == 2047.0f && u[0] == 61, "vertex c0: point size %g at 0x610, base instance %u at 0x644", ps, u[0]);
    memcpy(&a, bank + 0x120, 8);
    CHECK(a == 0x12340000 && !bank[0x200], "vertex c0: storage slot 1 at 0x120 is %#llx", (unsigned long long)a);
    return 0;
}

// A texture in a physical pool's memory that nothing wrote reads zero, as Switch pool memory starts zeroed
// (NX2NAVI_FIX_ZERO; the driver may return zeros without it, then this is a guard).
static int test_zero(const GfxBackend *b, uint8_t *guest, int *bytes) {
    pool_t phys = {.magic = NVN_POOL_MAGIC, .flags = 0x189, .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .size = 0x100000,
                   .game_mem = guest + 0xA00000};
    pool_t cpu = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000, .size = 0x100000, .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&phys, &phys), "pool_init of a physical pool");
    CHECK(b->pool_init(&cpu, &cpu), "pool_init of a CPU-visible pool for the zero part");
    static tex_t t;
    init_tex(&t, &phys, 0, NVN_FORMAT_RGBA8);
    t.d.width = 64, t.d.height = 64;
    nvn_tex_layout(&t.d, &t.L);
    t.serial = 301;
    memset(cpu.mem, 0xab, 64 * 64 * 4);
    static _Alignas(8) uint8_t stream[512];
    uint8_t *p = put(stream, OP_BEGIN, NULL, 0);
    p_copy_b2t down;
    memset(&down, 0, sizeof down);
    down.src = (uint64_t)(uintptr_t)cpu.mem, down.tex = &t, down.r = (NVNcopyRegion){0, 0, 0, 64, 64, 1};
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    CHECK(xf_submit(b, stream, p), "fence_wait of the zero submission timed out");
    for (int i = 0; i < 64 * 64 * 4; i++) CHECK(!cpu.mem[i], "unwritten texel byte %d of a physical-pool texture is %02x", i, cpu.mem[i]);
    b->texture_release(&t);
    b->pool_fini(&cpu);
    b->pool_fini(&phys);
    *bytes = 64 * 64 * 4;
    return 0;
}

// ClearBuffer / ReportValue (OP_CLEAR_BUF), Downsample (a linear OP_COPY_T2T to half size) and the stencil view
// aspect. Fills: 64 bytes in a CPU-visible pool, a 4-byte report over the first word in the same submission (GPU order:
// the report wins), and 64 bytes in a CPU-invisible physical pool copied out with COPY_B2B. Downsample: a 64x64 RGBA8
// texture to 32x32, each texel the 2x2 average within 1. Stencil: views of a D24S8 texture pick the stencil aspect in
// STENCIL mode and depth otherwise, and a stencil-aspect sampled view of the image is created.
static int test_nvnx(const GfxBackend *b, uint8_t *guest, int *bytes) {
    pool_t cpu = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000, .size = 0x100000, .game_mem = guest + 0x800000};
    pool_t phys = {.magic = NVN_POOL_MAGIC, .flags = 0x189, .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .size = 0x100000,
                   .game_mem = guest + 0xA00000};
    CHECK(b->pool_init(&cpu, &cpu) && b->pool_init(&phys, &phys), "pool_init of the nvnx pools");
    uint8_t *m = cpu.mem;
    memset(m, 0x11, 0x400);
    static tex_t src, dst;
    init_tex(&src, &cpu, 0x40000, NVN_FORMAT_RGBA8);
    src.d.width = 64, src.d.height = 64, src.serial = 401;
    nvn_tex_layout(&src.d, &src.L);
    init_tex(&dst, &cpu, 0x60000, NVN_FORMAT_RGBA8);
    dst.d.width = 32, dst.d.height = 32, dst.serial = 402;
    nvn_tex_layout(&dst.d, &dst.L);
    CHECK(src.L.size <= 0x20000 && dst.L.size <= 0x20000, "downsample textures exceed their pool slots");
    uint8_t *img = m + 0x1000, *out = m + 0x8000;
    for (int i = 0; i < 64 * 64 * 4; i++) img[i] = (uint8_t)((i * 37 + (i >> 8) * 11) & 0xff);
    memset(out, 0, 32 * 32 * 4);

    static _Alignas(8) uint8_t stream[2048];
    uint8_t *p = put(stream, OP_BEGIN, NULL, 0);
    const p_clear_buf f1 = {(uint64_t)(uintptr_t)m, 64, 0xdeadbeefu, 0}, rv = {(uint64_t)(uintptr_t)m, 4, 0x1234u, 0};
    p = put(p, OP_CLEAR_BUF, &f1, sizeof f1);
    p = put(p, OP_CLEAR_BUF, &rv, sizeof rv);
    const p_clear_buf f2 = {RT_GPU_VA_BASE + 0x100, 64, 0xa5a5a5a5u, 0};
    p = put(p, OP_CLEAR_BUF, &f2, sizeof f2);
    const p_copy_b2b back = {RT_GPU_VA_BASE + 0x100, (uint64_t)(uintptr_t)(m + 0x200), 64};
    p = put(p, OP_COPY_B2B, &back, sizeof back);
    p_copy_b2t up;
    memset(&up, 0, sizeof up);
    up.src = (uint64_t)(uintptr_t)img, up.tex = &src, up.r = (NVNcopyRegion){0, 0, 0, 64, 64, 1};
    p = put(p, OP_COPY_B2T, &up, sizeof up);
    p_copy_t2t ds;   // as nvnCommandBufferDownsample records it
    memset(&ds, 0, sizeof ds);
    ds.src = &src, ds.dst = &dst, ds.sr = (NVNcopyRegion){0, 0, 0, 64, 64, 1}, ds.dr = (NVNcopyRegion){0, 0, 0, 32, 32, 1};
    ds.flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
    p = put(p, OP_COPY_T2T, &ds, sizeof ds);
    p_copy_b2t down;
    memset(&down, 0, sizeof down);
    down.src = (uint64_t)(uintptr_t)out, down.tex = &dst, down.r = (NVNcopyRegion){0, 0, 0, 32, 32, 1};
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    CHECK(xf_submit(b, stream, p), "fence_wait of the nvnx submission timed out");

    uint32_t w0;
    memcpy(&w0, m, 4);
    CHECK(w0 == 0x1234u, "ReportValue word after the fill is %#x", w0);
    for (int i = 1; i < 16; i++) {
        uint32_t w;
        memcpy(&w, m + 4 * i, 4);
        CHECK(w == 0xdeadbeefu, "ClearBuffer word %d is %#x", i, w);
    }
    CHECK(m[64] == 0x11, "ClearBuffer wrote past its 64 bytes (%02x)", m[64]);
    for (int i = 0; i < 64; i++) CHECK(m[0x200 + i] == 0xa5, "physical-pool fill byte %d is %02x", i, m[0x200 + i]);
    int worst = 0;
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
            for (int ch = 0; ch < 4; ch++) {
                int sum = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) sum += img[((2 * y + dy) * 64 + 2 * x + dx) * 4 + ch];
                int d = abs(out[(y * 32 + x) * 4 + ch] - (sum + 2) / 4);
                if (d > worst) worst = d;
            }
    CHECK(worst <= 1, "downsampled texel off the 2x2 average by %d", worst);

    static tex_t dsx;
    init_tex(&dsx, &cpu, 0x80000, NVN_FORMAT_DEPTH24_STENCIL8);
    dsx.d.width = 64, dsx.d.height = 64, dsx.serial = 403;
    nvn_tex_layout(&dsx.d, &dsx.L);
    const VkTexImage *di = vk_texture_get(&dsx);
    CHECK(di && (di->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && (di->aspect & VK_IMAGE_ASPECT_DEPTH_BIT), "D24S8 image with both aspects");
    view_t sv, dv;
    memset(&sv, 0, sizeof sv), memset(&dv, 0, sizeof dv);
    sv.set_dsmode = 1, sv.dsmode = NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL;
    dv.set_dsmode = 1, dv.dsmode = NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    CHECK(vk_exec_view_aspect(di, &sv, 1) == VK_IMAGE_ASPECT_STENCIL_BIT && vk_exec_view_aspect(di, &dv, 1) == VK_IMAGE_ASPECT_DEPTH_BIT &&
              vk_exec_view_aspect(di, &sv, 0) == VK_IMAGE_ASPECT_DEPTH_BIT, "view aspects of a D24S8 texture");
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = di->image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = di->format,
                                .subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}};
    VkImageView view = VK_NULL_HANDLE;
    CHECK(vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) == VK_SUCCESS, "stencil-aspect view of the D24S8 image");
    vkDestroyImageView(vk_ctx()->device, view, NULL);
    b->texture_release(&src);
    b->texture_release(&dst);
    b->texture_release(&dsx);
    b->pool_fini(&cpu);
    b->pool_fini(&phys);
    *bytes = 64 + 64 + 32 * 32 * 4;
    return 0;
}

// The source and destination cross different mapping boundaries. Stop at a real source hole without touching the tail.
static int test_segmented_buffer_transfer(const GfxBackend *b, uint8_t *guest) {
    enum { PAGE = 0x10000 };
    const uint64_t a = RT_GPU_VA_BASE, c = a + 4*PAGE, sv = c + 4*PAGE, dv = sv + 3*PAGE;
    pool_t physical[2] = {
        {.magic=NVN_POOL_MAGIC, .flags=0x189, .mem=(uint8_t *)(uintptr_t)a, .size=4*PAGE, .game_mem=guest+0xA00000},
        {.magic=NVN_POOL_MAGIC, .flags=0x189, .mem=(uint8_t *)(uintptr_t)c, .size=4*PAGE, .game_mem=guest+0xB00000},
    };
    pool_t windows[2] = {
        {.magic=NVN_POOL_MAGIC, .flags=0x2a1, .mem=(uint8_t *)(uintptr_t)sv, .size=3*PAGE, .span=3*PAGE},
        {.magic=NVN_POOL_MAGIC, .flags=0x2a1, .mem=(uint8_t *)(uintptr_t)dv, .size=3*PAGE, .span=3*PAGE},
    };
    pool_t cpu = {.magic=NVN_POOL_MAGIC, .flags=0x22, .mem=guest+0xC00000, .size=0x100000, .game_mem=guest+0xC00000};
    CHECK(b->pool_init(&cpu,&cpu), "imported transfer result pool");
    for (int i=0;i<2;i++) {
        CHECK(b->pool_init(&physical[i],&physical[i]), "physical transfer pool %d", i);
        CHECK(b->pool_init(&windows[i],&windows[i]), "virtual transfer window %d", i);
    }
    if (vk_mirror_on) CHECK(vk_mem_mirror_probe((uintptr_t)cpu.mem), "installed transfer result mirror");
    const NVNmemoryPool *pa = (const NVNmemoryPool *)(const void *)&physical[0];
    const NVNmemoryPool *pc = (const NVNmemoryPool *)(const void *)&physical[1];
    const nvn_mapping_t source[] = {{pa,0,0,PAGE},{pc,0,PAGE,PAGE}};
    const nvn_mapping_t destination[] = {{pa,2*PAGE,0,PAGE},{pc,2*PAGE,PAGE,PAGE},{pa,3*PAGE,2*PAGE,PAGE}};
    CHECK(b->pool_map_virtual(&windows[0],2,source) && b->pool_map_virtual(&windows[1],3,destination),
          "segmented source and destination virtual mappings");

    _Alignas(8) uint8_t stream[4096];
    uint8_t *p = put(stream,OP_BEGIN,NULL,0);
    const p_clear_buf initial[] = {{a,PAGE,0x11111111,0},{c,PAGE,0x22222222,0},
        {a+2*PAGE,2*PAGE,0xa5a5a5a5,0},{c+2*PAGE,2*PAGE,0xa5a5a5a5,0}};
    for (unsigned i=0;i<sizeof initial/sizeof initial[0];i++) p=put(p,OP_CLEAR_BUF,&initial[i],sizeof initial[i]);
    CHECK(xf_submit(b,stream,p), "seed exact physical transfer bytes");

    vk_exec_buffer_transfer_probe_arm(1);
    p = put(stream,OP_BEGIN,NULL,0);
    const p_copy_b2b segmented = {sv+PAGE-32,dv+PAGE-16,PAGE+80};
    p = put(p,OP_COPY_B2B,&segmented,sizeof segmented);
    CHECK(xf_submit(b,stream,p), "segmented copy and bounded hole tail");
    VkBufferTransferProbeStats stats=vk_exec_buffer_transfer_probe_stats();
    CHECK(stats.copies==1 && stats.clears==0 && stats.legs==4 && stats.bytes==PAGE+32 && stats.partial==1,
          "segmented copy recorded %llu copies and %llu legs across %llu bytes with %llu partial tails",
          (unsigned long long)stats.copies,(unsigned long long)stats.legs,(unsigned long long)stats.bytes,
          (unsigned long long)stats.partial);

    uint8_t *out=cpu.mem;
    uint64_t result=(uintptr_t)out;
    const p_copy_b2b observe[] = {{a+3*PAGE-20,result,20},{c+2*PAGE,result+20,PAGE},
        {a+3*PAGE,result+PAGE+20,64}};
    p=put(stream,OP_BEGIN,NULL,0);
    for (unsigned i=0;i<sizeof observe/sizeof observe[0];i++) p=put(p,OP_COPY_B2B,&observe[i],sizeof observe[i]);
    CHECK(xf_submit(b,stream,p), "read segmented output by its physical aliases");
    b->guest_read(result,PAGE+84);
    for (unsigned i=0;i<PAGE+84;i++) {
        uint8_t expected=i<4 || i>=PAGE+36 ? 0xa5 : i<36 ? 0x11 : 0x22;
        CHECK(out[i]==expected,"segmented output byte %#x was %#x, expected %#x",i,out[i],expected);
    }

    // The next clear reaches the replacement physical page, while its former provider and neighbors remain intact.
    const nvn_mapping_t remap = {pc,3*PAGE,PAGE,PAGE};
    CHECK(b->pool_map_virtual(&windows[1],1,&remap), "replace one virtual destination mapping");
    vk_exec_buffer_transfer_probe_arm(1);
    p=put(stream,OP_BEGIN,NULL,0);
    const p_clear_buf split={dv+PAGE-16,32,0x33333333,0};
    const p_clear_buf mirror={result+2*PAGE,32,0x44444444,0};
    p=put(p,OP_CLEAR_BUF,&split,sizeof split);
    p=put(p,OP_CLEAR_BUF,&mirror,sizeof mirror);
    CHECK(xf_submit(b,stream,p), "clear across remapped pages and into the CPU-visible destination");
    stats=vk_exec_buffer_transfer_probe_stats();
    CHECK(stats.clears==2 && stats.legs==3 && stats.bytes==64 && stats.partial==0 &&
          stats.mirror_twin==(uint64_t)(vk_mirror_on!=0), "typed clear vectors and mirror twin emission");
    p=put(stream,OP_BEGIN,NULL,0);
    const p_copy_b2b replaced[]={{a+3*PAGE-20,result,20},{c+2*PAGE,result+20,32},
        {c+3*PAGE,result+52,32}};
    for (unsigned i=0;i<sizeof replaced/sizeof replaced[0];i++) p=put(p,OP_COPY_B2B,&replaced[i],sizeof replaced[i]);
    CHECK(xf_submit(b,stream,p), "read previous and replacement physical destinations");
    b->guest_read(result,84);
    b->guest_read(result+2*PAGE,32);
    for (unsigned i=0;i<84;i++) {
        uint8_t expected=i<4 || i>=68 ? 0xa5 : i<20 || i>=52 ? 0x33 : i<36 ? 0x11 : 0x22;
        CHECK(out[i]==expected,"remapped output byte %#x was %#x, expected %#x",i,out[i],expected);
    }
    for (unsigned i=0;i<32;i++) CHECK(out[2*PAGE+i]==0x44,"CPU-visible clear byte %u",i);
    vk_exec_buffer_transfer_probe_arm(0);
    for (int i=0;i<2;i++) b->pool_fini(&windows[i]);
    for (int i=0;i<2;i++) b->pool_fini(&physical[i]);
    b->pool_fini(&cpu);
    printf("probe-vkexec: retained copy legs 4, exact bytes %u, untouched hole tail 48, remapped clear legs 2, mirror twin %llu\n",
           (unsigned)(PAGE+32),(unsigned long long)stats.mirror_twin);
    return 0;
}

// VRAM mirror (vk_memory.c; skipped in import mode): a pool given its mirror as on a first buffer use, after which
// buffer copies and fills read and write the mirror. The first use copies the whole pool; CPU writes between
// submissions reach the mirror through write watch; GPU writes reach guest memory by the fence, and only the bytes
// written: CPU bytes beside a GPU fill keep their values.
static int test_mirror(const GfxBackend *b, uint8_t *guest, int *bytes) {
    *bytes = 0;
    if (!vk_mirror_on) return 0;
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000, .size = 0x100000,
                   .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool for the mirror part");
    uint8_t *m = pool.mem;
    for (int i = 0; i < 4096; i++) m[i] = (uint8_t)(i * 13 + 1);
    memset(m + 0x1000, 0xEE, 0x5000);
    CHECK(vk_mem_mirror_probe((uint64_t)(uintptr_t)m), "the pool got no VRAM mirror");
    static _Alignas(8) uint8_t stream[1024];
    uint64_t a = (uint64_t)(uintptr_t)m;
    // 1: B2B in the mirror after its first-use copy, and a fill; both written back.
    uint8_t *p = put(stream, OP_BEGIN, NULL, 0);
    const p_copy_b2b c1 = {a, a + 0x1000, 4096};
    const p_clear_buf f1 = {a + 0x2000, 0x800, 0x5a5a5a5au, 0};
    p = put(p, OP_COPY_B2B, &c1, sizeof c1);
    p = put(p, OP_CLEAR_BUF, &f1, sizeof f1);
    CHECK(xf_submit(b, stream, p), "fence_wait of the first mirror submission timed out");
    for (int i = 0; i < 4096; i++) CHECK(m[0x1000 + i] == m[i], "mirror B2B byte %d is %02x, expected %02x", i, m[0x1000 + i], m[i]);
    for (int i = 0; i < 0x800; i++) CHECK(m[0x2000 + i] == 0x5a, "mirror fill byte %d is %02x", i, m[0x2000 + i]);
    for (int i = 0x800; i < 0x1000; i++) CHECK(m[0x2000 + i] == 0xEE, "byte %#x past the mirror fill is %02x", i, m[0x2000 + i]);
    // 2: the CPU rewrites the source; the next submission copies the new bytes.
    for (int i = 0; i < 4096; i++) m[i] = (uint8_t)(i * 7 + 5);
    p = put(stream, OP_BEGIN, NULL, 0);
    const p_copy_b2b c2 = {a, a + 0x3000, 4096};
    p = put(p, OP_COPY_B2B, &c2, sizeof c2);
    CHECK(xf_submit(b, stream, p), "fence_wait of the second mirror submission timed out");
    for (int i = 0; i < 4096; i++)
        CHECK(m[0x3000 + i] == m[i], "B2B byte %d after a CPU write is %02x, expected %02x (write not mirrored)", i,
              m[0x3000 + i], m[i]);
    // 3: CPU bytes in the page of a GPU fill, written before the submission, survive the write-back.
    memset(m + 0x4000, 0x11, 0x800);
    p = put(stream, OP_BEGIN, NULL, 0);
    const p_clear_buf f3 = {a + 0x4800, 0x800, 0x77777777u, 0};
    p = put(p, OP_CLEAR_BUF, &f3, sizeof f3);
    CHECK(xf_submit(b, stream, p), "fence_wait of the third mirror submission timed out");
    for (int i = 0; i < 0x800; i++) CHECK(m[0x4000 + i] == 0x11, "CPU byte %#x beside a fill is %02x", i, m[0x4000 + i]);
    for (int i = 0; i < 0x800; i++) CHECK(m[0x4800 + i] == 0x77, "fill byte %#x is %02x", i, m[0x4800 + i]);
    // 4: two small flushes at the ends of the pool copy their own pages, not what lies between them.
    int fl_bad = vk_mem_flush_add_probe();
    CHECK(!fl_bad, "flush range list check %d failed", fl_bad);
    p = put(stream, OP_BEGIN, NULL, 0);
    const p_copy_b2b c4 = {a, a + 0x3000, 4096};
    p = put(p, OP_COPY_B2B, &c4, sizeof c4);
    CHECK(xf_submit(b, stream, p), "fence_wait of the mirror submission before the flushes timed out");
    uint64_t up0 = vk_mem_mirror_uploaded();
    nvn_pool_flushed(&pool, 0x6000, 0x1000);
    nvn_pool_flushed(&pool, 0xF0000, 0x1000);
    p = put(stream, OP_BEGIN, NULL, 0);
    p = put(p, OP_COPY_B2B, &c4, sizeof c4);
    CHECK(xf_submit(b, stream, p), "fence_wait of the mirror submission after the flushes timed out");
    // A submission's copies reach the counter with the next one, and c4's own write-back adds its destination page.
    CHECK(xf_submit(b, stream, p), "fence_wait of the mirror submission that counts the flushes timed out");
    uint64_t up = vk_mem_mirror_uploaded() - up0;
    if (!rt_knob("FLUSH_SPAN", "0"))
        CHECK(up >= 0x2000 && up < 0x10000, "two flushed pages copied %#llx bytes, 0xeb000 would be the span",
              (unsigned long long)up);
    b->pool_fini(&pool);
    *bytes = 4096 + 0x1000 + 4096 + 0x1000;
    return 0;
}

// A pool of zero bytes (NX2navi makes some now and then) initialises as an empty pool: no
// Vulkan buffer or memory (a buffer or import of 0 bytes is invalid usage) and no pool index entry, so a lookup at its
// address still finds the pool that starts there, before and after the empty pool's Finalize.
static int test_empty_pool(const GfxBackend *b, uint8_t *guest) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xE00000, .size = 0x100000,
                   .game_mem = guest + 0xE00000};
    pool_t empty = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xE00000, .size = 0,
                    .game_mem = guest + 0xE00000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of a CPU-visible pool for the empty-pool part");
    long errors = vk_validation_errors_total();
    CHECK(b->pool_init(&empty, &empty), "pool_init of a zero-byte pool");
    uint64_t a = (uint64_t)(uintptr_t)pool.mem;
    VkMemSpan whole, at, after;
    CHECK(vk_mem_resolve(a + 16, 16, &whole) && whole.buffer, "the pool beside an empty pool is not resolved");
    CHECK(vk_mem_resolve(a, 0, &at) && at.buffer == whole.buffer && at.offset == whole.offset - 16,
          "a lookup at the start of a pool finds the empty pool there");
    CHECK(vk_validation_errors_total() == errors, "the zero-byte pool raised %ld validation errors",
          vk_validation_errors_total() - errors);
    b->pool_fini(&empty);
    CHECK(vk_mem_resolve(a + 16, 16, &after) && after.buffer == whole.buffer,
          "the empty pool's Finalize took the pool beside it out of the index");
    b->pool_fini(&pool);
    return 0;
}

void nvnp_nvnCommandBufferInitialize(Cpu *);
void nvnp_nvnCommandBufferFinalize(Cpu *);
void nvnp_nvnCommandBufferAddCommandMemory(Cpu *);
void nvnp_nvnCommandBufferAddControlMemory(Cpu *);
void nvnp_nvnCommandBufferBeginRecording(Cpu *);
void nvnp_nvnCommandBufferEndRecording(Cpu *);
void nvnp_nvnCommandBufferClearTexture(Cpu *);
void nvnp_nvnCommandBufferCopyTextureToBuffer(Cpu *);
void nvnp_nvnCommandBufferSetRenderTargets(Cpu *);
void nvnp_nvnCommandBufferSetScissor(Cpu *);
void nvnp_nvnCommandBufferClearColor(Cpu *);
void nvnp_nvnTextureInitialize(Cpu *);
void nvnp_nvnTextureFinalize(Cpu *);

static uint64_t pinned_clear_proc(void (*proc)(Cpu *), const uint64_t args[8]) {
    Cpu cpu = {0};
    memcpy(cpu.x, args, sizeof cpu.x[0] * 8);
    proc(&cpu);
    return cpu.x[0];
}

static void pinned_texel_bytes(void *context, uint64_t offset, uint64_t size) {
    (void)offset;
    *(uint64_t *)context += size;
}

// Real recorded descriptors provide the same pin identity to the clear and its later read after address replacement.
static int test_pinned_clear(const GfxBackend *b, uint8_t *guest) {
    enum { SIDE = 32, LAYERS = 2, LEVELS = 2, OUT_STRIDE = SIDE * SIDE * 4, PASSES = 6 };
    static _Alignas(16) NVNcommandBuffer command;
    static _Alignas(8) uint8_t control[PASSES][4096];
    static _Alignas(8) uint8_t cold_control[4096];
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xA00000,
                   .size = 0x100000, .game_mem = guest + 0xA00000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of the pinned clear fixture");
    tex_t old, replacement = {0};
    init_tex(&old, &pool, 0, NVN_FORMAT_RGBA8);
    old.d.target = NVN_TEXTURE_TARGET_2D_ARRAY;
    old.d.width = old.d.height = SIDE;
    old.d.depth = LAYERS;
    old.d.levels = LEVELS;
    old.serial = nvn_new_serial();
    nvn_tex_layout(&old.d, &old.L);
    uint32_t input[SIDE * SIDE];
    for (unsigned lv = 0; lv < LEVELS; lv++) {
        unsigned side = SIDE >> lv;
        for (unsigned ly = 0; ly < LAYERS; ly++) {
            uint32_t value = 0xff403010u + lv * LAYERS + ly;
            for (unsigned i = 0; i < side * side; i++) input[i] = value;
            CHECK(!nvn_swizzle(&old.d, lv, ly, input, side * 4, old.mem), "swizzling pinned clear inputs");
        }
    }
    b->guest_written((uintptr_t)old.mem, old.L.size);
    uint8_t *output = pool.mem + 0x10000;
    uint8_t *package = pool.mem + 0x40000;
    const float red[4] = {1, 0, 0, 1};
    const float blue[4] = {0, 0, 1, 1}, yellow[4] = {1, 1, 0, 1};
    const view_t selected = {.set_levels = 1, .base_level = 1, .nlevels = 1,
                             .set_layers = 1, .min_layer = 1, .nlayers = 1};
    NVNcopyRegion region = {0, 0, 0, SIDE / 2, SIDE / 2, 1};
#define PC(proc, ...) pinned_clear_proc(nvnp_##proc, (const uint64_t[8]){(uintptr_t)&command, __VA_ARGS__})
    CHECK(PC(nvnCommandBufferInitialize, 0), "pinned clear command buffer initialization");
    PC(nvnCommandBufferAddCommandMemory, (uintptr_t)&pool, 0x20000, 0x4000);
    const tex_t *pin = NULL;
    VkImage original = VK_NULL_HANDLE;
    uint64_t exact_copies = 0;
    for (unsigned pass = 0; pass < PASSES; pass++) {
        PC(nvnCommandBufferAddControlMemory, (uintptr_t)control[pass], sizeof control[pass]);
        PC(nvnCommandBufferBeginRecording, 0);
        if (pass == 1 || pass == 2)
            PC(nvnCommandBufferClearTexture, (uintptr_t)&old, (uintptr_t)&selected, (uintptr_t)&region,
               (uintptr_t)red, pass == 1 ? 7 : 15);
        else if (pass == 3 || pass == 4) {
            if (pass == 3) {
                const tex_t *targets[] = {&old};
                const view_t *views[] = {&selected};
                PC(nvnCommandBufferSetRenderTargets, 1, (uintptr_t)targets, (uintptr_t)views, 0, 0);
            }
            // With no SetRenderTargets in pass 4, its command uses the exact target inherited across submissions.
            PC(nvnCommandBufferSetScissor, pass == 3 ? 4 : 6, pass == 3 ? 4 : 6, pass == 3 ? 8 : 4, pass == 3 ? 8 : 4);
            PC(nvnCommandBufferClearColor, 0, (uintptr_t)(pass == 3 ? blue : yellow), 15);
        }
        else for (unsigned lv = 0; lv < LEVELS; lv++) {
            for (unsigned ly = 0; ly < LAYERS; ly++) {
                view_t view = selected;
                view.base_level = lv;
                view.min_layer = ly;
                NVNcopyRegion all = {0, 0, 0, SIDE >> lv, SIDE >> lv, 1};
                PC(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&old, (uintptr_t)&view, (uintptr_t)&all,
                   (uintptr_t)(output + OUT_STRIDE * (lv * LAYERS + ly)), 0);
                if (pass == PASSES - 1)
                    PC(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&replacement, (uintptr_t)&view, (uintptr_t)&all,
                       (uintptr_t)(output + OUT_STRIDE * (LEVELS * LAYERS + lv * LAYERS + ly)), 0);
            }
        }
        if (pass == PASSES - 1) {
            PC(nvnCommandBufferSetScissor, 0, 0, 1 << 20, 1 << 20);
            PC(nvnCommandBufferSetRenderTargets, 0, 0, 0, 0, 0);
        }
        uint64_t handle = PC(nvnCommandBufferEndRecording, 0);
        CHECK(handle, "pinned clear pass %u recording", pass);
        if (pass == 1 || pass == 2) {
            const op_hdr *op = nvn_op_next(nvn_op_first(handle));
            CHECK(op->op == OP_CLEAR_TEX, "pinned clear operation kind");
            const tex_t *identity = ((const p_clear_tex *)nvn_op_payload(op))->tex;
            if (!pin) pin = identity;
            CHECK(identity == pin, "clear recordings share the actual texture pin");
        }
        if (pass == PASSES - 1) {
            unsigned copies = 0;
            for (const op_hdr *op = nvn_op_first(handle); op->op != OP_END; op = nvn_op_next(op)) {
                if (op->op != OP_COPY_T2B) continue;
                const tex_t *source = ((const p_copy_b2t *)nvn_op_payload(op))->tex;
                CHECK(source && (copies % 2 ? source != pin && source->serial == replacement.serial : source == pin),
                      "ordered transfer %u retained the selected old or replacement serial", copies);
                copies++;
            }
            CHECK(copies == 2 * LEVELS * LAYERS, "all old and replacement subresources were recorded");
            exact_copies = handle;
        }
        uint64_t fence = b->submit(&handle, 1);
        CHECK(b->fence_wait(fence, UINT64_C(5000000000)), "pinned clear pass %u completion", pass);
        if (pass == 0) {
            const VkTexImage *image = vk_texture_get(&old);
            CHECK(image && image->image, "pinned clear initial source residency");
            original = image->image;
        } else if (pass == 1) {
            const VkTexImage *image = vk_texture_get(&old);
            CHECK(image && image->image == original && !nvn_texture_pin_attachment(pin),
                  "unsupported mask must leave the immutable image and pin untouched");
        } else if (pass == 2) {
            CHECK(nvn_texture_pin_attachment(pin), "ordinary immutable clear actually took the pinned COW route");
            const VkTexImage *version = vk_texture_get(pin);
            CHECK(version && version->image != original, "clear published a distinct captured native image");
            VkImage image = version->image;
            tex_t partial = old;
            partial.serial = nvn_new_serial();
            for (unsigned i = 0; i < (SIDE / 2) * (SIDE / 2); i++) input[i] = 0xff00ff00u;
            b->write_texels(&partial, &selected, &region, (const uint8_t *)input, SIDE * 2, SIDE * SIDE);
            const VkTexImage *retained = vk_texture_get(pin);
            CHECK(retained && retained->image == image, "old serial %llu lost its retained native image %#llx (now %#llx)",
                  (unsigned long long)old.serial, (unsigned long long)(uintptr_t)image,
                  (unsigned long long)(uintptr_t)(retained ? retained->image : VK_NULL_HANDLE));
            uint64_t texel_bytes = 0;
            for (unsigned lv = 0; lv < LEVELS; lv++) {
                unsigned side = SIDE >> lv;
                for (unsigned ly = 0; ly < LAYERS; ly++) {
                    view_t view = selected;
                    view.base_level = lv;
                    view.min_layer = ly;
                    NVNcopyRegion all = {0, 0, 0, side, side, 1};
                    CHECK(!nvn_texels_box_ranges(&old.d, lv, ly, 0, 0, 0, side * 4, side,
                                                pinned_texel_bytes, &texel_bytes), "enumerate actual texture bytes");
                    memset(input, 0, sizeof input);
                    b->read_texels(&partial, &view, &all, (uint8_t *)input, side * 4, side * side * 4);
                    uint32_t expected = lv == 1 && ly == 1 ? 0xff00ff00u : 0xff403010u + lv * LAYERS + ly;
                    for (unsigned i = 0; i < side * side; i++)
                        CHECK(input[i] == expected, "partial replacement level %u layer %u pixel %u read %#x expected %#x",
                              lv, ly, i, input[i], expected);
                }
            }
            const VkTexImage *partial_image = vk_texture_get(&partial);
            CHECK(texel_bytes <= old.L.size, "layout texels exceed total backing");
            printf("probe-vkexec: partial replacement public texels passed, native projection %s, %llu layout padding bytes\n",
                   partial_image && partial_image->image ? "ready" : "blocked", (unsigned long long)(old.L.size - texel_bytes));
            // Packaged data initializes the entire replacement allocation, including the non-texel bytes.
            memset(package, 0x5a, old.L.size);
            for (unsigned lv = 0; lv < LEVELS; lv++) {
                unsigned side = SIDE >> lv;
                for (unsigned ly = 0; ly < LAYERS; ly++) {
                    uint32_t value = lv == 1 && ly == 1 ? 0xff00ff00u : 0xff604020u + lv * LAYERS + ly;
                    for (unsigned i = 0; i < side * side; i++) input[i] = value;
                    CHECK(!nvn_swizzle(&old.d, lv, ly, input, side * 4, package),
                          "swizzle fully initialized replacement level %u layer %u", lv, ly);
                }
            }
            tex_builder_t builder = {.d = old.d, .pool = &pool, .off = old.off, .pkg_data = package,
                                     .has_layout = old.has_layout};
            memcpy(builder.layout, old.layout, sizeof builder.layout);
            CHECK(pinned_clear_proc(nvnp_nvnTextureInitialize, (const uint64_t[8]){(uintptr_t)&replacement, (uintptr_t)&builder}),
                  "NVN packaged texture initialization");
            CHECK(replacement.mem == old.mem && replacement.serial != old.serial && replacement.serial != partial.serial,
                  "packaged replacement must create a distinct serial in the same mapped backing");
            const VkTexImage *next = vk_texture_get(&replacement);
            CHECK(next && next->image, "fully initialized replacement serial %llu did not resolve a native image",
                  (unsigned long long)replacement.serial);
            CHECK(next->image != image, "replacement serial %llu reused still-held native image %#llx",
                  (unsigned long long)replacement.serial, (unsigned long long)(uintptr_t)image);
        }
    }
    for (unsigned lv = 0; lv < LEVELS; lv++) {
        unsigned side = SIDE >> lv;
        for (unsigned ly = 0; ly < LAYERS; ly++) {
            const uint32_t *pixels = (const void *)(output + OUT_STRIDE * (lv * LAYERS + ly));
            for (unsigned i = 0; i < side * side; i++) {
                uint32_t value = 0xff403010u + lv * LAYERS + ly;
                if (lv == 1 && ly == 1) {
                    unsigned x = i % side, y = i / side;
                    value = 0xff0000ffu;
                    if (x >= 4 && x < 12 && y >= 4 && y < 12) value = 0xffff0000u;
                    if (x >= 6 && x < 10 && y >= 6 && y < 10) value = 0xff00ffffu;
                }
                CHECK(pixels[i] == value, "pinned clear level %u layer %u pixel %u read %#x expected %#x",
                      lv, ly, i, pixels[i], value);
            }
        }
    }
    for (unsigned lv = 0; lv < LEVELS; lv++) {
        unsigned side = SIDE >> lv;
        for (unsigned ly = 0; ly < LAYERS; ly++) {
            const uint32_t *pixels = (const void *)(output + OUT_STRIDE * (LEVELS * LAYERS + lv * LAYERS + ly));
            uint32_t expected = lv == 1 && ly == 1 ? 0xff00ff00u : 0xff604020u + lv * LAYERS + ly;
            for (unsigned i = 0; i < side * side; i++)
                CHECK(pixels[i] == expected, "replacement level %u layer %u native pixel %u read %#x expected %#x",
                      lv, ly, i, pixels[i], expected);
        }
    }
    // Let the last native use complete and the texture cross its ordinary victim epoch. The probe drives the
    // same residency detach as a real pressure callback and replays the same explicitly retained transfer.
    PC(nvnCommandBufferAddControlMemory, (uintptr_t)cold_control, sizeof cold_control);
    PC(nvnCommandBufferBeginRecording, 0);
    uint64_t advance = PC(nvnCommandBufferEndRecording, 0);
    CHECK(advance, "record unrelated scene before exact restore eviction");
    uint64_t advance_fence = b->submit(&advance, 1);
    CHECK(b->fence_wait(advance_fence, UINT64_C(5000000000)), "unrelated scene completed before restore eviction");
    uint64_t restore_before = vk_texture_restore_probe_count();
    CHECK(vk_texture_restore_probe_evict(&replacement), "replacement has a reclaimable exact immutable provider");
    memset(output, 0, OUT_STRIDE * 2 * LEVELS * LAYERS);
    b->guest_written((uintptr_t)output, OUT_STRIDE * 2 * LEVELS * LAYERS);
    uint64_t restore_fence = b->submit(&exact_copies, 1);
    CHECK(b->fence_wait(restore_fence, UINT64_C(5000000000)), "exact staged restore and copy completed");
    CHECK(vk_texture_restore_probe_count() == restore_before + 1, "ordered copy recorded exactly one staged restoration");
    for (unsigned lv = 0; lv < LEVELS; lv++) {
        unsigned side = SIDE >> lv;
        for (unsigned ly = 0; ly < LAYERS; ly++) {
            const uint32_t *pixels = (const void *)(output + OUT_STRIDE * (LEVELS * LAYERS + lv * LAYERS + ly));
            uint32_t expected = lv == 1 && ly == 1 ? 0xff00ff00u : 0xff604020u + lv * LAYERS + ly;
            for (unsigned i = 0; i < side * side; i++)
                CHECK(pixels[i] == expected, "restored level %u layer %u native pixel %u read %#x expected %#x",
                      lv, ly, i, pixels[i], expected);
        }
    }
    printf("probe-vkexec: retained nonresident CopyTextureToBuffer source restored all mips and layers\n");
    for (unsigned p = 0; p < PASSES; p++) PC(nvnCommandBufferAddControlMemory, (uintptr_t)control[p], sizeof control[p]);
    PC(nvnCommandBufferAddControlMemory, (uintptr_t)cold_control, sizeof cold_control);
    PC(nvnCommandBufferFinalize, 0);
#undef PC
    pinned_clear_proc(nvnp_nvnTextureFinalize, (const uint64_t[8]){(uintptr_t)&replacement});
    b->pool_fini(&pool);
    printf("probe-vkexec: pinned clear and inherited render target preserved old native pixels through same-address replacement\n");
    return 0;
}

void nvnp_nvnCommandBufferCopyBufferToTexture(Cpu *);
void nvnp_nvnCommandBufferClearBuffer(Cpu *);
void nvnp_nvnCommandBufferSetCopyRowStride(Cpu *);
void nvnp_nvnSyncWait(Cpu *);
uint64_t vk_exec_buffer_to_texture_probe_count(void);

static int test_pinned_buffer_texture(const GfxBackend *b, uint8_t *guest) {
    enum { SIDE = 32, LEVELS = 2, LAYERS = 2, PASSES = 5, STRIDE = 32, OUT_STRIDE = SIDE * SIDE * 4 };
    static _Alignas(16) NVNcommandBuffer command;
    static _Alignas(8) uint8_t control[PASSES][4096];
    pool_t cpu = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xA00000,
                  .size = 0x100000, .game_mem = guest + 0xA00000};
    pool_t device = {.magic = NVN_POOL_MAGIC, .flags = 0x189, .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE,
                     .size = 0x100000, .game_mem = guest + 0xB00000};
    CHECK(b->pool_init(&cpu, &cpu) && b->pool_init(&device, &device), "B2T CPU and native buffer backings");
    tex_t old, replacement = {0};
    init_tex(&old, &cpu, 0, NVN_FORMAT_RGBA8);
    old.serial = nvn_new_serial();
    old.d.target = NVN_TEXTURE_TARGET_2D_ARRAY;
    old.d.width = old.d.height = SIDE;
    old.d.levels = LEVELS;
    old.d.depth = LAYERS;
    nvn_tex_layout(&old.d, &old.L);
    const uint32_t base_old = 0xff403010u, base_new = 0xff604020u, red = 0xff0000ffu, blue = 0xffff0000u;
    uint8_t *output = cpu.mem + 0x10000, *package = cpu.mem + 0x40000;
    memset(package, 0x5a, old.L.size);
    uint32_t staging[SIDE * SIDE];
    for (unsigned lv = 0; lv < LEVELS; lv++) for (unsigned ly = 0; ly < LAYERS; ly++) {
        unsigned side = SIDE >> lv, subresource = lv * LAYERS + ly;
        for (unsigned i = 0; i < side * side; i++) staging[i] = base_old + subresource;
        CHECK(!nvn_swizzle(&old.d, lv, ly, staging, side * 4, old.mem), "B2T swizzle selected prior contents");
        for (unsigned i = 0; i < side * side; i++) staging[i] = base_new + subresource;
        CHECK(!nvn_swizzle(&old.d, lv, ly, staging, side * 4, package), "B2T swizzle replacement package");
    }
    b->guest_written((uintptr_t)old.mem, old.L.size);
    const view_t selected = {.set_levels = 1, .base_level = 1, .nlevels = 1, .set_layers = 1, .min_layer = 1, .nlayers = 1};
    const NVNcopyRegion first = {2, 3, 0, 6, 5, 1}, second = {8, 7, 0, 5, 6, 1};
    const uintptr_t src = (uintptr_t)device.mem + 0x4000;
    VkImage original = VK_NULL_HANDLE, private_image = VK_NULL_HANDLE;
    const tex_t *pin = NULL;
    uint64_t started = vk_exec_buffer_to_texture_probe_count();
#define BX(proc, ...) pinned_clear_proc(nvnp_##proc, (const uint64_t[8]){(uintptr_t)&command, __VA_ARGS__})
    CHECK(BX(nvnCommandBufferInitialize, 0), "B2T command buffer initialization");
    BX(nvnCommandBufferAddCommandMemory, (uintptr_t)&cpu, 0x20000, 0x4000);
    for (unsigned pass = 0; pass < PASSES; pass++) {
        BX(nvnCommandBufferAddControlMemory, (uintptr_t)control[pass], sizeof control[pass]);
        BX(nvnCommandBufferBeginRecording, 0);
        if (pass == 1) {
            BX(nvnCommandBufferSetCopyRowStride, 5);
            BX(nvnCommandBufferCopyBufferToTexture, src, (uintptr_t)&old, (uintptr_t)&selected, (uintptr_t)&first, 0);
            BX(nvnCommandBufferSetCopyRowStride, STRIDE);
            BX(nvnCommandBufferCopyBufferToTexture, (uintptr_t)device.mem + device.size + 0x1000,
               (uintptr_t)&old, (uintptr_t)&selected, (uintptr_t)&first, 0);
        } else if (pass == 2 || pass == 3) {
            const NVNcopyRegion *region = pass == 2 ? &first : &second;
            // This is a real earlier GPU producer in the same ordered submission, with no host staging copy.
            BX(nvnCommandBufferClearBuffer, src, STRIDE * region->height, pass == 2 ? red : blue);
            BX(nvnCommandBufferSetCopyRowStride, STRIDE);
            BX(nvnCommandBufferCopyBufferToTexture, src, (uintptr_t)&old, (uintptr_t)&selected, (uintptr_t)region, 0);
        } else {
            BX(nvnCommandBufferSetCopyRowStride, 0);
            for (unsigned lv = 0; lv < LEVELS; lv++) for (unsigned ly = 0; ly < LAYERS; ly++) {
                view_t view = selected;
                view.base_level = lv;
                view.min_layer = ly;
                NVNcopyRegion all = {0, 0, 0, SIDE >> lv, SIDE >> lv, 1};
                unsigned subresource = lv * LAYERS + ly;
                BX(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&old, (uintptr_t)&view, (uintptr_t)&all,
                   (uintptr_t)(output + OUT_STRIDE * subresource), 0);
                if (pass == PASSES - 1)
                    BX(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&replacement, (uintptr_t)&view, (uintptr_t)&all,
                       (uintptr_t)(output + OUT_STRIDE * (LEVELS * LAYERS + subresource)), 0);
            }
        }
        uint64_t handle = BX(nvnCommandBufferEndRecording, 0);
        CHECK(handle, "B2T recording %u", pass);
        if (pass >= 1 && pass <= 3) {
            unsigned copies = 0;
            for (const op_hdr *op = nvn_op_first(handle); op->op != OP_END; op = nvn_op_next(op)) {
                if (op->op != OP_COPY_B2T) continue;
                const tex_t *identity = ((const p_copy_b2t *)nvn_op_payload(op))->tex;
                if (!pin) pin = identity;
                CHECK(identity && identity == pin && identity->serial == old.serial, "B2T pins the same old serial");
                copies++;
            }
            CHECK(copies == (pass == 1 ? 2 : 1), "B2T operation count");
        }
        uint64_t submitted = b->submit(&handle, 1);
        uint64_t done_after_execution = 0;
        if (pass == 2) {
            CHECK(b->exec_wait(submitted, UINT64_C(5000000000)), "B2T prior command execution");
            done_after_execution = b->fence_done();
        }
        CHECK(b->fence_wait(submitted, UINT64_C(5000000000)), "B2T real submit %u", pass);
        CHECK(vk_exec_buffer_to_texture_probe_count() == started + (pass >= 2) + (pass >= 3),
              "B2T exact recorded copies after phase %u", pass);
        if (pass == 0 || pass == 1) {
            const VkTexImage *image = vk_texture_get(&old);
            CHECK(image && image->image, "B2T warmed immutable image remains resident");
            if (!pass) original = image->image;
            else CHECK(image->image == original && !nvn_texture_pin_attachment(pin), "invalid B2T never acquires its destination");
        } else if (pass == 2) {
            CHECK(nvn_texture_pin_attachment(pin), "first pitched B2T owns the exact writable texture pin");
            const VkTexImage *image = vk_texture_get(pin);
            CHECK(image && image->image != original, "first pitched B2T records native COW");
            private_image = image->image;
            tex_builder_t builder = {.d = old.d, .pool = &cpu, .off = old.off, .pkg_data = package, .has_layout = old.has_layout};
            memcpy(builder.layout, old.layout, sizeof builder.layout);
            // The guest wait also settles the CPU coherence callback for completed small image writes.
            sync_t completed = {.fenced = 1, .fence = submitted};
            CHECK(pinned_clear_proc(nvnp_nvnSyncWait, (const uint64_t[8]){(uintptr_t)&completed, 0}) ==
                NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED, "B2T guest synchronization of completed scene");
            uint64_t done_before_store = b->fence_done();
            CHECK(done_before_store >= submitted, "B2T prior CPU-coherent completion");
            CHECK(pinned_clear_proc(nvnp_nvnTextureInitialize, (const uint64_t[8]){(uintptr_t)&replacement, (uintptr_t)&builder}),
                  "B2T fixture packaged same-VA replacement");
            CHECK(replacement.mem == old.mem && replacement.L.size == old.L.size &&
                  !memcmp(replacement.mem, package, old.L.size), "B2T replacement bytes stored before native resolution");
            const VkTexImage *current = vk_texture_get(&replacement);
            printf("probe-vkexec: B2T replacement scene %llu completed fence after execution %llu before store %llu\n",
                   (unsigned long long)submitted, (unsigned long long)done_after_execution, (unsigned long long)done_before_store);
            CHECK(replacement.mem == old.mem && replacement.serial != pin->serial && current && current->image &&
                  current->image != private_image,
                  "new native texture projection coexists with the older pinned B2T destination");
        } else if (pass == 3) {
            const VkTexImage *image = vk_texture_get(pin);
            CHECK(image && image->image == private_image, "detached B2T kept the original private image");
        }
    }
#undef BX
    b->guest_read((uintptr_t)output, 2 * LEVELS * LAYERS * OUT_STRIDE);
    unsigned checked = 0;
    for (unsigned lv = 0; lv < LEVELS; lv++) for (unsigned ly = 0; ly < LAYERS; ly++) {
        unsigned side = SIDE >> lv, subresource = lv * LAYERS + ly;
        const uint32_t *before = (const uint32_t *)(output + OUT_STRIDE * subresource);
        const uint32_t *after = (const uint32_t *)(output + OUT_STRIDE * (LEVELS * LAYERS + subresource));
        for (unsigned y = 0; y < side; y++) for (unsigned x = 0; x < side; x++) {
            uint32_t expected = base_old + subresource;
            if (lv == 1 && ly == 1) {
                if (x >= 2 && x < 8 && y >= 3 && y < 8) expected = red;
                if (x >= 8 && x < 13 && y >= 7 && y < 13) expected = blue;
            }
            unsigned i = y * side + x;
            CHECK(before[i] == expected && after[i] == base_new + subresource,
                  "B2T level %u layer %u pixel %u,%u old/new %#x/%#x expected %#x/%#x",
                  lv, ly, x, y, before[i], after[i], expected, base_new + subresource);
            checked += 2;
        }
    }
    for (unsigned p = 0; p < PASSES; p++)
        pinned_clear_proc(nvnp_nvnCommandBufferAddControlMemory,
                          (const uint64_t[8]){(uintptr_t)&command, (uintptr_t)control[p], sizeof control[p]});
    pinned_clear_proc(nvnp_nvnCommandBufferFinalize, (const uint64_t[8]){(uintptr_t)&command});
    pinned_clear_proc(nvnp_nvnTextureFinalize, (const uint64_t[8]){(uintptr_t)&replacement});
    b->pool_fini(&device);
    b->pool_fini(&cpu);
    printf("probe-vkexec: exact pitched B2T recorded two GPU-backed copies and preserved %u old/new pixels\n", checked);
    return 0;
}

void nvnp_nvnCommandBufferCopyTextureToTexture(Cpu *);
void nvnp_nvnCommandBufferDownsample(Cpu *);
uint64_t vk_exec_texture_to_texture_probe_count(uint64_t *staged);

static int test_ordered_texture_copy(const GfxBackend *b, uint8_t *guest) {
    enum { SIDE = 32, SMALL = 16, PASSES = 7, TEXTURES = 7, PLANE = SIDE * SIDE };
    static _Alignas(16) NVNcommandBuffer command;
    static _Alignas(8) uint8_t control[PASSES][4096];
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xA00000,
                   .size = 0x100000, .game_mem = guest + 0xA00000};
    CHECK(b->pool_init(&pool, &pool), "T2T fixture imported backing");
    tex_t textures[TEXTURES];
    uint32_t expected[PLANE], temporary[PLANE];
    const uint32_t cold_color = 0xffe5b27fu, paired_color = 0xffa2d845u, seeded_color = 0xff7253c8u;
    for (unsigned i = 0; i < TEXTURES; i++) {
        init_tex(&textures[i], &pool, i * 0x10000, NVN_FORMAT_RGBA8);
        textures[i].serial = nvn_new_serial();
        unsigned side = i == 1 || i == 4 || i == 6 ? SMALL : SIDE;
        textures[i].d.width = textures[i].d.height = side;
        nvn_tex_layout(&textures[i].d, &textures[i].L);
        for (unsigned y = 0; y < side; y++) for (unsigned x = 0; x < side; x++) {
            uint32_t value = i == 2 ? cold_color : i == 3 ? paired_color : i == 5 ? seeded_color :
                             0xff007040u + ((y * side + x) << 5);
            temporary[y * side + x] = value;
        }
        if (!i) memcpy(expected, temporary, sizeof expected);
        CHECK(!nvn_swizzle(&textures[i].d, 0, 0, temporary, side * 4, textures[i].mem), "T2T source swizzle %u", i);
        b->guest_written((uintptr_t)textures[i].mem, textures[i].L.size);
    }
    uint32_t *result = (uint32_t *)(pool.mem + 0xa0000);
    const NVNcopyRegion boxes[3] = {{2, 3, 0, 7, 6, 1}, {4, 4, 0, 7, 6, 1}, {6, 5, 0, 7, 6, 1}};
    const NVNcopyRegion whole = {0, 0, 0, SIDE, SIDE, 1}, half = {0, 0, 0, SMALL, SMALL, 1};
    const tex_t *pin = NULL;
    VkImage original = VK_NULL_HANDLE, writable = VK_NULL_HANDLE, paired_original = VK_NULL_HANDLE;
    uint64_t staged, count = vk_exec_texture_to_texture_probe_count(&staged), scratch_base = staged;
    uint64_t restored = vk_texture_restore_probe_count();
#define TT(proc, ...) pinned_clear_proc(nvnp_##proc, (const uint64_t[8]){(uintptr_t)&command, __VA_ARGS__})
    CHECK(TT(nvnCommandBufferInitialize, 0), "T2T command buffer initialization");
    TT(nvnCommandBufferAddCommandMemory, (uintptr_t)&pool, 0x80000, 0x4000);
    for (unsigned phase = 0; phase < PASSES; phase++) {
        if (phase >= 4) {
            unsigned source = phase == 6 ? 5 : phase - 2;
            CHECK(vk_texture_restore_probe_evict(&textures[source]), "detach the cold source for T2T phase %u", phase);
        }
        if (phase == 6)
            CHECK(vk_texture_restore_probe_evict(&textures[6]), "detach the old destination image before private seeding");
        TT(nvnCommandBufferAddControlMemory, (uintptr_t)control[phase], sizeof control[phase]);
        TT(nvnCommandBufferBeginRecording, 0);
        if (!phase) {
            for (unsigned i = 0; i < TEXTURES; i++)
                TT(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&textures[i], 0,
                   (uintptr_t)(i == 1 || i == 4 || i == 6 ? &half : &whole),
                   (uintptr_t)(result + i * PLANE), 0);
        } else if (phase <= 2) {
            const NVNcopyRegion *s = &boxes[phase - 1], *d = &boxes[phase];
            TT(nvnCommandBufferCopyTextureToTexture, (uintptr_t)&textures[0], 0, (uintptr_t)s,
               (uintptr_t)&textures[0], 0, (uintptr_t)d, 0);
            memcpy(temporary, expected, sizeof temporary);
            for (int y = 0; y < s->height; y++) for (int x = 0; x < s->width; x++)
                expected[(d->yoffset + y) * SIDE + d->xoffset + x] = temporary[(s->yoffset + y) * SIDE + s->xoffset + x];
        } else if (phase <= 4) {
            unsigned source = phase == 3 ? 0 : 2;
            TT(nvnCommandBufferDownsample, (uintptr_t)&textures[source], (uintptr_t)&textures[1]);
            TT(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&textures[1], 0, (uintptr_t)&half,
               (uintptr_t)(result + PLANE * (phase - 2)), 0);
            if (phase == 3)
                TT(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&textures[0], 0, (uintptr_t)&whole, (uintptr_t)result, 0);
        } else {
            unsigned source = phase == 5 ? 3 : 5, target = source + 1;
            TT(nvnCommandBufferCopyTextureToTexture, (uintptr_t)&textures[source], 0, (uintptr_t)&boxes[0],
               (uintptr_t)&textures[target], 0, (uintptr_t)&boxes[1], 0);
            TT(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&textures[target], 0, (uintptr_t)&half,
               (uintptr_t)(result + PLANE * (phase - 2)), 0);
        }
        uint64_t handle = TT(nvnCommandBufferEndRecording, 0);
        CHECK(handle, "T2T phase %u recorded", phase);
        if (phase == 1 || phase == 2) {
            const op_hdr *header = nvn_op_next(nvn_op_first(handle));
            CHECK(header->op == OP_COPY_T2T, "T2T self-copy operation kind");
            const p_copy_t2t *copy = nvn_op_payload(header);
            if (!pin) pin = copy->src;
            CHECK(copy->src == pin && copy->dst == pin, "self-copy targets the same recorded pin");
        }
        CHECK(b->fence_wait(b->submit(&handle, 1), UINT64_C(5000000000)), "T2T phase %u completed", phase);
        CHECK(vk_exec_texture_to_texture_probe_count(&staged) == count + phase &&
              staged == scratch_base + (phase >= 2), "T2T phase %u actual ordered copy/scratch count", phase);
        if (!phase) {
            const VkTexImage *image = vk_texture_get(&textures[0]);
            CHECK(image && image->image, "T2T warm original native image");
            original = image->image;
            image = vk_texture_get(&textures[4]);
            CHECK(image && image->image, "T2T paired destination starts with immutable contents");
            paired_original = image->image;
        } else if (phase == 1 || phase == 2) {
            CHECK(nvn_texture_pin_attachment(pin), "T2T holds its actual native version on the command pin");
            const VkTexImage *image = vk_texture_get(pin);
            CHECK(image && image->image && image->image != original, "first T2T allocated a private image");
            if (phase == 1) writable = image->image;
            else CHECK(image->image == writable, "overlapping T2T modifies the same established private image");
        } else if (phase >= 5) {
            const op_hdr *header = nvn_op_next(nvn_op_first(handle));
            CHECK(header->op == OP_COPY_T2T, "T2T paired copy operation kind");
            const p_copy_t2t *copy = nvn_op_payload(header);
            CHECK(nvn_texture_pin_attachment(copy->dst), "T2T paired copy owns its first private destination");
            const VkTexImage *image = vk_texture_get(copy->dst);
            CHECK(image && image->image && (phase == 6 || image->image != paired_original),
                  "T2T phase %u consumed its private allocation", phase);
        }
    }
    CHECK(vk_texture_restore_probe_count() == restored + 3, "cold Downsample and paired copies restored three exact sources");
    b->guest_read((uintptr_t)result, 5 * PLANE * 4);
    for (unsigned i = 0; i < PLANE; i++)
        CHECK(result[i] == expected[i], "overlapping T2T pixel %u read %#x expected %#x", i, result[i], expected[i]);
    for (unsigned y = 0; y < SMALL; y++) for (unsigned x = 0; x < SMALL; x++) {
        unsigned i = y * SMALL + x, src = y * 2 * SIDE + x * 2;
        const unsigned offsets[4] = {0, 1, SIDE, SIDE + 1};
        for (unsigned shift = 0; shift < 32; shift += 8) {
            int sum = 0;
            for (unsigned p = 0; p < 4; p++) sum += (expected[src + offsets[p]] >> shift) & 255;
            int actual = (result[PLANE + i] >> shift) & 255;
            CHECK(abs(actual - (sum + 2) / 4) <= 1, "filtered T2T pixel %u channel %u read %d", i, shift / 8, actual);
        }
        CHECK(result[2 * PLANE + i] == cold_color, "cold Downsample pixel %u read %#x", i, result[2 * PLANE + i]);
        const NVNcopyRegion *d = &boxes[1];
        int overwritten = x >= (unsigned)d->xoffset && x < (unsigned)(d->xoffset + d->width) &&
                          y >= (unsigned)d->yoffset && y < (unsigned)(d->yoffset + d->height);
        uint32_t prior = 0xff007040u + (i << 5);
        uint32_t paired = overwritten ? paired_color : prior;
        CHECK(result[3 * PLANE + i] == paired, "paired T2T pixel %u read %#x expected %#x", i, result[3 * PLANE + i], paired);
        uint32_t seeded = overwritten ? seeded_color : prior;
        CHECK(result[4 * PLANE + i] == seeded, "directly seeded T2T pixel %u read %#x expected %#x", i, result[4 * PLANE + i], seeded);
    }
    for (unsigned i = 0; i < PASSES; i++) TT(nvnCommandBufferAddControlMemory, (uintptr_t)control[i], sizeof control[i]);
    TT(nvnCommandBufferFinalize, 0);
#undef TT
    for (unsigned i = 0; i < TEXTURES; i++) b->texture_release(&textures[i]);
    b->pool_fini(&pool);
    printf("probe-vkexec: exact T2T used one overlap scratch and three source restorations, including a directly seeded cold destination; checked %u pixels\n",
           PLANE + 4 * SMALL * SMALL);
    return 0;
}

enum { BC_SIDE = 32, BC_DEPTH = 4, BC_LEVELS = 2, BC_BYTES = 2304 };
typedef struct { uint32_t lo, hi; } ProbeBlock;
static const ProbeBlock bc_clear_block = {0x44332211, 0x88776655};

typedef struct {
    tex_t *old;
    tex_t replacement;
    const GfxBackend *backend;
    tex_builder_t builder;
    VkBuffer readback;
    VkImage newer;
    int result;
} BcPassProbe;

static ProbeBlock bc_probe_block(unsigned version, unsigned level, unsigned plane, unsigned x, unsigned y) {
    return (ProbeBlock){0x41420000u + version * 0x10000u + (level << 12) + (plane << 8) + (y << 4) + x,
                        0x25260000u + version * 0x20000u + (level << 12) + (plane << 8) + (y << 4) + x};
}

static int bc_probe_package(const tex_t *texture, unsigned version, uint8_t *package) {
    ProbeBlock blocks[BC_DEPTH][BC_SIDE / 4][BC_SIDE / 4];
    memset(package, 0xab, texture->L.size);
    for (unsigned level = 0; level < BC_LEVELS; ++level) {
        unsigned rows = BC_SIDE / 4 >> level, depth = BC_DEPTH >> level;
        ProbeBlock *linear = &blocks[0][0][0];
        for (unsigned plane = 0; plane < depth; ++plane)
            for (unsigned y = 0; y < rows; ++y)
                for (unsigned x = 0; x < rows; ++x)
                    linear[(plane * rows + y) * rows + x] = bc_probe_block(version, level, plane, x, y);
        if (nvn_swizzle(&texture->d, level, 0, linear, rows * sizeof *linear, package)) return 0;
    }
    return 1;
}

static int bc_replace_while_parked(void *context) {
    BcPassProbe *probe = context;
    tex_t *old = probe->old;
    vk_texture_range_gone((uintptr_t)old->mem, old->L.size, NULL);
    if (!pinned_clear_proc(nvnp_nvnTextureInitialize,
                           (const uint64_t[8]){(uintptr_t)&probe->replacement, (uintptr_t)&probe->builder})) return 0;
    const VkTexImage *replacement = vk_texture_get(&probe->replacement);
    if (!replacement || probe->replacement.mem != old->mem || probe->replacement.serial == old->serial) return 0;
    probe->newer = replacement->image;
    return probe->newer != VK_NULL_HANDLE;
}

static int record_bc_pass(VkExec *ex, BcPassProbe *probe) {
    VkExecGraphicsPass *pass __attribute__((cleanup(vk_exec_graphics_pass_cancel))) = NULL;
    CHECK(vk_exec_graphics_pass_capture(ex, 0, &pass) == VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED && !pass,
          "cold BC attachment requests the ordered owner");
    CHECK(vk_exec_graphics_pass_capture(ex, 1, &pass) == VK_EXEC_GRAPHICS_PASS_READY && pass,
          "ordered BC attachment captured the actual stand-in");
    const VkExecGraphicsPassShape *shape = vk_exec_graphics_pass_shape(pass);
    const VkExecTarget *target = &shape->color[0];
    VkImage old = target->bc_backing, standin = target->image;
    CHECK(old && standin && old != standin && target->level == 0 && target->bc_level == 1 &&
          target->base_layer == 1 && target->layers == 1 && shape->pass_w == 4 && shape->pass_h == 4 &&
          shape->pipe_color_fmt[0] == NVN_FORMAT_RG32UI && target->format == VK_FORMAT_R32G32_UINT,
          "3D BC key describes the actual block view and original mip/plane");
    CHECK(vk_sync_progress_prepare_pipeline(bc_replace_while_parked, probe) == VK_SYNC_PROGRESS_COMPLETE &&
          probe->newer && probe->newer != old && probe->newer != standin,
          "same-address replacement made a new native image while the recorder parked");
    CHECK(vk_exec_graphics_pass_current(ex, pass) && vk_exec_graphics_pass_activate(ex, pass) &&
          ex->in_pass && ex->color[0].image == standin && ex->color[0].bc_backing == old &&
          ex->target_bindings[0].private_detached && !ex->bc_written_mask,
          "resumed BC pass retained the old native target, with no guest writer on LOAD");
    VkClearAttachment clear = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
        .clearValue.color.uint32 = {bc_clear_block.lo, bc_clear_block.hi, 0, 0}};
    VkClearRect rect = {{{1, 1}, {2, 2}}, 0, 1};
    vkCmdClearAttachments(ex->cb, 1, &clear, 1, &rect);
    ex->recorded++;
    vk_exec_targets_written(ex);
    vk_exec_barrier(ex);
    for (unsigned version = 0; version < 2; ++version) {
        VkBufferImageCopy copy[BC_LEVELS];
        VkDeviceSize offset = version * BC_BYTES;
        for (unsigned level = 0; level < BC_LEVELS; ++level) {
            unsigned side = BC_SIDE >> level, depth = BC_DEPTH >> level;
            copy[level] = (VkBufferImageCopy){.bufferOffset = offset,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1}, .imageExtent = {side, side, depth}};
            offset += (side / 4) * (side / 4) * depth * sizeof(ProbeBlock);
        }
        vkCmdCopyImageToBuffer(ex->cb, version ? probe->newer : old, VK_IMAGE_LAYOUT_GENERAL, probe->readback,
                               BC_LEVELS, copy);
        ex->recorded++;
    }
    const VkMemoryBarrier2 host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    const VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1, .pMemoryBarriers = &host};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    return 0;
}

static void run_bc_pass(VkExec *ex, void *context) {
    BcPassProbe *probe = context;
    probe->result = record_bc_pass(ex, probe);
}

static int test_bc_pass(const GfxBackend *b, uint8_t *guest) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000, .size = 0x100000,
                   .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool, &pool), "pool_init of the ordered BC pass fixture");
    tex_t old;
    init_tex(&old, &pool, 0, NVN_FORMAT_RGTC1_UNORM);
    old.d.target = NVN_TEXTURE_TARGET_3D;
    old.d.width = old.d.height = BC_SIDE;
    old.d.depth = BC_DEPTH;
    old.d.levels = BC_LEVELS;
    old.serial = nvn_new_serial();
    nvn_tex_layout(&old.d, &old.L);
    CHECK(old.L.size <= 0x20000, "BC backing fits its independent fixture range");
    CHECK(bc_probe_package(&old, 0, old.mem), "pack old 3D BC blocks");
    b->guest_written((uintptr_t)old.mem, old.L.size);
    uint8_t *package = pool.mem + 0x20000;
    CHECK(bc_probe_package(&old, 1, package), "pack replacement 3D BC blocks");
    BcPassProbe probe = {.old = &old, .backend = b,
        .builder = {.d = old.d, .pool = &pool, .off = old.off, .pkg_data = package}, .result = 1};
    VkBufferCreateInfo buffer = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2 * BC_BYTES,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo placement = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocator allocator = vk_ctx()->vma;
    VmaAllocation allocation;
    VmaAllocationInfo mapped;
    CHECK(vmaCreateBuffer(allocator, &buffer, &placement, &probe.readback, &allocation, &mapped) == VK_SUCCESS &&
          mapped.pMappedData, "allocate host readback for both BC native owners");
    p_set_rt rt = {.n = 1, .color = {&old}, .has_cview = {1},
        .cview = {{.set_format = 1, .format = NVN_FORMAT_RG32UI, .set_levels = 1, .base_level = 1, .nlevels = 1,
                   .set_layers = 1, .min_layer = 1, .nlayers = 1}}};
    _Alignas(8) uint8_t stream[sizeof(rt) + 64];
    uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
    next = put(next, OP_SET_RT, &rt, sizeof rt);
    put(next, OP_END, NULL, 0);
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    vk_exec_probe_pass_arm(run_bc_pass, &probe);
    uint64_t fence = b->submit(&handle, 1);
    CHECK(b->fence_wait(fence, UINT64_C(5000000000)) && !probe.result, "ordered 3D BC scene completed");
    CHECK(vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE) == VK_SUCCESS, "invalidate BC readback");
    const ProbeBlock *actual = mapped.pMappedData;
    unsigned compared = 0;
    for (unsigned version = 0; version < 2; ++version)
        for (unsigned level = 0; level < BC_LEVELS; ++level) {
            unsigned rows = BC_SIDE / 4 >> level, depth = BC_DEPTH >> level;
            for (unsigned plane = 0; plane < depth; ++plane)
                for (unsigned y = 0; y < rows; ++y)
                    for (unsigned x = 0; x < rows; ++x, ++compared) {
                        ProbeBlock expected = bc_probe_block(version, level, plane, x, y);
                        if (!version && level == 1 && plane == 1 && x >= 1 && x <= 2 && y >= 1 && y <= 2)
                            expected = bc_clear_block;
                        CHECK(actual[compared].lo == expected.lo && actual[compared].hi == expected.hi,
                              "BC version %u mip %u plane %u block %u,%u was %08x/%08x expected %08x/%08x",
                              version, level, plane, x, y, actual[compared].lo, actual[compared].hi, expected.lo, expected.hi);
                    }
        }
    CHECK(compared * sizeof *actual == 2 * BC_BYTES, "compared all blocks from both native allocations");
    vmaDestroyBuffer(allocator, probe.readback, allocation);
    pinned_clear_proc(nvnp_nvnTextureFinalize, (const uint64_t[8]){(uintptr_t)&probe.replacement});
    b->pool_fini(&pool);
    printf("probe-vkexec: ordered BC stand-in preserved exact old mip/plane copyback through a parked same-address replacement (%u blocks)\n",
           compared);
    return 0;
}

void vk_exec_probe_shader(const uint8_t *control, const GfxShader *shader);

static int test_sample_restore(const GfxBackend *b, uint8_t *guest) {
    enum { SIDE = 32, TEX_SLOT = 17, SAMP_SLOT = 3 };
    const GfxShader *vs = gfx_shader_own(hdr_fullscreen_spv, sizeof hdr_fullscreen_spv / sizeof *hdr_fullscreen_spv,
        "{\"key\":\"sample-restore-vs\",\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
    const GfxShader *fs = gfx_shader_own(sample_restore_probe_spv, sizeof sample_restore_probe_spv / sizeof *sample_restore_probe_spv,
        "{\"key\":\"sample-restore-fs\",\"stage\":\"ps\",\"cbufs\":[],\"sbufs\":[],\"images\":[],"
        "\"textures\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":8,\"type\":3}]}");
    CHECK(vs && fs, "owned fullscreen and sampled shader modules");
    static uint8_t control[2][0x880], code[2][0x80];
    prog_t program = {.magic = NVN_PROG_MAGIC, .serial = nvn_new_serial(), .nstage = 2};
    for (uint32_t stage = 0; stage < 2; ++stage) {
        uint32_t size = sizeof code[stage];
        memcpy(control[stage] + 0x704, &size, sizeof size);
        memcpy(control[stage] + 0x714, &stage, sizeof stage);
        program.sd[stage] = (NVNshaderData){.data = (uintptr_t)code[stage], .control = control[stage]};
        vk_exec_probe_shader(control[stage], stage ? fs : vs);
    }
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000,
                   .size = 0x100000, .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool, &pool), "ordinary imported sample and output pool");
    tex_t source, output;
    init_tex(&source, &pool, 0, NVN_FORMAT_RGBA8);
    init_tex(&output, &pool, 0x10000, NVN_FORMAT_RGBA8);
    source.serial = nvn_new_serial(); output.serial = nvn_new_serial();
    source.d.width = source.d.height = output.d.width = output.d.height = SIDE;
    nvn_tex_layout(&source.d, &source.L);
    nvn_tex_layout(&output.d, &output.L);
    uint32_t expected[SIDE * SIDE], warm[SIDE * SIDE], cold[SIDE * SIDE];
    for (unsigned i = 0; i < SIDE * SIDE; ++i)
        expected[i] = 0xff205000u + 7 * (i % SIDE);
    CHECK(!nvn_swizzle(&source.d, 0, 0, expected, SIDE * sizeof *expected, source.mem), "write exact source texels");
    b->guest_written((uintptr_t)source.mem, source.L.size);
    vk_texture_register(TEX_SLOT, &source, NULL);
    const samp_par sampler = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
        .ws = NVN_WRAP_MODE_REPEAT, .wt = NVN_WRAP_MODE_REPEAT, .wr = NVN_WRAP_MODE_REPEAT, .lod_max = 1};
    vk_sampler_register(SAMP_SLOT, &sampler);

    _Alignas(8) uint8_t stream[4096];
    uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
    const p_set_rt targets = {.n = 1, .color = {&output}};
    next = put(next, OP_SET_RT, &targets, sizeof targets);
    const p_clear_color clear = {0, 0xf, {0, 0, 0, 1}};
    next = put(next, OP_CLEAR_COLOR, &clear, sizeof clear);
    p_viewports viewport = {.n = 1};
    viewport.v[0][2] = viewport.v[0][3] = SIDE; viewport.v[0][5] = 1.0f;
    next = put(next, OP_VIEWPORTS, &viewport, sizeof viewport);
    const p_scissor scissor = {0, 0, SIDE, SIDE};
    next = put(next, OP_SCISSOR, &scissor, sizeof scissor);
    const p_bind_program bind = {&program, 3};
    next = put(next, OP_BIND_PROGRAM, &bind, sizeof bind);
    const p_bind_handles sample = {.stage = NVN_STAGE_FS, .n = 1, .h = {TEX_SLOT | ((uint64_t)SAMP_SLOT << 20)}};
    next = put(next, OP_BIND_TEX, &sample, sizeof sample);
    const p_draw draw = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
    next = put(next, OP_DRAW, &draw, sizeof draw);
    put(next, OP_END, NULL, 0);
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    const NVNcopyRegion full = {0, 0, 0, SIDE, SIDE, 1};
    uint64_t fence = b->submit(&handle, 1);
    CHECK(b->fence_wait(fence, UINT64_C(5000000000)), "first actual sampled draw completed");
    b->read_texels(&output, NULL, &full, (uint8_t *)warm, SIDE * sizeof *warm, sizeof warm);
    for (unsigned i = 0; i < SIDE * SIDE; ++i)
        CHECK(warm[i] == expected[i], "warm sample pixel %u was %08x, expected %08x", i, warm[i], expected[i]);

    const op_hdr empty[2] = {{OP_BEGIN, sizeof(op_hdr)}, {OP_END, sizeof(op_hdr)}};
    uint64_t advance = (uint64_t)(uintptr_t)empty;
    fence = b->submit(&advance, 1);
    CHECK(b->fence_wait(fence, UINT64_C(5000000000)), "earlier sample scene and allocator use retired");
    uint64_t before = vk_texture_restore_probe_count();
    CHECK(vk_texture_restore_probe_evict(&source), "registered ordinary source had reclaimable physical residency");
    fence = b->submit(&handle, 1);
    CHECK(b->fence_wait(fence, UINT64_C(5000000000)), "same actual draw completed after native eviction");
    CHECK(vk_texture_restore_probe_count() == before + 1, "prepared direct sample issued one staged restoration");
    b->read_texels(&output, NULL, &full, (uint8_t *)cold, SIDE * sizeof *cold, sizeof cold);
    CHECK(!memcmp(warm, cold, sizeof warm), "all forced-cold output pixels equal the known warm sampled texels");
    b->texture_release(&source);
    b->texture_release(&output);
    b->pool_fini(&pool);
    printf("probe-vkexec: retained direct sampled texture restored once and rendered %u exact warm/cold pixels\n", SIDE * SIDE);
    return 0;
}

VkImage vk_exec_pool_probe_member(uint32_t kind, uint32_t logical, int release_activity);
uint64_t vk_exec_pool_probe_generation(uint64_t *exact_dispatches);
uint64_t vk_exec_pool_probe_finite_bindings(void);

typedef struct {
    uint8_t control[2][0x880], code[2][0x80];
    prog_t program;
} ProbeSampleProgram;

static void probe_sample_program(ProbeSampleProgram *definition, const GfxShader *vs, const GfxShader *fs) {
    definition->program = (prog_t){.magic = NVN_PROG_MAGIC, .serial = nvn_new_serial(), .nstage = 2};
    for (uint32_t stage = 0; stage < 2; stage++) {
        uint8_t *control = definition->control[stage];
        uint32_t size = sizeof definition->code[stage];
        memcpy(control + 0x704, &size, sizeof size);
        memcpy(control + 0x714, &stage, sizeof stage);
        definition->program.sd[stage] = (NVNshaderData){.data = (uintptr_t)definition->code[stage], .control = control};
        vk_exec_probe_shader(control, stage ? fs : vs);
    }
}

static int probe_sample_draw(const GfxBackend *b, const prog_t *program, tex_t *target, uint64_t bound_handle,
                             uint64_t pool_indices, const p_copy_b2b *seed, const p_bind_buf *storage) {
    _Alignas(8) uint8_t stream[4096];
    uint32_t width = target->d.width, height = target->d.height;
    uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
    if (seed) next = put(next, OP_COPY_B2B, seed, sizeof *seed);
    const p_set_rt attachments = {.n = 1, .color = {target}};
    next = put(next, OP_SET_RT, &attachments, sizeof attachments);
    const p_clear_color clear = {0, 0xf, {0, 0, 0, 1}};
    next = put(next, OP_CLEAR_COLOR, &clear, sizeof clear);
    p_viewports viewport = {.n = 1};
    viewport.v[0][2] = (float)width;
    viewport.v[0][3] = (float)height;
    viewport.v[0][5] = 1.0f;
    next = put(next, OP_VIEWPORTS, &viewport, sizeof viewport);
    const p_scissor scissor = {0, 0, width, height};
    next = put(next, OP_SCISSOR, &scissor, sizeof scissor);
    const p_bind_program bind = {program, 3};
    next = put(next, OP_BIND_PROGRAM, &bind, sizeof bind);
    if (pool_indices) {
        const p_bind_buf uniforms = {NVN_STAGE_FS, 1, pool_indices, 16};
        next = put(next, OP_BIND_UBO, &uniforms, sizeof uniforms);
    }
    if (storage) next = put(next, OP_BIND_SSBO, storage, sizeof *storage);
    if (bound_handle) {
        const p_bind_handles sample = {.stage = NVN_STAGE_FS, .n = 1, .h = {bound_handle}};
        next = put(next, OP_BIND_TEX, &sample, sizeof sample);
    }
    const p_draw draw = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
    next = put(next, OP_DRAW, &draw, sizeof draw);
    put(next, OP_END, NULL, 0);
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    return b->fence_wait(b->submit(&handle, 1), UINT64_C(5000000000));
}

static int test_graphics_device_pointer(const GfxBackend *b, uint8_t *guest) {
    enum { SIDE=32, PAGE=0x10000 };
    const GfxShader *vs = gfx_shader_own(hdr_fullscreen_spv, sizeof hdr_fullscreen_spv / sizeof *hdr_fullscreen_spv,
        "{\"key\":\"device-pointer-vs\",\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
    const GfxShader *fs = gfx_shader_own(device_pointer_draw_spv, sizeof device_pointer_draw_spv / sizeof *device_pointer_draw_spv,
        "{\"key\":\"device-pointer-fs\",\"stage\":\"ps\",\"cbufs\":[],\"textures\":[],\"images\":[],"
        "\"sbufs\":[{\"binding\":1,\"slot\":0,\"cb_slot\":0,\"cb_offset\":328,\"flags\":0,\"ptr\":0}]}");
    CHECK(vs && fs, "own fullscreen and readonly pointer-selected shader");
    ProbeSampleProgram program = {0};
    probe_sample_program(&program, vs, fs);

    pool_t host={.magic=NVN_POOL_MAGIC,.flags=0x22,.mem=guest+0xc00000,.game_mem=guest+0xc00000,.size=0x100000};
    pool_t physical={.magic=NVN_POOL_MAGIC,.flags=0x189,.mem=(uint8_t *)(uintptr_t)RT_GPU_VA_BASE,
                     .game_mem=guest+0xa00000,.size=2*PAGE};
    pool_t virtual={.magic=NVN_POOL_MAGIC,.flags=0x2a1,.mem=(uint8_t *)(uintptr_t)(RT_GPU_VA_BASE+2*PAGE),
                    .size=PAGE,.span=PAGE};
    CHECK(b->pool_init(&host,&host) && b->pool_init(&physical,&physical) && b->pool_init(&virtual,&virtual),
          "make host result and CPU-invisible graphics pointer table");
    tex_t target;
    init_tex(&target, &host, 0, NVN_FORMAT_RGBA8);
    target.serial=nvn_new_serial();
    target.d.width=target.d.height=SIDE;
    nvn_tex_layout(&target.d, &target.L);
    const NVNcopyRegion full={0,0,0,SIDE,SIDE,1};
    const float colors[2][4]={{1,0,0,1},{0,1,0,1}};
    const uint32_t expected[2]={0xff0000ff,0xff00ff00};
    const p_bind_buf table={.stage=NVN_STAGE_FS,.index=1,.addr=(uintptr_t)virtual.mem,.size=8};
    uint32_t pixels[SIDE*SIDE];
    for (unsigned phase=0; phase<2; phase++) {
        const nvn_mapping_t map={(const NVNmemoryPool *)(const void *)&physical, (int64_t)phase*PAGE, 0, PAGE};
        CHECK(b->pool_map_virtual(&virtual, 1, &map), "set graphics pointer mapping %u", phase);
        uint8_t *payload=host.mem+0x20000+phase*PAGE;
        uint64_t selected=(uintptr_t)payload;
        memcpy(payload, colors[phase], sizeof colors[phase]);
        memcpy(host.mem+0x50000, &selected, sizeof selected);
        b->guest_written(selected, sizeof colors[phase]);
        b->guest_written((uintptr_t)(host.mem+0x50000), sizeof selected);
        const p_copy_b2b seed={(uintptr_t)(host.mem+0x50000), (uintptr_t)virtual.mem, sizeof selected};
        CHECK(probe_sample_draw(b,&program.program,&target,0,0,&seed,&table), "ordered graphics pointer draw %u completed", phase);
        b->read_texels(&target,NULL,&full,(uint8_t *)pixels,SIDE*sizeof *pixels,sizeof pixels);
        for (unsigned i=0; i<SIDE*SIDE; i++)
            CHECK(pixels[i]==expected[phase], "native fragment pointer %u pixel %u was %08x, expected %08x",
                  phase,i,pixels[i],expected[phase]);
    }
    b->texture_release(&target);
    b->pool_fini(&virtual);
    b->pool_fini(&physical);
    b->pool_fini(&host);
    puts("probe-vkexec: actual fragment pointer table remap produced 1024 exact pixels per mapping");
    return 0;
}

static int test_pool_restore(const GfxBackend *b, uint8_t *guest, int finite) {
    enum { SIDE = 32, LEFT_SLOT = 17, RIGHT_SLOT = 18, COLD_SLOT = 19, SAMP_SLOT = 3, KIND_2D = 2 };
    const uint32_t left = 0xff285010, right = 0xff705020, replacement = 0xff2840db;
    const GfxShader *vs = gfx_shader_own(hdr_fullscreen_spv, sizeof hdr_fullscreen_spv / sizeof *hdr_fullscreen_spv,
        "{\"key\":\"pool-restore-vs\",\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
    char reflection[512];
    snprintf(reflection, sizeof reflection,
        "{\"key\":\"pool-restore-%s\",\"stage\":\"ps\",\"cbufs\":[%s],\"sbufs\":[],\"images\":[],\"textures\":["
        "{\"binding\":1,\"type\":3,\"array\":32768,\"pool\":1,\"separate\":true},"
        "{\"binding\":2,\"array\":4096,\"pool\":2,\"separate\":true}]}",
        finite ? "finite" : "per-invocation", finite ? "{\"binding\":3,\"slot\":4}" : "");
    const uint32_t *pooled_words = finite ? pool_finite_probe_spv : pool_restore_probe_spv;
    size_t pooled_count = (finite ? sizeof pool_finite_probe_spv : sizeof pool_restore_probe_spv) / sizeof *pooled_words;
    const GfxShader *pooled = gfx_shader_own(pooled_words, pooled_count, reflection);
    const GfxShader *direct = gfx_shader_own(sample_restore_probe_spv, sizeof sample_restore_probe_spv / sizeof *sample_restore_probe_spv,
        "{\"key\":\"pool-restore-direct-fs\",\"stage\":\"ps\",\"cbufs\":[],\"sbufs\":[],\"images\":[],"
        "\"textures\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":8,\"type\":3}]}");
    CHECK(vs && pooled && direct && !!gfx_shader_has_pool_index_plan(pooled) == finite,
          "owned fullscreen shader selected the expected finite or per-invocation index class");
    ProbeSampleProgram pool_program = {0}, direct_program = {0};
    probe_sample_program(&pool_program, vs, pooled);
    probe_sample_program(&direct_program, vs, direct);

    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000,
                   .size = 0x100000, .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool, &pool), "imported source and output pool");
    unsigned texture_count = finite ? 4 : 3;
    tex_t textures[4];
    for (unsigned i = 0; i < texture_count; i++) {
        init_tex(&textures[i], &pool, i * 0x10000, NVN_FORMAT_RGBA8);
        textures[i].serial = nvn_new_serial();
        textures[i].d.width = textures[i].d.height = SIDE;
        nvn_tex_layout(&textures[i].d, &textures[i].L);
    }
    uint32_t pixels[SIDE * SIDE];
    for (unsigned i = 0; i < 2; i++) {
        for (unsigned p = 0; p < SIDE * SIDE; p++) pixels[p] = i ? right : left;
        CHECK(!nvn_swizzle(&textures[i].d, 0, 0, pixels, SIDE * 4, textures[i].mem), "initialize sampled source %u", i);
        b->guest_written((uintptr_t)textures[i].mem, textures[i].L.size);
        vk_texture_register(LEFT_SLOT + i, &textures[i], NULL);
    }
    const samp_par sampler = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
        .ws = NVN_WRAP_MODE_REPEAT, .wt = NVN_WRAP_MODE_REPEAT, .wr = NVN_WRAP_MODE_REPEAT, .lod_max = 1};
    vk_sampler_register(SAMP_SLOT, &sampler);
    const NVNcopyRegion full = {0, 0, 0, SIDE, SIDE, 1};
    const op_hdr empty[2] = {{OP_BEGIN, sizeof(op_hdr)}, {OP_END, sizeof(op_hdr)}};
    uint64_t advance = (uint64_t)(uintptr_t)empty;
    uint64_t uniform_addr = 0;
    uint64_t prior_bindings = vk_exec_pool_probe_finite_bindings();
    if (finite) {
        const uint32_t indices[4] = {LEFT_SLOT, RIGHT_SLOT, COLD_SLOT, UINT32_MAX};
        uint8_t *uniform = pool.mem + 0x80000;
        memcpy(uniform, indices, sizeof indices);
        uniform_addr = (uintptr_t)uniform;
        b->guest_written(uniform_addr, sizeof indices);
        for (unsigned p = 0; p < SIDE * SIDE; p++) pixels[p] = 0xff32cdef;
        CHECK(!nvn_swizzle(&textures[3].d, 0, 0, pixels, SIDE * 4, textures[3].mem), "initialize unrelated slot");
        b->guest_written((uintptr_t)textures[3].mem, textures[3].L.size);
        vk_texture_register(COLD_SLOT, &textures[3], NULL);
        const uint64_t cold_handle = COLD_SLOT | ((uint64_t)SAMP_SLOT << 20);
        CHECK(probe_sample_draw(b, &direct_program.program, &textures[2], cold_handle, 0, NULL, NULL),
              "unrelated slot materialized for an actual direct GPU sample");
        b->read_texels(&textures[2], NULL, &full, (uint8_t *)pixels, SIDE * 4, sizeof pixels);
        for (unsigned i = 0; i < SIDE * SIDE; ++i)
            CHECK(pixels[i] == 0xff32cdef, "unrelated GPU material pixel %u", i);
        CHECK(b->fence_wait(b->submit(&advance, 1), UINT64_C(5000000000)), "unrelated native sample completed");
        CHECK(vk_texture_restore_probe_evict(&textures[3]), "detach unrelated material before its first indexed publication");
    }
    CHECK(probe_sample_draw(b, &pool_program.program, &textures[2], 0, uniform_addr, NULL, NULL),
          "first actual bindless draw completed");
    CHECK(vk_exec_pool_probe_finite_bindings() == prior_bindings + (unsigned)finite,
          "first draw only used a finite native grant in the uniform-index case");
    b->read_texels(&textures[2], NULL, &full, (uint8_t *)pixels, SIDE * 4, sizeof pixels);
    for (unsigned i = 0; i < SIDE * SIDE; i++) {
        uint32_t expected = i % SIDE < SIDE / 2 ? left : right;
        CHECK(pixels[i] == expected, "warm pool pixel %u was %08x, expected %08x", i, pixels[i], expected);
    }
    VkImage old_left = vk_exec_pool_probe_member(KIND_2D, LEFT_SLOT, 1);
    VkImage old_right = vk_exec_pool_probe_member(KIND_2D, RIGHT_SLOT, 1);
    if (finite) CHECK(!vk_exec_pool_probe_member(KIND_2D, COLD_SLOT, 0), "the first narrow set did not need unrelated residency");
    CHECK(old_left && old_right && old_left != old_right, "outgoing set holds both exact native images");

    // Overwrite only RIGHT_SLOT's backing and materialize its current immutable source outside the bindless set.
    for (unsigned i = 0; i < SIDE * SIDE; i++) pixels[i] = replacement;
    b->write_texels(&textures[1], NULL, &full, (const uint8_t *)pixels, SIDE * 4, sizeof pixels);
    uint64_t right_handle = RIGHT_SLOT | ((uint64_t)SAMP_SLOT << 20);
    CHECK(probe_sample_draw(b, &direct_program.program, &textures[2], right_handle, 0, NULL, NULL),
          "new right source completed an exact direct draw");
    b->read_texels(&textures[2], NULL, &full, (uint8_t *)pixels, SIDE * 4, sizeof pixels);
    for (unsigned i = 0; i < SIDE * SIDE; i++)
        CHECK(pixels[i] == replacement, "updated right source pixel %u was %08x, expected %08x", i, pixels[i], replacement);

    CHECK(b->fence_wait(b->submit(&advance, 1), UINT64_C(5000000000)), "advance past both actual uses before detaching residency");
    CHECK(vk_texture_restore_probe_evict(&textures[0]), "detach left residency while outgoing set owns the old native image");
    CHECK(vk_texture_restore_probe_evict(&textures[1]), "evict unpublished replacement right residency");
    uint64_t before = vk_texture_restore_probe_count();
    CHECK(probe_sample_draw(b, &pool_program.program, &textures[2], 0, uniform_addr, NULL, NULL),
          "required bindless draw completed after detaching selected sources");
    CHECK(vk_exec_pool_probe_finite_bindings() == prior_bindings + 2 * (unsigned)finite,
          "draw after restoration only used a finite native grant in the uniform-index case");
    CHECK(vk_texture_restore_probe_count() == before + 1, "exactly one staged restoration for the newly required right member");
    if (finite) CHECK(!vk_exec_pool_probe_member(KIND_2D, COLD_SLOT, 0), "the restored narrow set left unrelated material cold");
    CHECK(vk_exec_pool_probe_member(KIND_2D, LEFT_SLOT, 0) == old_left &&
          vk_exec_pool_probe_member(KIND_2D, RIGHT_SLOT, 0) != old_right, "incoming set carries the old left image and new right version");
    b->read_texels(&textures[2], NULL, &full, (uint8_t *)pixels, SIDE * 4, sizeof pixels);
    for (unsigned i = 0; i < SIDE * SIDE; i++) {
        uint32_t expected = i % SIDE < SIDE / 2 ? left : replacement;
        CHECK(pixels[i] == expected, "restored pool pixel %u was %08x, expected %08x", i, pixels[i], expected);
    }
    for (unsigned i = 0; i < texture_count; i++) b->texture_release(&textures[i]);
    b->pool_fini(&pool);
    printf("probe-vkexec: %s sampled pool kept the left VkImage and restored the updated right member once (%u exact pixels per draw)\n",
           finite ? "finite uniform-index" : "per-invocation", SIDE * SIDE);
    return 0;
}

static int test_pool_dispatch(const GfxBackend *b, uint8_t *guest, int pointer_table) {
    enum { SIDE = 32, LEFT_SLOT = 17, RIGHT_SLOT = 18, SAMP_SLOT = 3, COMPUTE = 5, SAMPLES = 32 };
    char reflection[640];
    snprintf(reflection, sizeof reflection,
        "{\"key\":\"pool-dispatch-cs\",\"stage\":\"cs\",\"cbufs\":[],\"images\":[],"
        "\"sbufs\":[{\"binding\":3,\"slot\":0,\"cb_slot\":0,\"cb_offset\":196,\"flags\":1%s}],\"textures\":["
        "{\"binding\":1,\"type\":3,\"array\":32768,\"pool\":1,\"separate\":true},"
        "{\"binding\":2,\"array\":4096,\"pool\":2,\"separate\":true}]}", pointer_table ? ",\"ptr\":0" : "");
    const GfxShader *cs = gfx_shader_own(pool_dispatch_probe_spv,
        sizeof pool_dispatch_probe_spv / sizeof *pool_dispatch_probe_spv, reflection);
    CHECK(cs, "owned pooled compute and reflected writable storage binding");
    ProbeSampleProgram definition = {0};
    definition.program = (prog_t){.magic = NVN_PROG_MAGIC, .serial = nvn_new_serial(), .nstage = 1};
    const uint32_t stage = COMPUTE, size = sizeof definition.code[0];
    memcpy(definition.control[0] + 0x704, &size, sizeof size);
    memcpy(definition.control[0] + 0x714, &stage, sizeof stage);
    definition.program.sd[0] = (NVNshaderData){.data = (uintptr_t)definition.code[0], .control = definition.control[0]};
    vk_exec_probe_shader(definition.control[0], cs);

    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000,
                   .size = 0x100000, .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool, &pool), "imported texture, indirect-argument and output backing");
    enum { TABLE_PAGE = 0x10000 };
    pool_t physical = {.magic=NVN_POOL_MAGIC, .flags=0x189, .size=TABLE_PAGE*2,
                       .mem=(uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .game_mem=guest+0xA00000};
    pool_t virtual = {.magic=NVN_POOL_MAGIC, .flags=0x2a1, .size=TABLE_PAGE, .span=TABLE_PAGE,
                      .mem=(uint8_t *)(uintptr_t)(RT_GPU_VA_BASE+TABLE_PAGE*2)};
    if (pointer_table) CHECK(b->pool_init(&physical,&physical) && b->pool_init(&virtual,&virtual),
                             "CPU-invisible physical allocation and virtual pointer table");
    tex_t texture[2];
    uint32_t texels[SIDE * SIDE];
    for (unsigned i = 0; i < 2; i++) {
        init_tex(&texture[i], &pool, i * 0x10000, NVN_FORMAT_RGBA8);
        texture[i].serial = nvn_new_serial();
        texture[i].d.width = texture[i].d.height = SIDE;
        nvn_tex_layout(&texture[i].d, &texture[i].L);
        for (unsigned j = 0; j < SIDE * SIDE; j++) texels[j] = i ? 0xff00ff00 : 0xff0000ff;
        CHECK(!nvn_swizzle(&texture[i].d, 0, 0, texels, SIDE * 4, texture[i].mem), "compute source %u swizzled", i);
        b->guest_written((uintptr_t)texture[i].mem, texture[i].L.size);
        vk_texture_register(LEFT_SLOT + i, &texture[i], NULL);
    }
    const samp_par sampler = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
        .ws = NVN_WRAP_MODE_REPEAT, .wt = NVN_WRAP_MODE_REPEAT, .wr = NVN_WRAP_MODE_REPEAT, .lod_max = 1};
    vk_sampler_register(SAMP_SLOT, &sampler);
    const VkDispatchIndirectCommand groups = {2, 1, 1};
    uint8_t *argument = pool.mem + 0x20000, *destination = pool.mem + 0x30000;
    memcpy(argument, &groups, sizeof groups);
    b->guest_written((uintptr_t)argument, sizeof groups);
    float samples[SAMPLES][4];
    for (unsigned phase = 0; phase < (pointer_table ? 2u : 1u); phase++) {
        uint64_t target = (uintptr_t)(destination + phase*TABLE_PAGE);
        memset((void *)(uintptr_t)target, 0xa5, sizeof samples);
        b->guest_written(target, sizeof samples);
        _Alignas(8) uint8_t stream[4096];
        uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
        if (pointer_table) {
            const nvn_mapping_t mapping = { (const NVNmemoryPool *)(const void *)&physical,
                (int64_t)phase*TABLE_PAGE, 0, TABLE_PAGE };
            CHECK(b->pool_map_virtual(&virtual,1,&mapping), "publish pointer-table mapping %u", phase);
            uint8_t *staging = argument + 64;
            memcpy(staging, &target, sizeof target);
            b->guest_written((uintptr_t)staging, sizeof target);
            const p_copy_b2b seed = {(uintptr_t)staging, (uintptr_t)virtual.mem, sizeof target};
            next = put(next, OP_COPY_B2B, &seed, sizeof seed);
        }
        const p_bind_program bind = {&definition.program, 1 << COMPUTE};
        next = put(next, OP_BIND_PROGRAM, &bind, sizeof bind);
        const p_bind_buf output = {.stage = COMPUTE, .index = 0,
            .addr = pointer_table ? (uintptr_t)virtual.mem : target,
            .size = pointer_table ? sizeof target : sizeof samples};
        next = put(next, OP_BIND_SSBO, &output, sizeof output);
        const p_dispatch_ind indirect = {.ind = (uintptr_t)argument};
        next = put(next, OP_DISPATCH_IND, &indirect, sizeof indirect);
        put(next, OP_END, NULL, 0);
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t exact_before, exact_after;
        uint64_t before = vk_exec_pool_probe_generation(&exact_before);
        CHECK(b->fence_wait(b->submit(&handle, 1), UINT64_C(5000000000)), "ordered pooled indirect compute completed");
        b->guest_read(target, sizeof samples);
        memcpy(samples, (void *)(uintptr_t)target, sizeof samples);
        for (unsigned i = 0; i < SAMPLES; i++) {
            int red = i < SAMPLES / 2;
            CHECK(samples[i][0] == red && samples[i][1] == !red && samples[i][2] == 0 && samples[i][3] == 1,
                  "compute sample %u read %.3f/%.3f/%.3f/%.3f", i, samples[i][0], samples[i][1], samples[i][2], samples[i][3]);
        }
        uint64_t after = vk_exec_pool_probe_generation(&exact_after);
        CHECK(exact_after == exact_before + 1, "the real selected indirect dispatch emitted with its prepared grant");
        CHECK(after == before + 1, "the exact indirect dispatch bound its prepared set with one publication");
        if (pointer_table && !phase) {
            memset(destination, 0x5a, sizeof samples);
            b->guest_written((uintptr_t)destination, sizeof samples);
        }
    }
    if (pointer_table) {
        b->guest_read((uintptr_t)destination, sizeof samples);
        for (unsigned i = 0; i < sizeof samples; i++)
            CHECK(destination[i] == 0x5a, "remapped pointer reused original output byte %u", i);
        b->pool_fini(&virtual);
        b->pool_fini(&physical);
        puts("probe-vkexec: actual CPU-invisible pointer table and new virtual mapping selected different output buffers");
    }
    for (unsigned i = 0; i < 2; i++) b->texture_release(&texture[i]);
    b->pool_fini(&pool);
    printf("probe-vkexec: ordered pooled DispatchIndirect wrote %u exact RGBA values through one native set publication\n", SAMPLES);
    return 0;
}

void nvnp_nvnCommandBufferBindProgram(Cpu *);
void nvnp_nvnCommandBufferBindImage(Cpu *);
void nvnp_nvnCommandBufferDispatchCompute(Cpu *);
void nvnp_nvnCommandBufferDispatchComputeIndirect(Cpu *);
void nvnp_nvnCommandBufferBindSeparateTexture(Cpu *);
uint64_t vk_exec_draw_probe_storage_first_writes(void);
uint64_t vk_exec_draw_probe_storage_aliases(void);

uint64_t vk_texture_probe_storage_cpu_seeds(void);

static int test_storage_first_write(const GfxBackend *b, uint8_t *guest, int paired, int dirty) {
    enum { SIDE = 32, LEVELS = 2, LAYERS = 2, PLANES = LEVELS * LAYERS, STRIDE = SIDE * SIDE, SLOT = 17, COMPUTE = 5 };
    static _Alignas(16) NVNcommandBuffer command;
    static _Alignas(8) uint8_t control[3][4096];
    const unsigned count = paired ? 2 : 1;
    const GfxShader *cs = paired ? gfx_shader_own(storage_write_plan_probe_spv,
        sizeof storage_write_plan_probe_spv / sizeof *storage_write_plan_probe_spv,
        "{\"key\":\"storage-write-plan-cs\",\"stage\":\"cs\",\"cbufs\":[],"
        "\"textures\":[{\"binding\":3,\"cbuf_slot\":2,\"handle\":90,\"type\":3,\"separate\":true}],"
        "\"images\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":72,\"type\":3,\"format\":\"R32Uint\",\"flags\":4},"
        "{\"binding\":2,\"cbuf_slot\":2,\"handle\":74,\"type\":3,\"format\":\"R32Uint\",\"flags\":4}]}")
      : gfx_shader_own(storage_first_write_probe_spv, sizeof storage_first_write_probe_spv / sizeof *storage_first_write_probe_spv,
        "{\"key\":\"storage-first-write-cs\",\"stage\":\"cs\",\"cbufs\":[],"
        "\"images\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":72,\"type\":3,\"format\":\"R32Uint\",\"flags\":4}]}");
    CHECK(cs && gfx_shader_write_inventory_complete(cs), "R32UI storage ImageWrite inventory");
    ProbeSampleProgram definition = {0};
    definition.program = (prog_t){.magic = NVN_PROG_MAGIC, .serial = nvn_new_serial(), .nstage = 1};
    const uint32_t stage = COMPUTE, code_size = sizeof definition.code[0];
    memcpy(definition.control[0] + 0x704, &code_size, sizeof code_size);
    memcpy(definition.control[0] + 0x714, &stage, sizeof stage);
    definition.program.sd[0] = (NVNshaderData){.data = (uintptr_t)definition.code[0], .control = definition.control[0]};
    vk_exec_probe_shader(definition.control[0], cs);

    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xb00000,
                   .size = 0x100000, .game_mem = guest + 0xb00000};
    CHECK(b->pool_init(&pool, &pool), "storage write probe memory pool");
    tex_t textures[2];
    const uint32_t base = 0x70200010u, written = 0xe5aa1234u;
    const uint32_t refreshed = 0x14500013u, post_native = 0x24681012u;
    uint32_t pixels[STRIDE];
    uint32_t *result = (uint32_t *)(pool.mem + 0x20000);
    for (unsigned t = 0; t < count; t++) {
        tex_t *texture = &textures[t];
        init_tex(texture, &pool, t * 0x10000, NVN_FORMAT_R32UI);
        texture->serial = nvn_new_serial();
        texture->d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
        texture->d.target = NVN_TEXTURE_TARGET_2D_ARRAY;
        texture->d.width = texture->d.height = SIDE;
        texture->d.depth = LAYERS;
        texture->d.levels = LEVELS;
        nvn_tex_layout(&texture->d, &texture->L);
        for (unsigned layer = 0; layer < LAYERS; layer++) for (unsigned level = 0; level < LEVELS; level++) {
            unsigned plane = level * LAYERS + layer, side = SIDE >> level;
            for (unsigned i = 0; i < side * side; i++) pixels[i] = base + plane * STRIDE + i;
            CHECK(!nvn_swizzle(&texture->d, level, layer, pixels, side * 4, texture->mem), "swizzle R32UI plane %u", plane);
        }
        b->guest_written((uintptr_t)texture->mem, texture->L.size);
    }
    const VkDispatchIndirectCommand groups = {6, 5, 1};
    uint8_t *argument = pool.mem + 0x80000;
    if (paired) {
        memcpy(argument, &groups, sizeof groups);
        b->guest_written((uintptr_t)argument, sizeof groups);
    }
    const view_t selected = {.set_levels = 1, .base_level = 1, .nlevels = 1, .set_layers = 1, .min_layer = 1, .nlayers = 1};
    VkImage previous[2] = {0}, current[2] = {0};
    VkTexturePresentSource *previous_hold[2] = {0};
    uint64_t cpu_seeds = vk_texture_probe_storage_cpu_seeds();
    uint64_t before = vk_exec_draw_probe_storage_first_writes();
    uint64_t aliases = vk_exec_draw_probe_storage_aliases();
#define SW(proc, ...) pinned_clear_proc(nvnp_##proc, (const uint64_t[8]){(uintptr_t)&command, __VA_ARGS__})
    CHECK(SW(nvnCommandBufferInitialize, 0), "storage first-write command initialization");
    SW(nvnCommandBufferAddCommandMemory, (uintptr_t)&pool, 0x40000, 0x4000);
    for (unsigned phase = 0; phase < 3; phase++) {
        SW(nvnCommandBufferAddControlMemory, (uintptr_t)control[phase], sizeof control[phase]);
        SW(nvnCommandBufferBeginRecording, 0);
        if (phase == 1) {
            SW(nvnCommandBufferBindProgram, (uintptr_t)&definition.program, 1 << COMPUTE);
            for (unsigned t = 0; t < count; t++) SW(nvnCommandBufferBindImage, COMPUTE, t, SLOT + t);
            if (paired) {
                SW(nvnCommandBufferBindSeparateTexture, COMPUTE, 0, SLOT);
                SW(nvnCommandBufferDispatchComputeIndirect, (uintptr_t)argument);
            } else SW(nvnCommandBufferDispatchCompute, groups.x, groups.y, groups.z);
        } else {
            for (unsigned t = 0; t < count; t++) {
                for (unsigned level = 0; level < LEVELS; level++) for (unsigned layer = 0; layer < LAYERS; layer++) {
                    view_t view = selected;
                    view.base_level = level;
                    view.min_layer = layer;
                    NVNcopyRegion whole = {0, 0, 0, SIDE >> level, SIDE >> level, 1};
                    unsigned plane = level * LAYERS + layer;
                    SW(nvnCommandBufferCopyTextureToBuffer, (uintptr_t)&textures[t], (uintptr_t)&view, (uintptr_t)&whole,
                       (uintptr_t)(result + ((phase / 2 * count + t) * PLANES + plane) * STRIDE), 0);
                }
            }
        }
        uint64_t handle = SW(nvnCommandBufferEndRecording, 0);
        CHECK(handle && b->fence_wait(b->submit(&handle, 1), UINT64_C(5000000000)), "storage phase %u completed", phase);
        CHECK(vk_exec_draw_probe_storage_first_writes() == before + (phase >= 1) * count, "phase %u emitted exact storage first writes", phase);
        CHECK(vk_exec_draw_probe_storage_aliases() == aliases + (phase >= 1 && paired), "phase %u used the exact sampled alias", phase);
        if (dirty) CHECK(vk_texture_probe_storage_cpu_seeds() == cpu_seeds + (phase >= 1),
                         "phase %u rebuilt the private storage image from exact current CPU bytes", phase);
        if (!phase) {
            for (unsigned t = 0; t < count; t++) {
                const VkTexImage *image = vk_texture_get(&textures[t]);
                CHECK(image && image->image && !image->gpu_owned && (image->usage & VK_IMAGE_USAGE_STORAGE_BIT),
                      "warm immutable source %u has genuine writable storage usage", t);
                previous[t] = image->image;
                vk_texture_register(SLOT + t, &textures[t], &selected);
                if (dirty && !t) {
                    previous_hold[t] = vk_texture_present_retain_image(image, previous[t]);
                    CHECK(previous_hold[t], "retained the actually sampled immutable native predecessor");
                    for (unsigned i = 0; i < STRIDE; i++) pixels[i] = refreshed + i;
                    CHECK(!nvn_swizzle(&textures[t].d, 0, 0, pixels, SIDE * 4, textures[t].mem),
                          "replace only mip0/layer0 through its actual changed CPU bytes");
                    b->guest_written((uintptr_t)textures[t].mem, textures[t].L.size);
                }
            }
        } else if (phase == 1) {
            for (unsigned t = 0; t < count; t++) {
                const tex_t *pin = nvn_texture_pin_acquire(&textures[t]);
                const VkTexImage *image = vk_texture_get(pin);
                int private = nvn_texture_pin_attachment(pin) && image && image->gpu_owned && image->image != previous[t];
                if (private) current[t] = image->image;
                nvn_texture_pin_release(pin);
                CHECK(private, "storage dispatch installed its pin's distinct private image %u", t);
            }
            if (paired) CHECK(current[0] != current[1], "distinct storage registrations own two native placements");
            if (dirty) {
                const view_t mip0 = {.set_levels = 1, .nlevels = 1, .set_layers = 1, .nlayers = 1};
                const NVNcopyRegion cpu_update = {11, 9, 0, 1, 1, 1};
                b->write_texels(&textures[0], &mip0, &cpu_update, (const uint8_t *)&post_native, 4, 4);
                const VkTexImage *again = vk_texture_get(&textures[0]);
                CHECK(again && again->image == current[0] && again->gpu_owned,
                      "actual ordered CPU texel update patched the completed native writer");
            }
        }
    }
    b->guest_read((uintptr_t)result, 2 * count * PLANES * STRIDE * 4);
    unsigned checked = 0;
    for (unsigned t = 0; t < count; t++) {
        uint32_t stored = written + (paired ? base + (PLANES - 1) * STRIDE + t : 0);
        for (unsigned level = 0; level < LEVELS; level++) for (unsigned layer = 0; layer < LAYERS; layer++) {
            unsigned plane = level * LAYERS + layer, side = SIDE >> level;
            for (unsigned y = 0; y < side; y++) for (unsigned x = 0; x < side; x++) {
                unsigned i = y * side + x;
                uint32_t prior = base + plane * STRIDE + i;
                uint32_t expected = level == 1 && layer == 1 && x >= 2 && x < 8 && y >= 3 && y < 8 ? stored : prior;
                if (dirty && !t && !level && !layer)
                    expected = x == 11 && y == 9 ? post_native : refreshed + i;
                CHECK(result[(t * PLANES + plane) * STRIDE + i] == prior &&
                      result[((count + t) * PLANES + plane) * STRIDE + i] == expected,
                      "storage target %u level %u layer %u pixel %u,%u expected %#x", t, level, layer, x, y, expected);
                checked += 2;
            }
        }
    }
    for (unsigned i = 0; i < 3; i++) SW(nvnCommandBufferAddControlMemory, (uintptr_t)control[i], sizeof control[i]);
    SW(nvnCommandBufferFinalize, 0);
#undef SW
    for (unsigned t = 0; t < count; t++) {
        if (previous_hold[t]) vk_texture_present_release(previous_hold[t]);
        b->texture_release(&textures[t]);
    }
    if (dirty) printf("probe-vkexec: dirty storage selected one exact CPU refresh; distinct old/current native images, "
                      "partial GPU ImageWrite then CPU WriteTexels; %u prior/current pixels verified\n", checked);
    b->pool_fini(&pool);
    printf("probe-vkexec: ordered storage first writes installed %u private images, used %d sampled aliases and checked %u prior/current pixels\n",
           count, paired, checked);
    return 0;
}

void vk_exec_draw_probe_force_compare(void);
uint64_t vk_exec_draw_probe_compare_copies(void);
int vk_exec_draw_probe_typed_storage_ready(void);

static int test_required_fallbacks(const GfxBackend *b, uint8_t *guest) {
    enum { SIDE = 16, TEXTURE = 17, SAMPLER = 3, COMPUTE = 5 };
    const GfxShader *cs = gfx_shader_own(fallback_probe_spv, sizeof fallback_probe_spv / sizeof *fallback_probe_spv,
        "{\"key\":\"required-fallback-cs\",\"stage\":\"cs\",\"cbufs\":[],"
        "\"textures\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":8,\"type\":1027}],"
        "\"images\":[{\"binding\":2,\"cbuf_slot\":2,\"handle\":72,\"type\":3,\"format\":\"R32Uint\"}],"
        "\"sbufs\":[{\"binding\":3,\"slot\":0,\"cb_slot\":0,\"cb_offset\":196,\"flags\":1}]}");
    CHECK(cs, "a native compare, typed read-only storage and writable SSBO module");
    ProbeSampleProgram definition = {0};
    definition.program = (prog_t){.magic = NVN_PROG_MAGIC, .serial = nvn_new_serial(), .nstage = 1};
    const uint32_t stage = COMPUTE, size = sizeof definition.code[0];
    memcpy(definition.control[0] + 0x704, &size, sizeof size);
    memcpy(definition.control[0] + 0x714, &stage, sizeof stage);
    definition.program.sd[0] = (NVNshaderData){.data = (uintptr_t)definition.code[0], .control = definition.control[0]};
    vk_exec_probe_shader(definition.control[0], cs);

    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000,
                   .size = 0x100000, .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool, &pool), "imported native source, arguments and reports");
    tex_t texture;
    init_tex(&texture, &pool, 0, NVN_FORMAT_R32F);
    texture.serial = nvn_new_serial();
    texture.d.width = texture.d.height = SIDE;
    nvn_tex_layout(&texture.d, &texture.L);
    float texels[SIDE * SIDE];
    for (unsigned i = 0; i < SIDE * SIDE; i++) texels[i] = (i % SIDE < SIDE/2) ? 0.25f : 0.75f;
    CHECK(!nvn_swizzle(&texture.d, 0, 0, texels, SIDE * sizeof *texels, texture.mem), "known R32F compare pattern");
    b->guest_written((uintptr_t)texture.mem, texture.L.size);
    vk_texture_register(TEXTURE, &texture, NULL);
    const samp_par sampler = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
        .ws = NVN_WRAP_MODE_REPEAT, .wt = NVN_WRAP_MODE_REPEAT, .wr = NVN_WRAP_MODE_REPEAT,
        .lod_max = 1, .cmp_mode = 1, .cmp_func = NVN_DEPTH_FUNC_LESS};
    vk_sampler_register(SAMPLER, &sampler);
    uint8_t *argument = pool.mem + 0x20000, *destination = pool.mem + 0x30000;
    const VkDispatchIndirectCommand indirect_data = {1, 1, 1};
    memcpy(argument, &indirect_data, sizeof indirect_data);
    b->guest_written((uintptr_t)argument, sizeof indirect_data);

    vk_exec_draw_probe_force_compare();
    uint64_t copies = vk_exec_draw_probe_compare_copies();
    CHECK(!vk_exec_draw_probe_typed_storage_ready(), "cold exact-format R32_UINT stand-in");
    for (unsigned phase = 0; phase < 2; phase++) {
        if (phase) {
            for (unsigned i = 0; i < SIDE * SIDE; i++) texels[i] = 1.0f - texels[i];
            const NVNcopyRegion full = {0, 0, 0, SIDE, SIDE, 1};
            b->write_texels(&texture, NULL, &full, (const uint8_t *)texels, SIDE * 4, sizeof texels);
        }
        uint32_t result[2];
        memset(destination, 0xa5, sizeof result);
        b->guest_written((uintptr_t)destination, sizeof result);
        _Alignas(8) uint8_t stream[4096];
        uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
        const p_bind_program program = {&definition.program, 1 << COMPUTE};
        next = put(next, OP_BIND_PROGRAM, &program, sizeof program);
        const p_bind_handles sample = {.stage = COMPUTE, .n = 1,
            .h = {TEXTURE | ((uint64_t)SAMPLER << 20)}};
        next = put(next, OP_BIND_TEX, &sample, sizeof sample);
        const p_bind_buf output = {.stage = COMPUTE, .index = 0, .addr = (uintptr_t)destination, .size = sizeof result};
        next = put(next, OP_BIND_SSBO, &output, sizeof output);
        // Leave the declared R32_UINT storage image unbound on purpose.
        if (phase) {
            const p_dispatch_ind dispatch = {.ind = (uintptr_t)argument};
            next = put(next, OP_DISPATCH_IND, &dispatch, sizeof dispatch);
        } else {
            const p_dispatch dispatch = {.x = 1, .y = 1, .z = 1};
            next = put(next, OP_DISPATCH, &dispatch, sizeof dispatch);
        }
        put(next, OP_END, NULL, 0);
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        CHECK(b->fence_wait(b->submit(&handle, 1), UINT64_C(5000000000)), "actual fallback compute completed");
        b->guest_read((uintptr_t)destination, sizeof result);
        memcpy(result, destination, sizeof result);
        CHECK(result[0] == phase && result[1] == 1-phase,
              "native compare and missing R32_UINT read %u/%u in phase %u", result[0], result[1], phase);
        CHECK(vk_exec_draw_probe_compare_copies() == copies + phase+1 && vk_exec_draw_probe_typed_storage_ready(),
              "actual earlier native compare copy and initialized typed storage provider");
    }
    b->texture_release(&texture);
    b->pool_fini(&pool);
    printf("probe-vkexec: prepared exact D32 compare and zero R32_UINT storage produced both CPU-checked outputs\n");
    return 0;
}


void vk_texture_alias_probe_counts(uint64_t *producers, uint64_t *assembled);
uint64_t vk_texture_alias_probe_relocations(void);
uint64_t vk_texture_alias_probe_load_cancels(void);
uint64_t vk_texture_alias_probe_dormant_loads(void);

static int test_current_alias_sample(const GfxBackend *b, pool_t *output_pool, tex_t *source,
                                     const uint8_t *expected, int mode, VkImage predecessor) {
    enum { LEFT = 17, RIGHT = 18, SAMPLER = 3, KIND = 2 };
    int finite = mode == 4 || mode == 22, dynamic = mode == 5 || mode == 19 || mode == 20;
    const int source_width = source->d.width, source_height = source->d.height;
    const GfxShader *vs = gfx_shader_own(hdr_fullscreen_spv, sizeof hdr_fullscreen_spv / sizeof *hdr_fullscreen_spv,
        "{\"key\":\"current-alias-vs\",\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
    const GfxShader *fs = NULL;
    if (finite || dynamic) {
        char reflection[512];
        snprintf(reflection, sizeof reflection,
            "{\"key\":\"current-alias-%s\",\"stage\":\"ps\",\"cbufs\":[%s],\"sbufs\":[],\"images\":[],\"textures\":["
            "{\"binding\":1,\"type\":3,\"array\":32768,\"pool\":1,\"separate\":true},"
            "{\"binding\":2,\"array\":4096,\"pool\":2,\"separate\":true}]}",
            finite ? "finite" : "dynamic", finite ? "{\"binding\":3,\"slot\":4}" : "");
        const uint32_t *words = finite ? pool_finite_probe_spv : pool_restore_probe_spv;
        size_t bytes = finite ? sizeof pool_finite_probe_spv : sizeof pool_restore_probe_spv;
        fs = gfx_shader_own(words, bytes / sizeof *words, reflection);
    } else {
        fs = gfx_shader_own(sample_restore_probe_spv,
            sizeof sample_restore_probe_spv / sizeof *sample_restore_probe_spv,
            "{\"key\":\"current-alias-direct\",\"stage\":\"ps\",\"cbufs\":[],\"sbufs\":[],\"images\":[],"
            "\"textures\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":8,\"type\":3}]}");
    }
    CHECK(vs && fs && !!gfx_shader_has_pool_index_plan(fs) == finite, "own actual alias sample shaders");
    ProbeSampleProgram program = {0};
    probe_sample_program(&program, vs, fs);
    const samp_par native_sampler = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
        .ws = NVN_WRAP_MODE_CLAMP_TO_EDGE, .wt = NVN_WRAP_MODE_CLAMP_TO_EDGE,
        .wr = NVN_WRAP_MODE_CLAMP_TO_EDGE, .lod_max = 1};
    vk_sampler_register(SAMPLER, &native_sampler);
    const view_t view = {.set_swizzle = 1, .swz = {NVN_TEXTURE_SWIZZLE_ZERO, NVN_TEXTURE_SWIZZLE_R,
                                                    NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_ONE}};
    vk_texture_register(LEFT, source, &view);
    if (dynamic) vk_texture_register(RIGHT, source, &view);

    uint64_t indices = 0;
    if (finite) {
        const uint32_t choice[4] = {LEFT, LEFT, 19, UINT32_MAX};
        indices = (uintptr_t)(output_pool->mem + 0x70000);
        memcpy((void *)(uintptr_t)indices, choice, sizeof choice);
        b->guest_written(indices, sizeof choice);
    }
    tex_t output;
    init_tex(&output, output_pool, 0x80000, NVN_FORMAT_RGBA8);
    output.serial = nvn_new_serial();
    output.d.width = source_width;
    output.d.height = source_height;
    nvn_tex_layout(&output.d, &output.L);
    const NVNcopyRegion full = {0, 0, 0, source_width, source_height, 1};
    const VkTexImage *old = predecessor ? NULL : vk_texture_get(source);
    CHECK(predecessor ? !vk_texture_alias_probe_receiver_image(source) : old && old->image,
          "sampled receiver starts truly cold or keeps its explicitly tested earlier VkImage");
    const VkImage old_image = predecessor ? predecessor : old->image;
    uint64_t writes, composed, confirmed_writes, confirmed_composed;
    vk_texture_alias_probe_counts(&writes, &composed);
    uint64_t finite_binds = vk_exec_pool_probe_finite_bindings();
    uint64_t handle = finite || dynamic ? 0 : LEFT | ((uint64_t)SAMPLER << 20);
    CHECK(probe_sample_draw(b, &program.program, &output, handle, indices, NULL, NULL), "native alias sample draw completed");
    vk_texture_alias_probe_counts(&confirmed_writes, &confirmed_composed);
    CHECK(confirmed_writes == writes && confirmed_composed > composed,
          "drawing composed exact reads created no fabricated image writer");
    if (predecessor) CHECK(!vk_texture_alias_probe_receiver_image(source),
                           "a first sampled draw did not allocate or select a legacy receiving Tex");
    const uint64_t observed_finite = vk_exec_pool_probe_finite_bindings() - finite_binds;
    CHECK(observed_finite == (unsigned)finite, "actual scoped native bound set");
    VkImage used = VK_NULL_HANDLE, other = VK_NULL_HANDLE;
    if (finite || dynamic) {
        used = vk_exec_pool_probe_member(KIND, LEFT, 0);
        CHECK(used && used != old_image, "actual indexed Vulkan image is the reconstruction");
        if (dynamic) {
            other = vk_exec_pool_probe_member(KIND, RIGHT, 0);
            CHECK(other && other != old_image, "all per-invocation selected members use reconstructed bytes");
        }
    }
    const size_t count = (size_t)source_width * source_height;
    uint32_t *pixels = malloc(count * sizeof *pixels);
    CHECK(pixels, "alias sample readback pixels");
    b->read_texels(&output, NULL, &full, (uint8_t *)pixels, source_width * 4, count * sizeof *pixels);
    for (size_t i = 0; i < count; ++i) {
        // The authored fullscreen VS maps UV to ndc=uv*2-1. The guest graphics viewport has negative height.
        const size_t x = i % source_width, y = i / source_width;
        const uint8_t source_pixel = expected[(source_height - 1 - y) * source_width + x];
        uint32_t target = UINT32_C(0xff000000) | ((uint32_t)source_pixel << 16) | ((uint32_t)source_pixel << 8);
        CHECK(pixels[i] == target, "current alias sample pixel %llu was %08x, expected %08x",
              (unsigned long long)i, pixels[i], target);
    }
    free(pixels);
    b->texture_release(&output);
    printf("probe-vkexec: exact current alias %s sampled %llu correctly swizzled native pixels\n",
        finite ? "finite pool" : dynamic ? "per-invocation pool" : "direct", (unsigned long long)count);
    printf("probe-vkexec: current alias sample mode %d native writes %llu->%llu compositions %llu->%llu finite binds +%llu physical old %p/slot17 %p/slot18 %p\n",
        mode, (unsigned long long)writes, (unsigned long long)confirmed_writes, (unsigned long long)composed,
        (unsigned long long)confirmed_composed, (unsigned long long)observed_finite, (void *)old_image,
        (void *)used, (void *)other);
    return 0;
}

static int test_current_image_alias(const GfxBackend *b, uint8_t *guest, int mode) {
    int canceled_load = mode == 9 || mode == 13;
    const int dormant_load = mode == 16;
    const int mrt_load = mode == 15;
    int replacement_load = mode == 12 || mode == 13 || mrt_load || dormant_load, fresh_load = mode == 8 || mode == 9;
    int padded_bound = mode == 14;
    int depth_alias = mode == 18 || mode == 19;
    int prospective = mode == 21 || mode == 22;
    int clear_texture = mode >= 11 && mode <= 16, migrate = mode >= 10 && mode <= 16;
    int mixed_load = fresh_load || replacement_load;
    int partial = mode == 2 || fresh_load, rendered = mode == 6 || mode == 7 || depth_alias, raster = mode == 7;
    enum { DRAW_X = 65, DRAW_Y = 32, DRAW_W = 31, DRAW_H = 19, DRAW_VALUE = 83,
           LOAD_WIDTH = 64, LOAD_HEIGHT = 64, LOAD_BYTES = LOAD_WIDTH * LOAD_HEIGHT,
           LOAD_X = 29, LOAD_Y = 22, LOAD_W = 19, LOAD_H = 13 };
    enum { OLD_WIDTH = 128, OLD_HEIGHT = 64, OLD_BYTES = OLD_WIDTH * OLD_HEIGHT,
           NEW_WIDTH = 16, NEW_HEIGHT = 8, NEW_BYTES = NEW_WIDTH * NEW_HEIGHT * 4 };
    pool_t physical = {.magic = NVN_POOL_MAGIC, .flags = 0x189,
        .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .size = 0x100000, .game_mem = guest + 0xa00000};
    pool_t imported = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000,
        .size = 0x100000, .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&physical, &physical) && b->pool_init(&imported, &imported), "current alias probe pools");
    tex_t old, newer;
    init_tex(&old, &physical, 0, NVN_FORMAT_R8);
    init_tex(&newer, &physical, 0, depth_alias ? NVN_FORMAT_DEPTH32F : NVN_FORMAT_R32UI);
    old.serial = nvn_new_serial();
    newer.serial = nvn_new_serial();
    old.d.width = OLD_WIDTH, old.d.height = padded_bound ? OLD_HEIGHT - 1 : OLD_HEIGHT;
    const unsigned old_pixels = old.d.width * old.d.height;
    newer.d.width = partial ? NEW_WIDTH * 2 : NEW_WIDTH;
    newer.d.height = depth_alias ? NEW_HEIGHT - 1 : partial ? NEW_HEIGHT * 2 : NEW_HEIGHT;
    nvn_tex_layout(&old.d, &old.L);
    nvn_tex_layout(&newer.d, &newer.L);
    CHECK(old.mem == newer.mem && newer.L.size < old.L.size && old.L.size <= 0x10000,
          "different real native formats share guest bytes with a strict partial overwrite");
    if (padded_bound) CHECK(old.L.size > old_pixels, "padded bound witness must have physical bytes outside its texels");
    const uint64_t prior_moves = vk_texture_alias_probe_relocations();
    VkImage original_bound = VK_NULL_HANDLE;
    VkTexturePresentSource *retained_bound = NULL;
    if (migrate) {
        const VkTexImage *warm = vk_texture_get(&old);
        CHECK(warm && warm->image && warm->bound, "pre-clobber probe genuinely starts in a shared Vulkan physical placement");
        original_bound = warm->image;
        retained_bound = vk_texture_present_retain_image(warm, original_bound);
        CHECK(retained_bound, "earlier recorded native descriptor owner retains the original VkImage");
    }
    uint8_t *previous = imported.mem + 0x20000, *patch = imported.mem + 0x30000;
    uint8_t *before = imported.mem + 0x40000, *after = imported.mem + 0x50000;
    uint8_t expected[OLD_BYTES];
    uint8_t *guest_bytes = malloc(old.L.size);
    CHECK(guest_bytes, "alias guest reference workspace");
    for (unsigned i = 0; i < OLD_BYTES; i++) previous[i] = rendered ? 37 : (uint8_t)(1 + i % 111);
    if (raster) for (unsigned y = DRAW_Y; y < DRAW_Y + DRAW_H; ++y)
        memset(previous + y * OLD_WIDTH + DRAW_X, DRAW_VALUE, DRAW_W);
    for (unsigned i = 0; i < NEW_BYTES; i++) patch[i] = rendered || clear_texture ? (uint8_t)(0xd4c3b2a1u >> ((i % 4) * 8)) :
        (uint8_t)(140 + (i * 17) % 97);
    const float depth_value = 0.6875f;
    if (depth_alias) for (unsigned i = 0; i < NEW_BYTES; i += sizeof depth_value)
        memcpy(patch + i, &depth_value, sizeof depth_value);
    CHECK(!nvn_swizzle(&old.d, 0, 0, previous, OLD_WIDTH, guest_bytes), "pack first layout reference");
    const NVNcopyRegion new_extent = {partial ? 5 : 0, partial ? 3 : 0, 0, NEW_WIDTH,
                                      depth_alias ? NEW_HEIGHT - 1 : NEW_HEIGHT, 1};
    const unsigned written_bytes = NEW_WIDTH * 4 * new_extent.height;
    if (depth_alias) CHECK(newer.L.size > written_bytes, "depth producer includes unmodified physical padding");
    CHECK(!nvn_texels_box(&newer.d, 0, 0, new_extent.xoffset * 4, new_extent.yoffset, 0, NEW_WIDTH * 4,
                         new_extent.height, guest_bytes, patch, NEW_WIDTH * 4, 0), "pack later box in the same guest bytes");
    CHECK(!nvn_deswizzle(&old.d, 0, 0, guest_bytes, expected, OLD_WIDTH), "recover expected original-layout pixels");
    tex_t load = {0};
    uint8_t load_expected[OLD_BYTES];
    if (fresh_load) {
        init_tex(&load, &physical, 0, NVN_FORMAT_R8);
        load.serial = nvn_new_serial();
        load.d.width = LOAD_WIDTH, load.d.height = LOAD_HEIGHT;
        nvn_tex_layout(&load.d, &load.L);
        CHECK(load.L.size <= old.L.size, "receiver layout is fully covered by two retained older placements");
        CHECK(!nvn_deswizzle(&load.d, 0, 0, guest_bytes, load_expected, LOAD_WIDTH),
              "known exact region contents in the freshly loaded native shape");
        for (unsigned y = LOAD_Y; y < LOAD_Y + LOAD_H; ++y)
            memset(load_expected + y * LOAD_WIDTH + LOAD_X, DRAW_VALUE, LOAD_W);
    }
    if (prospective) {
        init_tex(&load, &physical, 0, NVN_FORMAT_R8);
        load.serial = nvn_new_serial();
        load.d.width = LOAD_WIDTH, load.d.height = LOAD_HEIGHT;
        nvn_tex_layout(&load.d, &load.L);
        CHECK(load.L.size < old.L.size && load.L.size > newer.L.size &&
              !nvn_deswizzle(&load.d, 0, 0, guest_bytes, load_expected, LOAD_WIDTH),
              "unmaterialized receiver covers bytes from both native producers in a different shape");
    }
    if (replacement_load) {
        memcpy(load_expected, expected, sizeof expected);
        if (!canceled_load) for (unsigned y = LOAD_Y; y < LOAD_Y + LOAD_H; ++y)
            memset(load_expected + y * OLD_WIDTH + LOAD_X, DRAW_VALUE, LOAD_W);
    }
    free(guest_bytes);
    b->guest_written((uintptr_t)previous, OLD_BYTES);
    b->guest_written((uintptr_t)patch, NEW_BYTES);
    _Alignas(8) uint8_t stream[4096];
    uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
    const NVNcopyRegion old_extent = {0, 0, 0, OLD_WIDTH, (int)old.d.height, 1};
    const p_copy_b2t older_write = {.src = (uintptr_t)previous, .tex = &old, .r = old_extent};
    const p_copy_b2t original = {.src = (uintptr_t)before, .tex = &old, .r = old_extent};
    const p_copy_b2t newer_write = {.src = (uintptr_t)patch, .tex = &newer, .r = new_extent};
    const NVNcopyRegion load_extent = {0, 0, 0, LOAD_WIDTH, LOAD_HEIGHT, 1};
    tex_t recreated = old;
    if (dormant_load) recreated.serial = nvn_new_serial();
    const tex_t *receiver = dormant_load ? &recreated : &old;
    const p_copy_b2t assembled = {.src = (uintptr_t)after, .tex = mode == 8 ? &load : receiver,
                                  .r = mode == 8 ? load_extent : old_extent};
    uint64_t writes, combined;
    vk_texture_alias_probe_counts(&writes, &combined);
    const uint64_t canceled = vk_texture_alias_probe_load_cancels();
    const uint64_t dormant = vk_texture_alias_probe_dormant_loads();
    tex_t paint = {0};
    ProbeSampleProgram pipeline = {0};
    if (raster || mixed_load) {
        init_tex(&paint, &imported, 0x90000, NVN_FORMAT_R8);
        paint.serial = nvn_new_serial();
        paint.d.width = paint.d.height = 8;
        nvn_tex_layout(&paint.d, &paint.L);
        uint8_t paint_bytes[64];
        memset(paint_bytes, DRAW_VALUE, sizeof paint_bytes);
        CHECK(!nvn_swizzle(&paint.d, 0, 0, paint_bytes, 8, paint.mem), "paint input exact CPU bytes before replay");
        b->guest_written((uintptr_t)paint.mem, paint.L.size);
        vk_texture_register(17, &paint, NULL);
        const samp_par samp = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
            .ws = NVN_WRAP_MODE_CLAMP_TO_EDGE, .wt = NVN_WRAP_MODE_CLAMP_TO_EDGE,
            .wr = NVN_WRAP_MODE_CLAMP_TO_EDGE, .lod_max = 1};
        vk_sampler_register(3, &samp);
        const GfxShader *vs = gfx_shader_own(hdr_fullscreen_spv, sizeof hdr_fullscreen_spv / sizeof *hdr_fullscreen_spv,
            "{\"key\":\"current-alias-load-vs\",\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
        const GfxShader *fs = gfx_shader_own(sample_restore_probe_spv,
            sizeof sample_restore_probe_spv / sizeof *sample_restore_probe_spv,
            "{\"key\":\"current-alias-load-fs\",\"stage\":\"ps\",\"cbufs\":[],\"sbufs\":[],\"images\":[],"
            "\"textures\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":8,\"type\":3}]}");
        CHECK(vs && fs, "actual ordinary graphics program for partial loaded raster");
        probe_sample_program(&pipeline, vs, fs);
    }
    if (rendered) {
        const p_set_rt target = {.n = 1, .color = {&old}};
        const p_clear_color clear = {0, 0xf, {37.0f / 255.0f, 0, 0, 1}};
        next = put(next, OP_SET_RT, &target, sizeof target);
        next = put(next, OP_CLEAR_COLOR, &clear, sizeof clear);
    } else next = put(next, OP_COPY_B2T, &older_write, sizeof older_write);
    if (raster) {
        p_viewports viewport = {.n = 1};
        viewport.v[0][2] = OLD_WIDTH;
        viewport.v[0][3] = OLD_HEIGHT;
        viewport.v[0][5] = 1.0f;
        next = put(next, OP_VIEWPORTS, &viewport, sizeof viewport);
        const p_scissor region = {DRAW_X, DRAW_Y, DRAW_W, DRAW_H};
        next = put(next, OP_SCISSOR, &region, sizeof region);
        const p_bind_program bind = {&pipeline.program, 3};
        next = put(next, OP_BIND_PROGRAM, &bind, sizeof bind);
        const p_bind_handles texture = {.stage = NVN_STAGE_FS, .n = 1, .h = {(UINT64_C(3) << 20) | 17}};
        next = put(next, OP_BIND_TEX, &texture, sizeof texture);
        const p_draw draw = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
        next = put(next, OP_DRAW, &draw, sizeof draw);
        const p_set_rt none = {0};
        next = put(next, OP_SET_RT, &none, sizeof none);
        const p_scissor full = {0, 0, OLD_WIDTH, OLD_HEIGHT};
        next = put(next, OP_SCISSOR, &full, sizeof full);
    }
    next = put(next, OP_COPY_T2B, &original, sizeof original);
    if (depth_alias) {
        const p_set_rt target = {.depth = &newer};
        const p_scissor scissor = {0, 0, new_extent.width, new_extent.height};
        const p_clear_ds clear = {.depth = depth_value, .depth_mask = 1};
        next = put(next, OP_SET_RT, &target, sizeof target);
        next = put(next, OP_SCISSOR, &scissor, sizeof scissor);
        next = put(next, OP_CLEAR_DS, &clear, sizeof clear);
    } else if (rendered) {
        const p_set_rt target = {.n = 1, .color = {&newer}};
        p_clear_color clear = {0, 0xf, {0, 0, 0, 0}};
        const uint32_t value = UINT32_C(0xd4c3b2a1);
        memcpy(&clear.rgba[0], &value, sizeof value);
        next = put(next, OP_SET_RT, &target, sizeof target);
        next = put(next, OP_CLEAR_COLOR, &clear, sizeof clear);
    } else if (clear_texture) {
        const p_clear_tex clear = {.tex = &newer, .is_uint = 1, .r = new_extent, .mask = 0xf,
                                   .c.u = {UINT32_C(0xd4c3b2a1), 0, 0, 0}};
        next = put(next, OP_CLEAR_TEX, &clear, sizeof clear);
    } else next = put(next, OP_COPY_B2T, &newer_write, sizeof newer_write);
    VkImage old_mixed = VK_NULL_HANDLE, peer_image = VK_NULL_HANDLE, depth_image = VK_NULL_HANDLE;
    VkTexturePresentSource *retained_mixed = NULL;
    tex_t peer = {0}, depth = {0};
    if (replacement_load) {
        CHECK(xf_submit(b, stream, next), "earlier bound image, relocation and later ClearTexture actually complete");
        const VkTexImage *prior = vk_texture_get(&old);
        CHECK(prior && prior->image && prior->image != original_bound && !prior->bound &&
              vk_texture_alias_probe_relocations() == prior_moves + 1, "older native receiver remains the current writer's physical clone");
        old_mixed = prior->image;
        retained_mixed = vk_texture_present_retain_image(prior, old_mixed);
        CHECK(retained_mixed, "retain the actual earlier clone across the next mixed LOAD");
        if (dormant_load) b->texture_release(&old);
        next = put(stream, OP_BEGIN, NULL, 0);
    }
    if (mrt_load) {
        init_tex(&peer, &imported, 0x60000, NVN_FORMAT_R8);
        init_tex(&depth, &imported, 0x70000, NVN_FORMAT_DEPTH32F);
        peer.serial = nvn_new_serial();
        depth.serial = nvn_new_serial();
        peer.d.width = depth.d.width = OLD_WIDTH;
        peer.d.height = depth.d.height = OLD_HEIGHT;
        nvn_tex_layout(&peer.d, &peer.L);
        nvn_tex_layout(&depth.d, &depth.L);
        CHECK(peer.L.size <= 0x10000 && depth.L.size <= 0x10000, "stock MRT peers occupy real disjoint imported intervals");
        const p_set_rt peers = {.n = 1, .color = {&peer}, .depth = &depth};
        const p_clear_color color_seed = {0, 0xf, {67.0f / 255.0f, 0, 0, 1}};
        const p_clear_ds depth_seed = {0.75f, 1, 0, 0};
        next = put(next, OP_SET_RT, &peers, sizeof peers);
        next = put(next, OP_CLEAR_COLOR, &color_seed, sizeof color_seed);
        next = put(next, OP_CLEAR_DS, &depth_seed, sizeof depth_seed);
        CHECK(xf_submit(b, stream, next), "actual independent color and depth peers initialized");
        const VkTexImage *color_owner = vk_texture_get(&peer), *depth_owner = vk_texture_get(&depth);
        CHECK(color_owner && color_owner->image && depth_owner && depth_owner->image &&
              color_owner->image != depth_owner->image && depth_owner->aspect == VK_IMAGE_ASPECT_DEPTH_BIT,
              "MRT peers both have exact native owners");
        peer_image = color_owner->image;
        depth_image = depth_owner->image;
        next = put(stream, OP_BEGIN, NULL, 0);
    }
    if (mixed_load) {
        const p_set_rt selected = {.n = mrt_load ? 2 : 1, .color = {replacement_load ? receiver : &load,
            mrt_load ? &peer : NULL}, .depth = mrt_load ? &depth : NULL};
        next = put(next, OP_SET_RT, &selected, sizeof selected);
        p_viewports viewport = {.n = 1};
        viewport.v[0][2] = replacement_load ? OLD_WIDTH : LOAD_WIDTH;
        viewport.v[0][3] = replacement_load ? OLD_HEIGHT : LOAD_HEIGHT;
        viewport.v[0][5] = 1.0f;
        next = put(next, OP_VIEWPORTS, &viewport, sizeof viewport);
        const p_scissor scissor = {LOAD_X, LOAD_Y, canceled_load ? 0 : LOAD_W, LOAD_H};
        next = put(next, OP_SCISSOR, &scissor, sizeof scissor);
        if (mrt_load) {
            const chmask_st mask = {.mask = 0xf}; // FS writes color 0. Keep color 1 unchanged.
            p_bind_state channels = {.kind = 2}, depth_state = {.kind = 5};
            const ds_st state = {.dt = 1, .dw = 1, .df = NVN_DEPTH_FUNC_ALWAYS};
            memcpy(channels.bytes, &mask, sizeof mask);
            memcpy(depth_state.bytes, &state, sizeof state);
            next = put(next, OP_BIND_STATE, &channels, sizeof channels);
            next = put(next, OP_BIND_STATE, &depth_state, sizeof depth_state);
        }
        const p_bind_program bind = {&pipeline.program, 3};
        next = put(next, OP_BIND_PROGRAM, &bind, sizeof bind);
        const p_bind_handles texture = {.stage = NVN_STAGE_FS, .n = 1, .h = {(UINT64_C(3) << 20) | 17}};
        next = put(next, OP_BIND_TEX, &texture, sizeof texture);
        const p_draw draw = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
        next = put(next, OP_DRAW, &draw, sizeof draw);
    }
    next = put(next, OP_COPY_T2B, &assembled, sizeof assembled);
    CHECK(xf_submit(b, stream, next), "ordered native current-byte alias copy completed");
    uint64_t written_after, combined_after;
    vk_texture_alias_probe_counts(&written_after, &combined_after);
    CHECK(written_after == writes + 2 - (unsigned)migrate + (unsigned)(raster || mode == 8 || mode == 12 || mrt_load || dormant_load) &&
          combined_after == combined + 1 + (unsigned)canceled_load,
          "actual staged renderer producers and one real composed output (actual %llu / %llu)",
          (unsigned long long)(written_after - writes), (unsigned long long)(combined_after - combined));
    b->guest_read((uintptr_t)before, old_pixels);
    b->guest_read((uintptr_t)after, mode == 8 ? LOAD_BYTES : old_pixels);
    CHECK(!memcmp(before, previous, old_pixels), "original native producer survived independently");
    CHECK(!memcmp(after, mode == 8 || replacement_load ? load_expected : expected, mode == 8 ? LOAD_BYTES : old_pixels),
          "all current receiver pixels match exact retained bytes and actual raster");
    unsigned changed = 0;
    for (unsigned i = 0; i < old_pixels; i++) changed += expected[i] != previous[i];
    CHECK(changed == written_bytes, "only the strict known later byte subset changed (%u bytes)", changed);
    const VkTexImage *a = vk_texture_get(receiver), *c = vk_texture_get(&newer);
    CHECK(a && c && a->image && c->image && a->image != c->image && !a->bound && !c->bound,
          "two actually separate native image placements at the same guest address");
    if (migrate) {
        CHECK(a->image != original_bound && vk_texture_alias_probe_relocations() == prior_moves + 1,
              "actual earlier compatible pool image was cloned and re-owned without a guest write");
        printf("probe-vkexec: current image alias mode %d old physical bound image %p retained, clone %p, "
               "later %p through %s; %u earlier and %u current native pixels verified after the real prefix\n",
               mode, (void *)original_bound, (void *)(replacement_load ? old_mixed : a->image), (void *)c->image,
               clear_texture ? "ClearTexture" : "CopyBufferToTexture", old_pixels, old_pixels);
        if (padded_bound)
            printf("probe-vkexec: actual padded bound footprint %ux%u, %u meaningful pixels in %llu held bytes\n",
                   old.d.width, old.d.height, old_pixels, (unsigned long long)old.L.size);
        vk_texture_present_release(retained_bound);
    }
    if (depth_alias) {
        CHECK(c->format == VK_FORMAT_D32_SFLOAT && c->aspect == VK_IMAGE_ASPECT_DEPTH_BIT && c->gpu_owned &&
              c->height == newer.d.height && c->width == newer.d.width, "actual owned D32 pixels supplied the color alias");
        printf("probe-vkexec: current image alias mode %d actual full ClearDepthStencil D32 %ux%u VkImage %p, "
               "%u initialized bytes of %llu physical bytes, %u earlier and %u exact mixed color pixels\n",
               mode, c->width, c->height, (void *)c->image, written_bytes, (unsigned long long)newer.L.size,
               old_pixels, old_pixels);
    }
    if ((mode >= 3 && mode <= 5) || mode == 19) {
        int bad = test_current_alias_sample(b, &imported, &old, expected, mode, VK_NULL_HANDLE);
        if (bad) return bad;
    }
    if (prospective) {
        int bad = test_current_alias_sample(b, &imported, &load, load_expected, mode, a->image);
        if (bad) return bad;
        printf("probe-vkexec: mode %d first sampled receiver %ux%u serial %llu composed directly from two retained native writers; no legacy Tex\n",
               mode, load.d.width, load.d.height, (unsigned long long)load.serial);
        b->texture_release(&load);
    }
    if (raster || mixed_load) b->texture_release(&paint);
    if (mrt_load) {
        const VkTexImage *color_owner = vk_texture_get(&peer), *depth_owner = vk_texture_get(&depth);
        CHECK(color_owner && color_owner->image == peer_image && depth_owner && depth_owner->image == depth_image &&
              peer_image != a->image && depth_image != a->image, "mixed LOAD keeps both exact disjoint peer images");
        uint8_t color_pixels[OLD_BYTES];
        float depth_pixels[OLD_BYTES];
        b->read_texels(&peer, NULL, &old_extent, color_pixels, OLD_WIDTH, sizeof color_pixels);
        b->read_texels(&depth, NULL, &old_extent, (uint8_t *)depth_pixels, OLD_WIDTH * sizeof(float), sizeof depth_pixels);
        for (unsigned y = 0; y < OLD_HEIGHT; y++) for (unsigned x = 0; x < OLD_WIDTH; x++) {
            const unsigned i = y * OLD_WIDTH + x;
            const int drawn = x >= LOAD_X && x < LOAD_X + LOAD_W && y >= LOAD_Y && y < LOAD_Y + LOAD_H;
            CHECK(color_pixels[i] == 67 && depth_pixels[i] == (drawn ? 0.0f : 0.75f),
                  "disjoint MRT peer at (%u,%u) color %u depth %f", x, y, color_pixels[i], depth_pixels[i]);
        }
        printf("probe-vkexec: current image alias mode 15 actual disjoint color %p and depth %p, "
               "%u unchanged second-color pixels and %d native depth writes\n",
               (void *)peer_image, (void *)depth_image, OLD_BYTES, LOAD_W * LOAD_H);
        b->texture_release(&depth);
        b->texture_release(&peer);
    }
    if (replacement_load) {
        CHECK(canceled_load ? a->image == old_mixed : a->image != old_mixed,
              "the existing mixed receiver keeps its exact predecessor until a real draw issues");
        CHECK(vk_texture_alias_probe_load_cancels() == canceled + (unsigned)canceled_load,
              "cancel retires actual tentative replacement placement exactly once");
        if (dormant_load) {
            CHECK(vk_texture_alias_probe_dormant_loads() == dormant + 1 && recreated.serial != old.serial,
                  "the real mixed LOAD acquired a newly initialized object with the earlier dormant placement still held");
            printf("probe-vkexec: current image alias mode 16 actual recreated serial %llu from dormant %llu; "
                   "old %p distinct from selected %p, partial %d native pixels and all %d exact\n",
                   (unsigned long long)recreated.serial, (unsigned long long)old.serial,
                   (void *)old_mixed, (void *)a->image, LOAD_W * LOAD_H, OLD_BYTES);
        }
        printf("probe-vkexec: current image alias mode %d existing %dx%d mixed LOAD, old %p selected %p; "
               "%d actual draw pixels and %d exact current pixels (canceled %d)\n",
               mode, OLD_WIDTH, OLD_HEIGHT, (void *)old_mixed, (void *)a->image,
               canceled_load ? 0 : LOAD_W * LOAD_H, OLD_BYTES, canceled_load);
        vk_texture_present_release(retained_mixed);
    } else if (canceled_load) {
        CHECK(vk_texture_alias_probe_load_cancels() == canceled + 1,
              "the actual newly admitted but skipped native attachment was retired");
        printf("probe-vkexec: current image alias mode 9 prepared a fresh LOAD, canceled its unissued draw and "
               "retired the native placement; 2 actual writers and 8192 preserved current pixels\n");
    } else if (mixed_load) {
        const VkTexImage *new_target = vk_texture_get(&load);
        CHECK(new_target && new_target->image && new_target->image != a->image && new_target->image != c->image &&
              !new_target->bound, "actual new attachment received its independently placed native LOAD contents");
        printf("probe-vkexec: current image alias mode 8 seed from two native writers into %dx%d new LOAD, "
               "%d scissored native draw pixels, %d exact readback pixels\n",
               LOAD_WIDTH, LOAD_HEIGHT, LOAD_W * LOAD_H, LOAD_BYTES);
        b->texture_release(&load);
    }
    b->texture_release(receiver);
    b->texture_release(&newer);
    b->pool_fini(&physical);
    b->pool_fini(&imported);
    if (!mixed_load) printf("probe-vkexec: current image alias mode %d, later box %d,%d %dx%d in %dx%d, actual %s: "
           "preserved %u earlier bytes, applied %u later bytes, and matched %u native receiver pixels\n",
           mode, new_extent.xoffset, new_extent.yoffset, new_extent.width, new_extent.height, newer.d.width, newer.d.height,
           raster ? "render clears and 589 partial LOAD draw pixels" : rendered ? "render clears" :
           clear_texture ? "ClearTexture" : "transfer writes",
           old_pixels - changed, changed, old_pixels);
    return 0;
}

static int test_current_image_ordered_prior_writer(const GfxBackend *b, uint8_t *guest) {
    enum { W = 128, H = 64, OLD_W = 32, OLD_H = 16, X = 5, Y = 3, DW = 19, DH = 9, T = 17, S = 3 };
    pool_t physical = {.magic = NVN_POOL_MAGIC, .flags = 0x189,
        .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .size = 0x100000, .game_mem = guest + 0xa00000};
    pool_t imported = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000,
        .size = 0x100000, .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&physical, &physical) && b->pool_init(&imported, &imported), "prior writer pools");
    tex_t earlier, later, paint, output;
    init_tex(&earlier, &physical, 0, NVN_FORMAT_R8);
    init_tex(&later, &physical, 0, NVN_FORMAT_R8);
    init_tex(&paint, &imported, 0x10000, NVN_FORMAT_R8);
    init_tex(&output, &imported, 0x20000, NVN_FORMAT_RGBA8);
    earlier.serial = nvn_new_serial(); later.serial = nvn_new_serial();
    paint.serial = nvn_new_serial(); output.serial = nvn_new_serial();
    earlier.d.width = OLD_W; earlier.d.height = OLD_H;
    later.d.width = output.d.width = W; later.d.height = output.d.height = H;
    paint.d.width = paint.d.height = 8;
    nvn_tex_layout(&earlier.d, &earlier.L); nvn_tex_layout(&later.d, &later.L);
    nvn_tex_layout(&paint.d, &paint.L); nvn_tex_layout(&output.d, &output.L);
    CHECK(earlier.mem == later.mem && earlier.L.size < later.L.size, "different shapes intentionally overlap exact backing");
    const VkTexImage *old = vk_texture_get_alias_transfer_write(&earlier);
    CHECK(old && old->image && !old->bound, "earlier writable image is independently placed without an earlier producer");
    VkImage original = old->image;

    uint8_t source[64], guest_bytes[4096] = {0};
    CHECK(paint.L.size <= sizeof guest_bytes, "paint plan fits");
    memset(source, 51, sizeof source);
    CHECK(!nvn_swizzle(&paint.d, 0, 0, source, 8, guest_bytes), "stage exact paint texture");
    memcpy(paint.mem, guest_bytes, paint.L.size);
    b->guest_written((uintptr_t)paint.mem, paint.L.size);
    const GfxShader *vs = gfx_shader_own(hdr_fullscreen_spv, sizeof hdr_fullscreen_spv / sizeof *hdr_fullscreen_spv,
        "{\"key\":\"ordered-prior-writer-vs\",\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
    const GfxShader *fs = gfx_shader_own(sample_restore_probe_spv, sizeof sample_restore_probe_spv / sizeof *sample_restore_probe_spv,
        "{\"key\":\"ordered-prior-writer-fs\",\"stage\":\"ps\",\"cbufs\":[],\"sbufs\":[],\"images\":[],"
        "\"textures\":[{\"binding\":1,\"cbuf_slot\":2,\"handle\":8,\"type\":3}]}");
    CHECK(vs && fs, "own actual prior-writer shaders");
    ProbeSampleProgram pipeline = {0};
    probe_sample_program(&pipeline, vs, fs);
    const samp_par native_sampler = {.minf = NVN_MIN_FILTER_NEAREST, .magf = NVN_MAG_FILTER_NEAREST,
        .ws = NVN_WRAP_MODE_CLAMP_TO_EDGE, .wt = NVN_WRAP_MODE_CLAMP_TO_EDGE,
        .wr = NVN_WRAP_MODE_CLAMP_TO_EDGE, .lod_max = 1};
    vk_sampler_register(S, &native_sampler);
    const uint64_t handle = T | ((uint64_t)S << 20);
    vk_texture_register(T, &paint, NULL);
    uint64_t writers, composed, after_writers, after_composed;
    vk_texture_alias_probe_counts(&writers, &composed);
    // A full ClearColor first would make this a projected writer. Record only the real partial raster here.
    _Alignas(8) uint8_t stream[4096];
    uint8_t *tail = put(stream, OP_BEGIN,NULL,0);
    const p_set_rt target = {.n=1,.color={&earlier}};
    tail = put(tail,OP_SET_RT,&target,sizeof target);
    p_viewports viewport = {.n=1};
    viewport.v[0][2] = OLD_W;
    viewport.v[0][3] = OLD_H;
    viewport.v[0][5] = 1.0f;
    tail = put(tail,OP_VIEWPORTS,&viewport,sizeof viewport);
    const p_scissor raster = {X,Y,DW,DH};
    tail = put(tail,OP_SCISSOR,&raster,sizeof raster);
    const p_bind_program program = {&pipeline.program,3};
    tail = put(tail,OP_BIND_PROGRAM,&program,sizeof program);
    const p_bind_handles sample = {.stage=NVN_STAGE_FS,.n=1,.h={handle}};
    tail = put(tail,OP_BIND_TEX,&sample,sizeof sample);
    const p_draw draw = {NVN_DRAW_PRIMITIVE_TRIANGLES,0,3,0,0};
    tail = put(tail,OP_DRAW,&draw,sizeof draw);
    CHECK(xf_submit(b,stream,tail), "old independent partial native raster completed");
    vk_texture_alias_probe_counts(&after_writers, &after_composed);
    old = vk_texture_get(&earlier);
    CHECK(old && old->image == original && !old->bound && old->gpu_owned,
          "real earlier raster retains independently placed image (before %p after %p bound %d owned %d)",
          (void *)original, (void *)(old ? old->image : VK_NULL_HANDLE), old ? old->bound : -1, old ? old->gpu_owned : -1);
    CHECK(after_writers == writers && after_composed == composed,
          "earlier image acquired no fragment proof (writers %llu->%llu images %llu->%llu)",
          (unsigned long long)writers,(unsigned long long)after_writers,
          (unsigned long long)composed,(unsigned long long)after_composed);
    VkTexturePresentSource *retained = vk_texture_present_retain_image(old, original);
    CHECK(retained, "older exact VkImage is retained while the later writer records");
    const NVNcopyRegion small = {0,0,0,OLD_W,OLD_H,1}, whole = {0,0,0,W,H,1};
    uint8_t earlier_pixels[OLD_W * OLD_H];
    b->read_texels(&earlier,NULL,&small,earlier_pixels,OLD_W,sizeof earlier_pixels);
    for (unsigned y=Y;y<Y+DH;y++) for (unsigned x=X;x<X+DW;x++)
        CHECK(earlier_pixels[y*OLD_W+x] == 51, "actual earlier raster pixel (%u,%u) was %u",
              x,y,earlier_pixels[y*OLD_W+x]);

    const p_clear_tex clear = {.tex=&later,.r=whole,.mask=0xf,.c.f={83.0f/255.0f,0,0,1}};
    tail = put(stream, OP_BEGIN,NULL,0);
    tail = put(tail, OP_CLEAR_TEX,&clear,sizeof clear);
    CHECK(xf_submit(b,stream,tail), "later fully initialized native writer completed");
    const VkTexImage *selected = vk_texture_get(&later);
    CHECK(selected && selected->image && selected->image != original && !selected->bound,
          "the current registered input owns its different exact Vulkan image");
    const view_t view = {.set_swizzle=1,.swz={NVN_TEXTURE_SWIZZLE_ZERO,NVN_TEXTURE_SWIZZLE_R,
                                             NVN_TEXTURE_SWIZZLE_R,NVN_TEXTURE_SWIZZLE_ONE}};
    vk_texture_register(T,&later,&view);
    printf("probe-vkexec: mode 17 actual earlier unprojected %dx%d VkImage %p; later full %dx%d VkImage %p\n",
           OLD_W,OLD_H,(void *)original,W,H,(void *)selected->image);
    fflush(stdout);
    CHECK(probe_sample_draw(b,&pipeline.program,&output,handle,0,NULL,NULL),
          "qualified later exact fragment draw succeeds with earlier legacy native owner alive");
    vk_texture_alias_probe_counts(&after_writers,&after_composed);
    CHECK(after_writers == writers+1 && after_composed == composed,
          "use current selected VkImage with one real typed writer and no reconstruction");
    uint32_t pixels[W*H];
    b->read_texels(&output,NULL,&whole,(uint8_t *)pixels,W*sizeof pixels[0],sizeof pixels);
    for (unsigned i=0;i<W*H;i++)
        CHECK(pixels[i] == UINT32_C(0xff535300), "exact later pixel %u was %08x", i,pixels[i]);
    printf("probe-vkexec: mode 17 ordered retained earlier writes, %u prior and %u exact latest native pixels\n",
           DW*DH,W*H);
    vk_texture_present_release(retained);
    b->texture_release(&earlier);b->texture_release(&later);
    b->texture_release(&paint);b->texture_release(&output);
    b->pool_fini(&physical);b->pool_fini(&imported);
    return 0;
}

static int test_current_bound_d32_alias(const GfxBackend *b, uint8_t *guest) {
    enum { COLOR_W = 512, COLOR_H = 1024, PIXELS = COLOR_W * COLOR_H, DEPTH_W = 256, DEPTH_H = 144 };
    pool_t physical = {.magic = NVN_POOL_MAGIC, .flags = 0x189, .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE,
        .size = 0x100000, .game_mem = guest + 0xc00000};
    pool_t imported = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000,
        .size = 0x400000, .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&physical, &physical) && b->pool_init(&imported, &imported), "bound D32 alias pools");
    tex_t color, depth;
    init_tex(&color, &physical, 0, NVN_FORMAT_R8);
    init_tex(&depth, &physical, 0, NVN_FORMAT_DEPTH32F);
    color.serial = nvn_new_serial(), depth.serial = nvn_new_serial();
    color.d.width = COLOR_W, color.d.height = COLOR_H;
    depth.d.width = DEPTH_W, depth.d.height = DEPTH_H;
    nvn_tex_layout(&color.d, &color.L);
    nvn_tex_layout(&depth.d, &depth.L);
    const unsigned useful_depth = DEPTH_W * DEPTH_H * sizeof(float);
    CHECK(color.mem == depth.mem && color.L.size == PIXELS && depth.L.size == 0x40000 && depth.L.size > useful_depth,
          "stock-sized depth is a padded strict overlay of larger color");
    const VkTexImage *p = vk_texture_get(&depth);
    CHECK(p && p->image && p->bound && p->format == VK_FORMAT_D32_SFLOAT,
          "stock-sized depth actually begins physically bound");
    const VkImage original_depth = p->image;
    p = vk_texture_get_alias_transfer_write(&color);
    CHECK(p && p->image && !p->bound && p->image != original_depth, "earlier color has valid independent placement");
    const VkImage original_color = p->image;
    VkTexturePresentSource *prior = vk_texture_present_retain_image(p, original_color);
    CHECK(prior, "real earlier color user retains original native image across depth clear");

    const float value = 0.6875f;
    uint8_t *bytes = malloc(color.L.size), *reference = malloc(PIXELS);
    float *plane = malloc(useful_depth);
    CHECK(bytes && reference && plane, "depth/color reference workspace");
    memset(bytes, 37, color.L.size);
    for (uint32_t i = 0; i < DEPTH_W * DEPTH_H; ++i) plane[i] = value;
    CHECK(!nvn_swizzle(&depth.d, 0, 0, (const uint8_t *)plane, DEPTH_W * sizeof(float), bytes) &&
          !nvn_deswizzle(&color.d, 0, 0, bytes, reference, COLOR_W), "map exact authored depth words into color layout");
    free(plane);
    free(bytes);

    const NVNcopyRegion full = {0, 0, 0, COLOR_W, COLOR_H, 1};
    const p_set_rt color_target = {.n = 1, .color = {&color}}, depth_target = {.depth = &depth};
    const p_clear_color color_clear = {0, 0xf, {37.0f / 255.0f, 0, 0, 1}};
    const p_clear_ds depth_clear = {.depth = value, .depth_mask = 1};
    const p_scissor scissor = {0, 0, DEPTH_W, DEPTH_H};
    uint8_t *before = imported.mem + 0x300000, *after = imported.mem + 0x380000;
    const p_copy_b2t early = {.src = (uintptr_t)before, .tex = &color, .r = full};
    const p_copy_b2t late = {.src = (uintptr_t)after, .tex = &color, .r = full};
    uint64_t writers, composed;
    vk_texture_alias_probe_counts(&writers, &composed);
    _Alignas(8) uint8_t stream[4096];
    uint8_t *next = put(stream, OP_BEGIN, NULL, 0);
    next = put(next, OP_SET_RT, &color_target, sizeof color_target);
    next = put(next, OP_CLEAR_COLOR, &color_clear, sizeof color_clear);
    next = put(next, OP_COPY_T2B, &early, sizeof early);
    next = put(next, OP_SET_RT, &depth_target, sizeof depth_target);
    next = put(next, OP_SCISSOR, &scissor, sizeof scissor);
    next = put(next, OP_CLEAR_DS, &depth_clear, sizeof depth_clear);
    next = put(next, OP_COPY_T2B, &late, sizeof late);
    CHECK(xf_submit(b, stream, next), "real physically aliased color and D32 clear completed");
    uint64_t after_writers, after_composed;
    vk_texture_alias_probe_counts(&after_writers, &after_composed);
    CHECK(after_writers == writers + 1 && after_composed == composed + 1,
          "one independently retained color producer and one composed snapshot using real bound depth");
    p = vk_texture_get(&color);
    CHECK(p && p->image == original_color && !p->bound,
          "earlier authoritative color image remains independently retained");
    p = vk_texture_get(&depth);
    CHECK(p && p->image == original_depth && p->bound && p->gpu_owned,
          "actual D32 clear kept its existing native bound image");
    b->guest_read((uintptr_t)before, PIXELS);
    b->guest_read((uintptr_t)after, PIXELS);
    size_t changed = 0;
    for (size_t i = 0; i < PIXELS; ++i) {
        CHECK(before[i] == 37 && after[i] == reference[i], "bound D32 color pixel %llu", (unsigned long long)i);
        changed += before[i] != after[i];
    }
    CHECK(changed == useful_depth, "only %u meaningful depth bytes changed; physical padding preserved", useful_depth);
    printf("probe-vkexec: mode 20 actual bound D32 %ux%u %p, prior independent color %p retained; "
           "%u meaningful of %llu held depth bytes, %u earlier/current pixels each\n",
           DEPTH_W, DEPTH_H, (void *)original_depth, (void *)original_color, useful_depth,
           (unsigned long long)depth.L.size, PIXELS);
    int sampled = test_current_alias_sample(b, &imported, &color, reference, 20, VK_NULL_HANDLE);
    free(reference);
    vk_texture_present_release(prior);
    b->texture_release(&depth);
    b->texture_release(&color);
    b->pool_fini(&imported);
    b->pool_fini(&physical);
    return sampled;
}


#include "vk_exec_sparse_buffer_probe.inc"

int probe_vkexec(void) {
    vk_hdr_off();   // the present checks read 8-bit swapchain pixels
    layouts_equal = 0;
    main_tid = plat_thread_id();
    const GfxBackend *b = &gfx_vulkan_backend;
    uint64_t heap = 0;
    CHECK(mem_set_heap_size(16ull << 20, &heap) == 0 && heap, "SetHeapSize 16 MiB");
    uint8_t *guest = (uint8_t *)(uintptr_t)heap;
    const char *fallback_probe = cfg_get("NX2NAVI_PROBE_REQUIRED_FALLBACKS");
    int fallback_only = fallback_probe && !strcmp(fallback_probe, "1");
    const char *sparse_buffer_probe = cfg_get("NX2NAVI_PROBE_SPARSE_BUFFER");
    int sparse_buffer_only = sparse_buffer_probe && !strcmp(sparse_buffer_probe, "1");
    const char *present_probe = cfg_get("NX2NAVI_PROBE_PRESENT_ONLY");
    int present_only = present_probe && !strcmp(present_probe, "1");
    const char *bc_probe = cfg_get("NX2NAVI_PROBE_BC_PASS");
    int bc_only = bc_probe && !strcmp(bc_probe, "1");
    const char *sample_probe = cfg_get("NX2NAVI_PROBE_SAMPLE_RESTORE");
    int sample_only = sample_probe && !strcmp(sample_probe, "1");
    const char *pool_probe = cfg_get("NX2NAVI_PROBE_POOL_RESTORE");
    int finite_pool = rt_knob("PROBE_POOL_FINITE", "0") == 1;
    int pool_only = finite_pool || (pool_probe && !strcmp(pool_probe, "1"));
    const char *dispatch_probe = cfg_get("NX2NAVI_PROBE_POOL_DISPATCH");
    int dispatch_only = dispatch_probe && !strcmp(dispatch_probe, "1");
    const char *pointer_probe = cfg_get("NX2NAVI_PROBE_DEVICE_POINTER_TABLE");
    int pointer_only = pointer_probe && !strcmp(pointer_probe, "1");
    const char *graphics_pointer_probe = cfg_get("NX2NAVI_PROBE_GRAPHICS_DEVICE_POINTER");
    int graphics_pointer_only = graphics_pointer_probe && !strcmp(graphics_pointer_probe, "1");
    const char *b2t_probe = cfg_get("NX2NAVI_PROBE_PINNED_B2T");
    int b2t_only = b2t_probe && !strcmp(b2t_probe, "1");
    const char *storage_write_probe = cfg_get("NX2NAVI_PROBE_STORAGE_FIRST_WRITE");
    int storage_write_only = storage_write_probe && !strcmp(storage_write_probe, "1");
    const char *storage_dirty_probe = cfg_get("NX2NAVI_PROBE_STORAGE_DIRTY_WRITE");
    const int storage_dirty_only = storage_dirty_probe && !strcmp(storage_dirty_probe, "1");
    const char *storage_plan_probe = cfg_get("NX2NAVI_PROBE_STORAGE_WRITE_PLAN");
    int storage_plan_only = storage_plan_probe && !strcmp(storage_plan_probe, "1");
    const char *t2t_probe = cfg_get("NX2NAVI_PROBE_ORDERED_T2T");
    int t2t_only = t2t_probe && !strcmp(t2t_probe, "1");
    const char *segmented_probe = cfg_get("NX2NAVI_PROBE_SEGMENTED_BUFFER_TRANSFER");
    int segmented_only = segmented_probe && !strcmp(segmented_probe,"1");
    uint64_t first = 0, last = 0;
    int w = 0, h = 0, inside = 0, xfer_bytes = 0;
    int bad, sync_bad = 0, draw_bad = 0, drawn = 0, tries = 0, dispatched = 0;
    int zero_bytes = 0, nvnx_bytes = 0, mirror_bytes = 0;
    const char *alias_setting = cfg_get("NX2NAVI_PROBE_CURRENT_IMAGE_ALIAS");
    char *alias_end = NULL;
    const long current_alias = alias_setting ? strtol(alias_setting, &alias_end, 10) : 0;
    const int current_alias_only = alias_end && !*alias_end && (current_alias >= 1 && current_alias <= 22);
    if (current_alias_only && current_alias == 20) {
        bad = test_current_bound_d32_alias(b, guest);
    } else if (current_alias_only) {
        bad = current_alias == 17 ? test_current_image_ordered_prior_writer(b, guest) :
                                    test_current_image_alias(b, guest, (int)current_alias);
    } else if (sparse_buffer_only) {
        bad = test_sparse_buffer_producer(b, guest);
    } else if (segmented_only) {
        bad = test_segmented_buffer_transfer(b, guest);
    } else if (fallback_only) {
        bad = test_required_fallbacks(b, guest);
    } else if (storage_write_only || storage_plan_only || storage_dirty_only) {
        bad = test_storage_first_write(b, guest, storage_plan_only && !storage_dirty_only, storage_dirty_only);
    } else if (t2t_only) {
        bad = test_ordered_texture_copy(b, guest);
    } else if (b2t_only) {
        bad = test_pinned_buffer_texture(b, guest);
    } else if (graphics_pointer_only) {
        bad = test_graphics_device_pointer(b, guest);
    } else if (dispatch_only || pointer_only) {
        bad = test_pool_dispatch(b, guest, pointer_only);
    } else if (pool_only) {
        bad = test_pool_restore(b, guest, finite_pool);
    } else if (sample_only) {
        bad = test_sample_restore(b, guest);
    } else if (bc_only) {
        bad = test_bc_pass(b, guest);
    } else if (present_only) {
        bad = test_present(b, guest, &w, &h);
    } else {
        bad = test_sync(b, guest, &first, &last) || test_present(b, guest, &w, &h);
        sync_bad = bad;
        if (!bad) bad = test_clear(b, guest, &inside);
        if (!bad) bad = draw_bad = vk_pipeline_selftest() || test_draw(b, guest, &drawn, &tries, &dispatched);
        if (!bad) bad = test_xfer(b, guest, &xfer_bytes);
        if (!bad) bad = test_c0() || test_zero(b, guest, &zero_bytes);
        if (!bad) bad = test_nvnx(b, guest, &nvnx_bytes);
        if (!bad) bad = test_mirror(b, guest, &mirror_bytes);
        if (!bad) bad = test_empty_pool(b, guest);
        if (!bad) bad = test_pinned_clear(b, guest);
    }
    b->finish();
    if (fallback_only) {
        vk_pipeline_stop_preparation();
        vk_exec_draw_mfb_stop();
        vk_sync_shutdown();
    }
    vk_exec_draw_mfb_stop();
    vk_pipeline_shutdown();   // the pipelines and the draw part's stand-ins go before the device
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    if (sparse_buffer_only) {
        printf("probe-vkexec: sparse buffer producer %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (segmented_only) {
        printf("probe-vkexec: ordered segmented buffer transfer %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (fallback_only) {
        printf("probe-vkexec: required typed fallbacks %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (storage_write_only || storage_plan_only) {
        printf("probe-vkexec: ordered storage %s %s (%ld validation errors)\n",
               storage_plan_only ? "write plan" : "first write", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (t2t_only) {
        printf("probe-vkexec: ordered T2T %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (b2t_only) {
        printf("probe-vkexec: ordered pinned B2T %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (graphics_pointer_only) {
        printf("probe-vkexec: ordered graphics device pointer %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (dispatch_only || pointer_only) {
        printf("probe-vkexec: ordered %s %s (%ld validation errors)\n",
               pointer_only ? "device pointer table" : "pooled compute", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (current_alias_only) {
        printf("probe-vkexec: current image alias %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return !!(bad || errors);
    }
    if (pool_only) {
        printf("probe-vkexec: ordered pool restore %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (sample_only) {
        printf("probe-vkexec: ordered sample restore %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (bc_only) {
        printf("probe-vkexec: ordered BC pass %s (%ld validation errors)\n", bad || errors ? "FAILED" : "passed", errors);
        return bad || errors;
    }
    if (present_only) {
        if (bad || errors) {
            printf("probe-vkexec: present-only FAILED (%ld validation errors)\n", errors);
            return 1;
        }
        printf("probe-vkexec: present-only passed (two exact guest colors through resize to %dx%d, backend shutdown completed)\n", w, h);
        return 0;
    }
    if (bad || errors) {
        printf("probe-vkexec: FAILED (%s, %ld validation errors)\n",
               sync_bad ? "a sync check failed" : draw_bad ? "a draw check failed" : bad ? "a clear or xfer check failed" : "checks passed", errors);
        return 1;
    }
    printf("probe-vkexec: sync part passed (%d submissions, fences %llu..%llu, %d completion callbacks in submit order "
           "on the completion thread, guest word visible after fence_wait, finish drained, present %dx%d exact, "
           "0 validation errors)\n",
           SUBMITS + TAIL + 1, (unsigned long long)first, (unsigned long long)last, SUBMITS + TAIL, w, h);
    printf("probe-vkexec: clear part passed (RGBA8 + D32 %dx%d: image clears outside a pass, scissored attachment "
           "clears in a pass on %d texels, read back exact, 0 validation errors)\n",
           TEX_W, TEX_H, inside);
    printf("probe-vkexec: draw part passed (pipeline self-test; one triangle of census pair %d changed %d of %d texels of the "
           "cleared RGBA8 target; %d census compute shaders dispatched; 0 validation errors)\n",
           tries, drawn, TEX_W * TEX_H, dispatched);
    if (layouts_equal)
        printf("probe-vkexec: layout part passed (the same draw through independent sets 0 and 1 and the shared set 0: "
               "%d of %d texels equal)\n", layouts_equal, TEX_W * TEX_H);
    else
        printf("probe-vkexec: layout part skipped (no graphics pipeline library: shared set 0 only)\n");
    printf("probe-vkexec: xfer part passed (B2B between CPU-visible pools, B2T + T2B round trip and T2T of a %dx%d RGBA8 "
           "texture exact, CLEAR_TEX exact, event WRITE + ADD, timestamp and samples-passed reports in guest memory after "
           "fence_wait, %d bytes compared, 0 validation errors)\n",
           XF_W, XF_H, xfer_bytes);
    printf("probe-vkexec: indirect part passed (DrawElementsIndirect with GPU-copied arguments: count 3 texel-equal to the "
           "direct draw, count 0 draws nothing, %d texels compared; %d census compute shaders dispatched indirectly)\n",
           ind_draw_texels, ind_dispatches);
    printf("probe-vkexec: nvnx part passed (ClearBuffer then ReportValue in GPU order, a CPU-invisible pool fill copied out, "
           "Downsample 64x64 -> 32x32 within 1 of the 2x2 average, stencil view aspect of D24S8, %d bytes compared)\n",
           nvnx_bytes);
    printf("probe-vkexec: c0 and zero parts passed (c0 tables, dispatch size, point size and base instance at their offsets, "
           "unbound slots zero; %d bytes of an unwritten physical-pool texture read zero, NX2NAVI_FIX_ZERO=%s)\n",
           zero_bytes, cfg_value("NX2NAVI_FIX_ZERO"));
    if (vk_mirror_on)
        printf("probe-vkexec: mirror part passed (B2B and fill in a VRAM mirror written back by the fence, a CPU write "
               "between submissions mirrored, CPU bytes beside a fill kept, %d bytes compared)\n", mirror_bytes);
    else
        printf("probe-vkexec: mirror part skipped (memory mode import)\n");
    if (marks_skipped)
        printf("probe-vkexec: pass marks part skipped (no VK_AMD_buffer_marker, or NX2NAVI_PASS_CRUMBS=0)\n");
    else
        printf("probe-vkexec: pass marks part passed (pass, dispatch, pass in one submission: the GPU's last started and "
               "finished mark is the last pass, ring records exact); the report reads:\n%s\n", marks_text);
    printf("probe-vkexec: empty pool part passed (a zero-byte pool at the start of another: no Vulkan objects, lookups "
           "there find the other pool before and after its Finalize)\n");
    return 0;
}
