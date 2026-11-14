// Differential checks of production state compilation/application against frozen scalar replay functions.
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"

enum { VK_BR_GROUP_BUFFER, VK_BR_GROUP_TEXTURE };
#define NVN_STAGE_TASK 6
#define NVN_STAGE_MESH 7
typedef struct DrawCtx {
    uint64_t bind_gen;
    uint32_t prep_dirty[2];
    uint32_t prep_ubo[8][18], prep_ssbo[8][16], prep_tex[8][128], prep_img[8][32];
} DrawCtx;
static DrawCtx *dc;
VkExecFrame vk_exec_frame;
void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
enum { W_BIND };
static void warn_once(int which, const char *fmt, ...) { (void)which; (void)fmt; }
const char *nvn_op_name(uint32_t op) { (void)op; return "test operation"; }
#include "vk_exec_state.h"
#include "scalar_state_oracle.h"

static uint32_t random_state = 0x134aa77d;
static uint32_t rng(void) { random_state ^= random_state << 13; random_state ^= random_state >> 17; return random_state ^= random_state << 5; }
static void fill(void *p, size_t bytes) { uint8_t *b = p; for (size_t i = 0; i < bytes; i++) b[i] = (uint8_t)rng(); }
static _Alignas(8) uint8_t commands[1u << 20], clone[1u << 20];
static size_t used;
static unsigned compiled_blocks, consumed_operations, emitted_writes, fallback_operations;
static unsigned front_blocks, front_draws;

static void append(uint32_t opcode, const void *payload, size_t bytes) {
    size_t size = (sizeof(op_hdr) + bytes + 7) & ~(size_t)7;
    assert(used + size <= sizeof commands);
    op_hdr header = {opcode, (uint32_t)size};
    memcpy(commands + used, &header, sizeof header);
    memset(commands + used + sizeof header, 0, size - sizeof header);
    memcpy(commands + used + sizeof header, payload, bytes);
    used += size;
}

typedef struct { ExecStateBuild blob; size_t bytes; unsigned calls; } Output;
static void emit(void *context, const void *data, size_t bytes) {
    Output *out = context;
    assert(bytes <= sizeof out->blob && (bytes & 7) == 0);
    memcpy(&out->blob, data, bytes), out->bytes = bytes, out->calls++;
}

static void compare_state(const VkExec *initial, VkExec *expected, VkExec *actual, DrawCtx *ref, DrawCtx *got) {
    struct DrawCtx *saved_context = expected->dc;
    expected->dc = actual->dc;
    if (memcmp(expected, actual, sizeof *actual)) {
        const uint8_t *a = (const uint8_t *)expected, *b = (const uint8_t *)actual;
        for (size_t i = 0; i < sizeof *actual; i++) if (a[i] != b[i]) {
            fprintf(stderr, "state mismatch at byte %zu: expected %u got %u\n", i, a[i], b[i]); break;
        }
        abort();
    }
    expected->dc = saved_context;
    assert(ref->bind_gen == got->bind_gen);
    assert((got->prep_dirty[0] & ~ref->prep_dirty[0]) == 0);
    assert((got->prep_dirty[1] & ~ref->prep_dirty[1]) == 0);
    uint32_t need[2] = {0};
    for (unsigned st = 0; st < 8; st++) {
        for (unsigned i = 0; i < 18; i++)
            if (initial->ubo[st][i].addr != actual->ubo[st][i].addr || initial->ubo[st][i].size != actual->ubo[st][i].size)
                need[0] |= got->prep_ubo[st][i];
        for (unsigned i = 0; i < 16; i++)
            if (initial->ssbo[st][i].addr != actual->ssbo[st][i].addr || initial->ssbo[st][i].size != actual->ssbo[st][i].size)
                need[0] |= got->prep_ssbo[st][i];
        for (unsigned i = 0; i < 128; i++)
            if (initial->tex[st][i] != actual->tex[st][i]) need[1] |= got->prep_tex[st][i];
        for (unsigned i = 0; i < 32; i++)
            if (initial->img[st][i] != actual->img[st][i]) need[0] |= got->prep_img[st][i];
    }
    assert((got->prep_dirty[0] & need[0]) == need[0]);
    assert((got->prep_dirty[1] & need[1]) == need[1]);
}

static int ordinary_draw(uint32_t op) { return op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND; }

static void exercise_mode(int front) {
    VkExec initial, expected, actual;
    DrawCtx ref, got;
    fill(&initial, sizeof initial), fill(&ref, sizeof ref);
    ref.prep_dirty[0] = ref.prep_dirty[1] = 0;
    got = ref, actual = expected = initial;
    actual.dc = &got, expected.dc = &ref;
    memcpy(clone, commands, used);
    // Submission clones can contain freshly resolved deferred handle values. Compilation uses the original copy.
    for (size_t at = 0; at < used;) {
        op_hdr *h = (op_hdr *)(clone + at);
        if (h->op == OP_BIND_TEX || h->op == OP_BIND_SAMP || h->op == OP_BIND_IMG) {
            p_bind_handles *b = (p_bind_handles *)(h + 1);
            if (b->src) fill(b->h, (size_t)b->n * sizeof b->h[0]);
        }
        at += h->size;
    }
    for (size_t at = 0; at < used;) {
        Output out = {0};
        NvnExecStateSpan span = {0};
        NvnExecFrontSpan fspan = {0};
        dc = &got;
        int compiled = front ? nvn_exec_front_compile(commands, at, used, emit, &out, &fspan) :
                               nvn_exec_state_compile(commands, at, used, emit, &out, &span);
        if (front) span.end=fspan.end,span.ops=fspan.ops;
        if (compiled) {
            assert(out.calls == 1 && span.end > at && span.end <= used && span.ops >= 2);
            uint32_t count = 0, draws = 0;
            for (size_t pos = at; pos < span.end; count++) {
                const op_hdr *h = (const op_hdr *)(clone + pos);
                dc = &ref;
                if (front && ordinary_draw(h->op)) draws++;
                else reference_state(&expected, h);
                pos += h->size;
                assert(pos <= span.end);
            }
            assert(count == span.ops);
            if (front) { assert(draws==fspan.draws && draws<=16); front_blocks++,front_draws+=draws; }
            dc = &got;
            vk_exec_draw_state_apply(&actual, clone, &out.blob);
            compiled_blocks++, consumed_operations += count, emitted_writes += out.blob.count;
            at = span.end;
        } else {
            assert(out.calls == 0);
            const op_hdr *h = (const op_hdr *)(clone + at);
            if (!front || !ordinary_draw(h->op)) {
                dc = &ref; reference_state(&expected, h);
                dc = &got; reference_state(&actual, h);
            }
            fallback_operations++, at += h->size;
        }
        compare_state(&initial, &expected, &actual, &ref, &got);
    }
}
static void exercise(void) { exercise_mode(0); }

static NvnExecStep primary[8192];
static _Alignas(8) uint8_t primary_data[4u << 20];
static size_t primary_count, primary_bytes;
static unsigned composed_blocks, composed_draws, composed_original_ops;

static void make_primary(int mixed) {
    primary_count=primary_bytes=0;
    for (size_t at=0;at<used;) {
        const op_hdr *h=(const op_hdr *)(commands+at);
        NvnExecStep step={.begin=at,.end=at+h->size,.ops=1,.opcode=h->op};
        Output output={0}; NvnExecStateSpan span={0};
        if ((!mixed || primary_count%3) && nvn_exec_state_compile(commands,at,used,emit,&output,&span)) {
            assert(primary_bytes+output.bytes<=sizeof primary_data);
            step.end=span.end; step.ops=span.ops; step.opcode=NVN_EXEC_STATE;
            step.state_offset=(uint32_t)primary_bytes; step.state_bytes=(uint32_t)output.bytes;
            memcpy(primary_data+primary_bytes,&output.blob,output.bytes); primary_bytes+=output.bytes;
        }
        step.flags=nvn_record_exec_flags(step.opcode);
        assert(primary_count<sizeof primary/sizeof primary[0]); primary[primary_count++]=step;
        at=step.end;
    }
}

static void exercise_composition(int mixed) {
    make_primary(mixed);
    memcpy(clone,commands,used);
    for (size_t at=0;at<used;) {
        op_hdr *h=(op_hdr *)(clone+at);
        if (h->op==OP_BIND_TEX || h->op==OP_BIND_SAMP || h->op==OP_BIND_IMG) {
            p_bind_handles *p=(p_bind_handles *)(h+1);
            if (p->src) fill(p->h,(size_t)p->n*sizeof p->h[0]);
        }
        at+=h->size;
    }
    for (size_t index=0;index<primary_count;) {
        Output output={0}; NvnExecFrontComposeSpan span={0};
        if (!nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,index,emit,&output,&span)) {
            assert(!output.calls); index++; continue;
        }
        assert(output.calls==1 && span.end_step>index && span.end_step<=primary_count);
        assert(span.end==primary[span.end_step-1].end && span.ops<=1024 && span.draws>=2 && span.draws<=16);
        VkExec initial, expected, actual; DrawCtx ref, got;
        fill(&initial,sizeof initial); fill(&ref,sizeof ref); ref.prep_dirty[0]=ref.prep_dirty[1]=0;
        DrawCtx before_ctx=ref;
        got=ref; actual=expected=initial; actual.dc=&got; expected.dc=&ref;
        uint32_t ops=0,draws=0;
        size_t begin=primary[index].begin;
        for (size_t at=begin;at<span.end;) {
            const op_hdr *h=(const op_hdr *)(clone+at); ops++;
            if (ordinary_draw(h->op)) draws++;
            else { dc=&ref; reference_state(&expected,h); }
            at+=h->size; assert(at<=span.end);
        }
        assert(ops==span.ops && draws==span.draws);
        dc=&got; vk_exec_draw_state_apply(&actual,clone,&output.blob);
        compare_state(&initial,&expected,&actual,&ref,&got);
        // Independently recompile the same interval from source. A source compiler can stop sooner at its own
        // scratch bound, so finish a short remainder with scalar state operations and compare the final effects.
        DrawCtx source_ctx=before_ctx;
        VkExec source_state=initial; source_state.dc=&source_ctx;
        for (size_t at=begin;at<span.end;) {
            Output original={0}; NvnExecFrontSpan reference={0};
            if (nvn_exec_front_compile(commands,at,span.end,emit,&original,&reference)) {
                dc=&source_ctx; vk_exec_draw_state_apply(&source_state,clone,&original.blob); at=reference.end;
            } else {
                const op_hdr *h=(const op_hdr *)(clone+at);
                if (!ordinary_draw(h->op)) { dc=&source_ctx; reference_state(&source_state,h); }
                at+=h->size;
            }
        }
        compare_state(&initial,&source_state,&actual,&source_ctx,&got);
        NvnFrontStep summary={begin,span.end,span.ops,span.draws,0,(uint32_t)output.bytes};
        NvnRecordPlan plan={.commands=commands,.bytes=used,.front=&summary,.nfront=1};
        size_t cursor=SIZE_MAX;
        assert(nvn_record_front_slice(&plan,(const op_hdr *)(commands+begin),(const op_hdr *)(commands+span.end),span.draws,&cursor));
        cursor=SIZE_MAX;
        assert(!nvn_record_front_slice(&plan,(const op_hdr *)(commands+begin),(const op_hdr *)(commands+span.end-8),span.draws,&cursor));
        cursor=SIZE_MAX;
        assert(!nvn_record_front_slice(&plan,(const op_hdr *)(commands+begin),(const op_hdr *)(commands+span.end),span.draws-1,&cursor));
        cursor=SIZE_MAX;
        assert(!nvn_record_front_slice(&plan,(const op_hdr *)(commands+begin+8),(const op_hdr *)(commands+span.end),span.draws,&cursor));
        composed_blocks++; composed_draws+=draws; composed_original_ops+=ops;
        index=span.end_step;
    }
}

static void poison_emit(void *context,const void *data,size_t bytes) {
    emit(context,data,bytes);
    memset(primary_data,0xdd,primary_bytes); memset(primary,0xdd,primary_count*sizeof primary[0]);
}

static void composition_boundaries(void) {
    used=0;
    p_bind_buf buffer={0,0,1234,64}; p_draw draw={0};
    for (int i=0;i<3;i++) {
        buffer.addr++; append(OP_BIND_UBO,&buffer,sizeof buffer);
        buffer.addr++; append(OP_BIND_UBO,&buffer,sizeof buffer);
        append(OP_DRAW,&draw,sizeof draw);
    }
    size_t stop=used;
    p_barrier barrier={0}; append(OP_BARRIER,&barrier,sizeof barrier);
    make_primary(0);
    Output output={0}; NvnExecFrontComposeSpan span={0};
    assert(nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
    assert(span.end==stop && span.ops==9 && span.draws==3 && span.end_step==6);
    output.calls=0;
    NvnExecFrontComposeSpan untouched={1,2,3,4}; span=untouched;
    assert(!nvn_exec_front_compose(commands+1,used-1,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
    assert(!output.calls && !memcmp(&span,&untouched,sizeof span));
    NvnExecStep saved=primary[0];
    ExecStateBuild *blob=(ExecStateBuild *)primary_data; ExecStateBuild original={0};
    memcpy(&original,blob,saved.state_bytes);
    for (unsigned bad=0;bad<10;bad++) {
        primary[0]=saved; memcpy(blob,&original,saved.state_bytes);
        switch(bad) {
        case 0: primary[0].end++; break;
        case 1: primary[0].state_offset=(uint32_t)primary_bytes; break;
        case 2: primary[0].state_bytes=8; break;
        case 3: primary[0].ops=1; break;
        case 4: primary[0].reserved=1; break;
        case 5: blob->count=193; break;
        case 6: blob->writes[0].source=used; break;
        case 7: blob->writes[0].destination=sizeof(VkExec); break;
        case 8: blob->bind_increments=primary[0].ops+1; break;
        case 9: blob->writes[0].operation=255; break;
        }
        output.calls=0; NvnExecFrontComposeSpan unchanged={1,2,3,4}; span=unchanged;
        assert(!nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
        assert(!output.calls && !memcmp(&span,&unchanged,sizeof span));
    }
    primary[0]=saved; memcpy(blob,&original,saved.state_bytes);
    output.calls=0;
    assert(nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,poison_emit,&output,&span));
    assert(output.calls==1 && span.end==stop && span.end_step==6 && span.ops==9 && span.draws==3);
}

static void composition_limits(void) {
    p_bind_buf buffer={0,0,1234,64}; p_draw draw={0};
    Output output={0}; NvnExecFrontComposeSpan span={0};
    used=0;
    for (unsigned i=0;i<1021;i++) { buffer.addr++; append(OP_BIND_UBO,&buffer,sizeof buffer); }
    append(OP_DRAW,&draw,sizeof draw); append(OP_DRAW,&draw,sizeof draw);
    size_t stop=used;
    append(OP_BIND_UBO,&buffer,sizeof buffer); append(OP_BIND_UBO,&buffer,sizeof buffer);
    append(OP_DRAW,&draw,sizeof draw);
    make_primary(0);
    assert(nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
    assert(span.ops==1023 && span.draws==2 && span.end==stop && span.end_step==3);
    exercise_composition(0);

    used=0;
    for (unsigned i=0;i<18;i++) {
        buffer.addr++; append(OP_BIND_UBO,&buffer,sizeof buffer);
        buffer.addr++; append(OP_BIND_UBO,&buffer,sizeof buffer);
        append(OP_DRAW,&draw,sizeof draw);
        if (i==15) stop=used;
    }
    make_primary(0);
    assert(nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
    assert(span.ops==48 && span.draws==16 && span.end==stop && span.end_step==32);
    exercise_composition(0);

    // A near-capacity merge can fail after inserting some writes. It must preserve both the last complete
    // primary interval and that interval's state, rather than publishing a partly applied next block.
    used=0;
    for (unsigned i=0;i<2;i++) {
        append(OP_BIND_UBO,&buffer,sizeof buffer); append(OP_BIND_UBO,&buffer,sizeof buffer);
        append(OP_DRAW,&draw,sizeof draw);
    }
    for (unsigned st=0;st<8;st++) for (unsigned slot=0;slot<18;slot++) {
        p_bind_buf value={(int)st,(int)slot,rng(),64}; append(OP_BIND_UBO,&value,sizeof value);
    }
    for (unsigned st=0;st<6;st++) for (unsigned slot=0;slot<16;slot++) {
        p_bind_buf value={(int)st,(int)slot,rng(),64}; append(OP_BIND_SSBO,&value,sizeof value);
    }
    append(OP_DRAW,&draw,sizeof draw);
    make_primary(0);
    assert(nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
    assert(span.end_step==5 && span.end==primary[4].end && span.ops==191 && span.draws==2 && output.blob.count==185);
    exercise_composition(0);

    // The same rare path must also accept an almost-full block when its writes replace existing slots.
    used=0;
    for (unsigned round=0;round<2;round++) {
        for (unsigned st=0;st<8;st++) for (unsigned slot=0;slot<18;slot++) {
            p_bind_buf value={(int)st,(int)slot,rng(),64}; append(OP_BIND_UBO,&value,sizeof value);
        }
        append(OP_DRAW,&draw,sizeof draw);
    }
    make_primary(0);
    assert(nvn_exec_front_compose(commands,used,primary,primary_count,primary_data,primary_bytes,0,emit,&output,&span));
    assert(span.end==used && span.ops==290 && span.draws==2 && output.blob.count==144);
    exercise_composition(0);
}

static void add_random(void) {
    switch (rng() % 21) {
    case 0: case 1: { p_bind_buf b = {(int)(rng()%8), (int)(rng()%18), rng(), rng()%4096}; append(OP_BIND_UBO,&b,sizeof b); break; }
    case 2: { p_bind_buf b = {(int)(rng()%6), (int)(rng()%16), rng(), rng()%4096}; append(OP_BIND_SSBO,&b,sizeof b); break; }
    case 3: case 4: case 5: {
        p_bind_handles b; fill(&b,sizeof b); b.stage=rng()%8; b.first=rng()%16; b.n=rng()%17; b.src=rng()&1;
        unsigned op = rng()%3; append(op==0?OP_BIND_TEX:op==1?OP_BIND_SAMP:OP_BIND_IMG,&b,offsetof(p_bind_handles,h)+(size_t)b.n*sizeof b.h[0]); break;
    }
    case 6: { p_bind_vattr b; fill(&b,sizeof b); b.first=rng()%17; b.n=rng()%(17-b.first); append(OP_BIND_VATTR,&b,offsetof(p_bind_vattr,st)+(size_t)b.n*sizeof b.st[0]); break; }
    case 7: { p_bind_vstream b; fill(&b,sizeof b); b.first=rng()%17; b.n=rng()%(17-b.first); append(OP_BIND_VSTREAM,&b,offsetof(p_bind_vstream,st)+(size_t)b.n*sizeof b.st[0]); break; }
    case 8: { p_bind_vbuf b; fill(&b,sizeof b); b.first=rng()%17; b.n=rng()%(17-b.first); append(OP_BIND_VBUF,&b,offsetof(p_bind_vbuf,r)+(size_t)b.n*sizeof b.r[0]); break; }
    case 9: { p_bind_state b; fill(&b,sizeof b); b.kind=rng()%7; append(OP_BIND_STATE,&b,sizeof b); break; }
    case 10: { struct { uint32_t offset,size; uint8_t data[128]; } b; fill(&b,sizeof b); b.offset=rng()%897; b.size=rng()%129; append(OP_PUSH_CC,&b,8+b.size); break; }
    case 11: { p_viewports b; fill(&b,sizeof b); b.first=rng()%17; b.n=rng()%(17-b.first); append(OP_VIEWPORTS,&b,sizeof b); break; }
    case 12: { p_stencil b; fill(&b,sizeof b); b.which=rng()%4; b.face=rng()%4; append(OP_STENCIL,&b,sizeof b); break; }
    case 13: { p_scissor b; fill(&b,sizeof b); append(OP_SCISSOR,&b,sizeof b); break; }
    case 14: { p_depth_range b; fill(&b,sizeof b); append(OP_DEPTH_RANGE,&b,sizeof b); break; }
    case 15: { p_poly_offset b; fill(&b,sizeof b); append(OP_POLY_OFFSET,&b,sizeof b); break; }
    case 16: { p_blend_color b; fill(&b,sizeof b); append(OP_BLEND_COLOR,&b,sizeof b); break; }
    case 17: { p_flag b={(int32_t)rng()}; append((rng()&1)?OP_DEPTH_CLAMP:OP_RASTER_DISCARD,&b,sizeof b); break; }
    case 18: { p_prim_restart b={(int32_t)rng(),rng()}; append(OP_PRIM_RESTART,&b,sizeof b); break; }
    case 19: { p_copy_stride b={(int64_t)rng()}; append(OP_COPY_STRIDE,&b,sizeof b); break; }
    case 20: { p_bind_program b={(const prog_t *)(uintptr_t)(rng() & ~7u),(int32_t)rng()}; append(OP_BIND_PROGRAM,&b,sizeof b); break; }
    }
}

static void boundaries(void) {
    p_bind_buf buffer = {0,0,1234,64};
    p_draw draw = {0};
    used=0;
    append(OP_BIND_UBO,&buffer,sizeof buffer);
    buffer.addr=4567; append(OP_BIND_UBO,&buffer,sizeof buffer);
    size_t stop=used; append(OP_DRAW,&draw,sizeof draw);
    Output out={0}; NvnExecStateSpan span={0};
    assert(nvn_exec_state_compile(commands,0,used,emit,&out,&span));
    assert(span.end==stop && span.ops==2 && out.blob.count==1);
    out.calls=0;
    assert(!nvn_exec_state_compile(commands,stop,used,emit,&out,&span) && !out.calls);

    p_bind_handles handles={.stage=0,.first=0,.n=33};
    used=0; append(OP_BIND_TEX,&handles,sizeof handles);
    assert(!nvn_exec_state_compile(commands,0,used,emit,&out,&span) && !out.calls);
    handles.n=-1; used=0; append(OP_BIND_TEX,&handles,sizeof handles);
    assert(!nvn_exec_state_compile(commands,0,used,emit,&out,&span) && !out.calls);
    p_bind_vattr attr={.first=0,.n=4};
    used=0; append(OP_BIND_VATTR,&attr,offsetof(p_bind_vattr,st));
    assert(!nvn_exec_state_compile(commands,0,used,emit,&out,&span) && !out.calls);
    ((op_hdr *)commands)->size=0;
    assert(!nvn_exec_state_compile(commands,0,used,emit,&out,&span) && !out.calls);

    VkExec ex={0}; DrawCtx context={0}; ex.dc=&context;
    context.prep_ubo[0][0]=1,context.prep_tex[0][0]=2;
    ex.ubo[0][0]=(p_bind_buf){0,0,100,64};
    used=0; buffer=ex.ubo[0][0]; buffer.addr=200; append(OP_BIND_UBO,&buffer,sizeof buffer);
    buffer.addr=100; append(OP_BIND_UBO,&buffer,sizeof buffer);
    assert(nvn_exec_state_compile(commands,0,used,emit,&out,&span));
    vk_exec_draw_state_apply(&ex,commands,&out.blob);
    assert(context.bind_gen==2 && context.prep_dirty[0]==0 && context.prep_dirty[1]==0);
    buffer.addr=300; memcpy(commands+sizeof(op_hdr),&buffer,sizeof buffer);
    buffer.addr=400; memcpy(commands+((op_hdr *)commands)->size+sizeof(op_hdr),&buffer,sizeof buffer);
    vk_exec_draw_state_apply(&ex,commands,&out.blob);
    assert(context.bind_gen==4 && context.prep_dirty[0]==1 && context.prep_dirty[1]==0);
}

static void interval_oracle(void) {
    uint32_t saved_random = random_state;
    uint8_t source[4096], initial[1024], expected[1024], actual[1024];
    fill(source, sizeof source);
    for (unsigned trial = 0; trial < 2000; trial++) {
        ExecStateBuild b = {0};
        fill(initial, sizeof initial);
        memcpy(expected, initial, sizeof expected);
        for (unsigned step = 0; step < 50; step++) {
            uint32_t dst = rng() % sizeof expected;
            uint32_t bytes = rng() % 129;
            if (bytes > sizeof expected - dst) bytes = sizeof expected - dst;
            size_t src = rng() % (sizeof source - sizeof expected);
            unsigned operation = rng() % 3;
            if (operation == ES_BOOL) bytes = 1;
            // Exercise exact replacement, nested splitting, adjacency and multi-interval coverage explicitly.
            if (step < 7) {
                static const uint32_t starts[] = {100, 120, 120, 140, 90, 400, 0};
                static const uint32_t sizes[] = {100, 20, 20, 60, 130, 100, 600};
                dst = starts[step], bytes = sizes[step], operation = step & 1;
            }
            assert(b.count + 2 <= EXEC_STATE_WRITES);
            exec_state_write(&b, dst, bytes, src, operation, ES_NO_NOTIFY);
            if (operation == ES_COPY) memcpy(expected + dst, source + src, bytes);
            else if (operation == ES_ZERO) memset(expected + dst, 0, bytes);
            else {
                int32_t flag;
                memcpy(&flag, source + src, sizeof flag);
                expected[dst] = !!flag;
            }
            memcpy(actual, initial, sizeof actual);
            for (unsigned i = 0; i < b.count; i++) {
                const ExecStateWrite *w = &b.writes[i];
                assert(w->bytes && w->destination + w->bytes <= sizeof actual);
                assert(!i || b.writes[i-1].destination + b.writes[i-1].bytes <= w->destination);
                if (w->operation == ES_COPY) memcpy(actual + w->destination, source + w->source, w->bytes);
                else if (w->operation == ES_ZERO) memset(actual + w->destination, 0, w->bytes);
                else {
                    int32_t flag;
                    assert(w->bytes == 1);
                    memcpy(&flag, source + w->source, sizeof flag);
                    actual[w->destination] = !!flag;
                }
            }
            assert(!memcmp(expected, actual, sizeof expected));
            assert(exec_state_blob_valid(&b, offsetof(ExecStateBuild, writes) + b.count * sizeof b.writes[0], sizeof source));
        }
    }
    ExecStateBuild malformed = {.count = 2, .writes = {{0, 32, 8, ES_COPY, 0}, {8, 16, 8, ES_COPY, 0}}};
    size_t bytes = offsetof(ExecStateBuild, writes) + 2 * sizeof(ExecStateWrite);
    assert(!exec_state_blob_valid(&malformed, bytes, sizeof source));
    malformed.writes[1].destination = 36;
    assert(!exec_state_blob_valid(&malformed, bytes, sizeof source));
    malformed.writes[1].destination = 40;
    assert(exec_state_blob_valid(&malformed, bytes, sizeof source));
    Output out = {0};
    exec_state_emit(&malformed, emit, &out);
    assert(out.calls == 1 && out.bytes == bytes && !memcmp(&malformed, &out.blob, bytes));
    random_state = saved_random;
    puts("interval_oracle: 100000 updates checked after every write; overlap/split/order/boolean/source offsets and malformed blobs passed");
}

int main(void) {
    interval_oracle();
    boundaries();
    composition_boundaries();
    composition_limits();
    for (unsigned trial=0; trial<300; trial++) {
        used=0;
        for (unsigned i=0; i<200; i++) add_random();
        exercise();
    }
    used=0;
    for(unsigned i=0;i<1200;i++) { p_bind_buf b={0,0,i,64}; append(OP_BIND_UBO,&b,sizeof b); }
    exercise();
    for (unsigned trial=0;trial<60;trial++) {
        used=0;
        for (unsigned i=0;i<65;i++) {
            add_random(); add_random(); add_random();
            p_draw draw={0}; append(OP_DRAW,&draw,sizeof draw);
        }
        exercise_mode(1);
        exercise_composition(0);
        exercise_composition(1);
    }
    assert(front_blocks>=240 && front_draws>=3840);
    used=0;
    for(unsigned st=0;st<8;st++) for(unsigned slot=0;slot<18;slot++) {
        p_bind_buf b={(int)st,(int)slot,rng(),64}; append(OP_BIND_UBO,&b,sizeof b);
    }
    for(unsigned st=0;st<6;st++) for(unsigned slot=0;slot<16;slot++) {
        p_bind_buf b={(int)st,(int)slot,rng(),64}; append(OP_BIND_SSBO,&b,sizeof b);
    }
    exercise();
    assert(compiled_blocks>300 && consumed_operations>60000 && emitted_writes<consumed_operations*2);
    assert(composed_blocks>=400 && composed_draws>=7000);
    printf("exec_front_compose: %u summaries, %u draws, %u original operations; primary/single composition, scalar/source oracles, clone values, boundaries and emitter relocation passed\n",composed_blocks,composed_draws,composed_original_ops);
    printf("exec_state_check: %u blocks, %u source ops, %u retained writes, %u scalar boundaries; front %u blocks/%u draws; differential state, deferred clone values, dirty dependencies and bounded splits passed\n",compiled_blocks,consumed_operations,emitted_writes,fallback_operations,front_blocks,front_draws);
    return 0;
}
