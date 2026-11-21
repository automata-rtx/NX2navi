/* Bind and issue an ordinary retained provider without selecting memory again. */
typedef uint64_t VkBuffer;
typedef struct { VkBuffer buffer; uint64_t offset, range; } VkDescriptorBufferInfo;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { uint64_t backing, offset, size; } RcSourceRange;
typedef struct { uint64_t address, mapping; RcSourceRange range; } RcInterval;
typedef struct { uint64_t allocation, incarnation, offset; } RcRegistryBufferLocation;
typedef struct { VkBuffer buffer; uint64_t offset; int live; } VkMemSpan;
typedef struct { int refs, live; RcRegistryBufferLocation location; } VkMemSpanCapture;
typedef struct { const GfxShaderRes *res; uint32_t count,kind,type,stage,set,binding; } VkPipeBinding;
typedef struct {
    uint32_t set,binding;
    uint64_t addr,size,authority_size;
    VkDescriptorBufferInfo descriptor;
    VkMemSpan span;
    const GfxShaderRes *resource;
    VkMemSpanCapture *native;
    RcInterval *mapped;
    uint32_t mapped_count;
    int visible,written,recorded,writer_noted,borrowed_writer;
    const VkPipeBinding *selected_binding;
    int borrowed_initializer;
} EffectStorageCapture;
typedef struct {
    unsigned prepared, recorded, issued;
    uint64_t cb;
    uint64_t bytes;
} VkTextureAliasRead;
typedef struct {
    EffectStorageCapture storage;
    p_bind_buf bound;
    VkTextureAliasRead *seed;
} OrdinaryStorageInput;
typedef uint64_t VkCommandBuffer;
typedef struct { unsigned revision; } VkPipeInfo;
typedef struct { unsigned attached_source; } VkExecGraphicsPassShape;
typedef struct {
    const VkPipeInfo *info;
    RcRegistryPosition read;
    uint32_t count;
    int recorded;
    OrdinaryStorageInput entries[2];
} OrdinaryStorageInputs;
typedef struct { EffectStorageCapture *storage; uint32_t storage_count; RcRegistryPosition read_position; } EffectInputCapture;
typedef struct { uint8_t key[8]; } GfxShader;
typedef struct { int present; } ResourceCommandEffects;
typedef struct {
    uint32_t opcode;
    uint64_t program_serial;
    EffectInputCapture *capture[8];
    const GfxShader *shader[8];
    ResourceCommandEffects *commands;
    OrdinaryStorageInputs *storage_writes;
    struct { OrdinaryStorageInputs *storage_writes; } compat_descriptors;
} EffectDeclaration;
typedef struct {
    uint64_t program,shader,buffer,offset,size;
    uint32_t opcode,stage,set,binding;
    int borrowed;
} ResourceBufferIssueTrace;
static EffectDeclaration *effect_current;
static struct { VkBuffer sink; } Z={97};
static struct { uint64_t sbo_max; } LIM={1024};
static RcRegistryPosition now={42,80};
static unsigned native_records, accesses, issued, refused, retained_outputs, traced, dispatches;
static void vk_texture_alias_buffer_write_input_issued(VkTextureAliasRead *seed,uint64_t cb) {
    if (!seed || !seed->prepared || !seed->recorded || seed->cb!=cb || !dispatches) abort();
    if (!seed->issued) seed->issued++;
}
static int resource_effects_image_trace_enabled(void) { return 1; }
static const uint8_t *gfx_shader_key(const GfxShader *shader) { return shader->key; }
static void resource_effects_trace_buffer_issued(ResourceCommandEffects *effects,const RcInterval *mapped,
                                                  uint32_t count,ResourceBufferIssueTrace trace) {
    if (!effects || !effects->present || count!=1 || mapped->mapping!=17 ||
        trace.program!=33 || trace.shader!=UINT64_C(0x1200000000000034) ||
        trace.buffer!=51 || trace.offset!=256 || trace.size!=60 || trace.opcode!=27 ||
        trace.stage!=5 || trace.set!=2 || trace.binding!=4 || trace.borrowed) abort();
    traced++;
}
enum { ZERO_BYTES=128 };
static int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void) { return now; }
static VkDescriptorBufferInfo zero_info(uint64_t range) { return (VkDescriptorBufferInfo){98,0,range}; }
static void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *why) {
    (void)ex; (void)opcode; (void)why; refused++;
}
static int vk_mem_span_record_input(VkMemSpanCapture *hold, const VkMemSpan *span) {
    return hold->live && span->live && hold->location.offset==span->offset;
}
static VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *hold) {
    ++hold->refs; return hold;
}
static void vk_exec_record_native_input_provider(VkExec *ex, VkMemSpanCapture **hold) {
    (void)ex; if (!*hold) abort(); --(*hold)->refs; *hold=NULL; native_records++;
}
static int note_buffer_written_provider(VkExec *ex,uint64_t address,uint64_t bytes,const VkMemSpan *span,
                                        int visible,const RcInterval *mapped,uint32_t count,VkMemSpanCapture *hold) {
    (void)ex;
    if (!hold || hold->refs!=1 || span->offset!=hold->location.offset || visible || count!=1 ||
        mapped->address!=address || mapped->range.size!=bytes || mapped->mapping!=17 ||
        hold->location.allocation!=125) abort();
    accesses++;
    retained_outputs++;
    return 1;
}
static void vk_mem_gpu_wrote_shader(const VkMemSpan *span,uint64_t bytes) {
    if (!span->live || bytes!=64) abort();
    issued++;
}
static void vk_usnap_drop(uint64_t address,uint64_t bytes) {
    if (!address || bytes!=64) abort();
}
static unsigned transfer_records, pass_splits;
static int nvn_stage_of(int stage) { return stage; }
static EffectStorageCapture *effect_storage_find(EffectInputCapture *capture,uint32_t set,uint32_t binding) {
    for (uint32_t i=0;i<capture->storage_count;i++)
        if (capture->storage[i].set==set && capture->storage[i].binding==binding) return &capture->storage[i];
    return NULL;
}
static int vk_texture_alias_buffer_write_input_current(const VkMemSpanCapture *hold,const VkMemSpan *span,
                                                        uint64_t address,uint64_t size,VkTextureAliasRead *seed) {
    return address && size && hold->live && span->live && hold->location.offset==span->offset &&
        (!seed || (seed->prepared && seed->bytes==size));
}
static int ordinary_storage_seed_attachments_disjoint(const VkExec *ex,const VkExecGraphicsPassShape *pass,
                                                        const OrdinaryStorageInput *input) {
    (void)ex;
    return input->seed && (!pass || !pass->attached_source);
}
static void vk_texture_alias_buffer_write_input_record(VkTextureAliasRead *seed,uint64_t cb,uint64_t scene) {
    if (!seed->prepared || seed->recorded || scene!=now.scene) abort();
    seed->recorded++;
    seed->cb=cb;
    transfer_records++;
}
static int vk_exec_record_between_renderings(VkExec *ex,int (*record)(uint64_t,void *),void *opaque) {
    pass_splits++;
    return record(ex->cb,opaque);
}
/* production binding and issued functions */
static int run_provider(void) {
#define VERIFY(x) do { if (!(x)) { puts("failed: " #x); return 1; } } while (0)
    VkExec ex={.cb=46};
    GfxShaderRes resource={0};
    VkPipeBinding physical_binding={.res=&resource,.count=1,.stage=5,.set=2,.binding=4};
    VkPipeBinding other_binding=physical_binding;
    VkMemSpanCapture hold={.refs=1,.live=1,.location={125,0,256}};
    RcInterval mapping={.address=0x8000000400,.mapping=17,.range={13,1024,64}};
    EffectStorageCapture storage={.set=2,.binding=4,.addr=mapping.address,.size=60,.authority_size=64,
        .descriptor={51,256,60},.span={51,256,1},.native=&hold,.mapped=&mapping,.mapped_count=1,
        .written=1,.selected_binding=&physical_binding,.resource=&resource};
    EffectInputCapture capture={.storage=&storage,.storage_count=1,.read_position=now};
    const GfxShader shader={{0x12,0,0,0,0,0,0,0x34}};
    ResourceCommandEffects commands={.present=1};
    EffectDeclaration declaration={.opcode=27,.program_serial=33,.capture={[5]=&capture},
        .shader={[5]=&shader},.commands=&commands};
    effect_current=&declaration;
    VkDescriptorBufferInfo output={0};
    /* Real existing physical bytes and normalized actual access flow through one held placement. */
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&physical_binding,1,&output) && output.buffer==51 &&
           output.offset==256 && output.range==60 && accesses==1 && native_records==1);
    VERIFY(hold.refs==1 && !issued && retained_outputs==1);
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&physical_binding,1,&output) && accesses==1);
    ordinary_storage_writers_issued(&declaration);
    ordinary_storage_writers_issued(&declaration);
    VERIFY(issued==1 && traced==1);
    /* A known copied prior seed cannot qualify an output before actual writer issue. */
    VkTextureAliasRead seed={.prepared=1,.recorded=1,.cb=ex.cb};
    OrdinaryStorageInputs inputs={.recorded=1,.count=1,.entries={{.storage=storage,.seed=&seed}}};
    declaration.storage_writes=&inputs;
    VERIFY(ordinary_storage_input_find(ordinary_storage_effect_inputs(&declaration),&physical_binding)==&inputs.entries[0]);
    VERIFY(!ordinary_storage_input_find(ordinary_storage_effect_inputs(&declaration),&other_binding));
    declaration.storage_writes=NULL;
    declaration.compat_descriptors.storage_writes=&inputs;
    VERIFY(ordinary_storage_input_find(ordinary_storage_effect_inputs(&declaration),&physical_binding)==&inputs.entries[0]);
    storage.borrowed_initializer=1;
    ordinary_storage_inputs_issued(&ex,&declaration);
    ordinary_storage_writers_issued(&declaration);
    VERIFY(!seed.issued);
    /* A real recorded work item publishes the seed-based result only once. */
    storage.recorded=1;
    dispatches++;
    ordinary_storage_inputs_issued(&ex,&declaration);
    ordinary_storage_writers_issued(&declaration);
    ordinary_storage_inputs_issued(&ex,&declaration);
    ordinary_storage_writers_issued(&declaration);
    VERIFY(seed.issued==1 && issued==2 && traced==2);
    /* Keep subsequent checks anchored on the original baseline counters. */
    issued=traced=1;
    storage.borrowed_initializer=0;
    declaration.storage_writes=NULL;
    /* Different selected binding cannot reuse the same translated resource or its publication. */
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&other_binding,1,&output) && !output.buffer && refused==1);
    /* A canceled earlier attempt or stale placement never acquires a native use or publishes. */
    storage.recorded=storage.writer_noted=0;
    hold.live=0;
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&physical_binding,1,&output) && !output.buffer && refused==2);
    ordinary_storage_writers_issued(&declaration);
    VERIFY(issued==1 && accesses==1 && native_records==1);
    hold.live=1;
    capture.read_position.ordinal-=2;
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&physical_binding,1,&output) && !output.buffer && refused==3);
    capture.read_position=now;
    /* A real absence uses only the write sink. Read-only absence uses the zero source. */
    storage.native=NULL;
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&physical_binding,1,&output) && output.buffer==Z.sink);
    storage.written=0;
    VERIFY(effect_ordinary_descriptor(&ex,&capture,&storage,&physical_binding,0,&output) && output.buffer==98);
    ordinary_storage_writers_issued(&declaration);
    VERIFY(issued==1 && traced==1 && accesses==1 && native_records==1 && hold.refs==1);
    capture.storage_count=0;
    ordinary_storage_writers_issued(&declaration);
    puts("retained physical SSBO actual offset/range, issued write, absent sink, stale layout/cut and no-output cancellation");
    return 0;
#undef VERIFY
}

static int run_seed_cohort(VkExec *ex) {
#define CHECK(x) do { if (!(x)) { puts("failed seed cohort: " #x); return 1; } } while (0)
    const GfxShaderRes resources[2]={{.cbuf_slot=0,.handle=68,.count=1},{.cbuf_slot=0,.handle=72,.count=1}};
    const VkPipeBinding bindings[2]={{.res=&resources[0],.stage=5,.binding=4},{.res=&resources[1],.stage=5,.binding=8}};
    VkMemSpanCapture holds[2]={{.live=1,.location={1,0,512}},{.live=1,.location={2,0,1024}}};
    const VkPipeInfo info={.revision=4}, changed={.revision=5};
    VkTextureAliasRead seeds[2]={{.prepared=1,.bytes=64},{.prepared=1,.bytes=64}};
    OrdinaryStorageInputs inputs={.info=&info,.read=now,.count=2};
    EffectStorageCapture actual[2];
    for (uint32_t i=0;i<2;i++) {
        const p_bind_buf bound=ex->ssbo[5][i+8];
        actual[i]=(EffectStorageCapture){.binding=bindings[i].binding,.resource=&resources[i],
            .borrowed_initializer=1,.addr=bound.addr,.size=60,.authority_size=64,.native=&holds[i],
            .span={51+i,holds[i].location.offset,1},.selected_binding=&bindings[i]};
        inputs.entries[i]=(OrdinaryStorageInput){.storage=actual[i],.bound=bound,.seed=&seeds[i]};
    }
    EffectInputCapture capture={.storage=actual,.storage_count=2,.read_position=now};
    EffectDeclaration declaration={.capture={[5]=&capture},.storage_writes=&inputs};
    VkExecGraphicsPassShape pass={.attached_source=1};
    ex->cb=123;
    effect_current=&declaration;
    CHECK(!ordinary_storage_writes_record(ex,&info,&pass,1) && !transfer_records && !pass_splits);
    pass.attached_source=0;
    CHECK(!ordinary_storage_writes_record(ex,&changed,&pass,1) && !transfer_records && !pass_splits);
    holds[1].live=0;
    CHECK(!ordinary_storage_writes_record(ex,&info,&pass,1) && !transfer_records && !pass_splits);
    holds[1].live=1;
    actual[1].borrowed_initializer=0;
    CHECK(!ordinary_storage_writes_record(ex,&info,&pass,1) && !transfer_records && !pass_splits);
    actual[1].borrowed_initializer=1;
    inputs.read.ordinal-=2;
    CHECK(!ordinary_storage_writes_record(ex,&info,&pass,1) && !transfer_records && !pass_splits);
    inputs.read=now;
    CHECK(ordinary_storage_writes_record(ex,&info,&pass,1) && transfer_records==2 && pass_splits==1);
    CHECK(seeds[0].cb==123 && seeds[1].cb==123 && !seeds[0].issued && !seeds[1].issued);
    CHECK(!ordinary_storage_writes_record(ex,&info,&pass,1) && transfer_records==2);
    ordinary_storage_inputs_issued(ex,&declaration);
    ordinary_storage_writers_issued(&declaration);
    CHECK(!seeds[0].issued && !seeds[1].issued);
    inputs.recorded=0;
    inputs.entries[0].seed=inputs.entries[1].seed=NULL;
    CHECK(ordinary_storage_writes_record(ex,&info,NULL,0) && transfer_records==2 && pass_splits==1);
    effect_current=NULL;
    puts("writable storage validates the cohort before copy; stale, changed layout, overlap, canceled and direct controls");
    return 0;
#undef CHECK
}
