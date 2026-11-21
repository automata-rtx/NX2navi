// Internal to gfx_shader.c. Build the scalar index slice once from the selected translated module.
// Unsupported control flow or an escaped descriptor pointer leaves the full-table proof in effect.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    PX_ENTRY=15, PX_TYPE_BOOL=20, PX_TYPE_INT=21, PX_TYPE_FLOAT=22, PX_TYPE_VECTOR=23, PX_TYPE_IMAGE=25,
    PX_TYPE_SAMPLER=26, PX_TYPE_ARRAY=28, PX_TYPE_RUNTIME_ARRAY=29, PX_TYPE_STRUCT=30, PX_TYPE_POINTER=32,
    PX_TRUE=41, PX_FALSE=42, PX_CONST=43, PX_CONST_NULL=46, PX_FN=54, PX_PARAM=55,
    PX_END=56, PX_VAR=59, PX_LOAD=61, PX_STORE=62, PX_CHAIN=65, PX_IN_CHAIN=66, PX_ARRAY_LENGTH=68,
    PX_DECORATE=71, PX_MEMBER_DECORATE=72, PX_GROUP_DECORATE=74, PX_GROUP_MEMBER_DECORATE=75,
    PX_COMPOSITE_EXTRACT=81, PX_COPY=83, PX_SAMPLED_IMAGE=86,
    PX_BITCAST=124, PX_ADD=128, PX_SUB=130, PX_MUL=132, PX_NOT=168, PX_SELECT=169,
    PX_EQ=170, PX_NE=171, PX_SGT=173, PX_RSHIFT=194, PX_LSHIFT=196, PX_OR=197, PX_XOR=198,
    PX_AND=199, PX_LABEL=248, PX_EXT=12, PX_EXT_IMPORT=11
};
enum {
    PI_VALUE=1, PI_U32, PI_UMIN,
    PI_NODE_LIMIT=384, PI_RESULT_LIMIT=48, PI_CHAIN_LIMIT=10, PI_RECURSION_LIMIT=96,
    PI_DESCRIPTOR_BINDING=33, PI_DESCRIPTOR_SET=34, PI_ARRAY_STRIDE=6, PI_MEMBER_OFFSET=35,
    PI_RELAXED_PRECISION=0, PI_NO_SIGNED_WRAP=4469, PI_NO_UNSIGNED_WRAP=4470,
    PI_STORAGE_UNIFORM_CONSTANT=0, PI_STORAGE_UNIFORM=2, PI_STORAGE_FUNCTION=7, PI_STORAGE_STORAGE_BUFFER=12
};
enum { PI_SAMPLER_POOL=1, PI_SAMPLER_ELEMENT, PI_SAMPLER_LOAD, PI_SAMPLER_COMBINED };
typedef struct {
    uint32_t constant;
    uint16_t op, a, b, c;
    const GfxShaderRes *resource;
    uint8_t resource_kind;
} PoolIndexNode;
typedef struct { uint16_t node, type; } PoolIndexResult;
typedef struct { uint16_t node; uint8_t mode; } PoolSamplerResult;
typedef struct {
    uint16_t node_count, result_count, texture_node_count, sampler_count;
    PoolSamplerResult sampler_nodes[PI_RESULT_LIMIT];
    const GfxShaderRes *domain_source;
    uint32_t domain_maximum, domain_base_word;
    uint16_t domain_address, inactive_guard;
    uint8_t sampler_modes;
    PoolIndexResult results[PI_RESULT_LIMIT];
    PoolIndexNode nodes[];
} ShaderPoolIndices;

typedef struct {
    const uint32_t *insn, *store, *terminator;
    uint32_t type, function, block, store_block, visited, binding, set, stride;
    uint16_t stores, node;
    uint16_t sampler_node;
    uint8_t pool, visiting, escaped, inexact, sampler, sampler_mode, pool_access;
} PoolSpvId;
typedef struct {
    const uint32_t *words;
    uint64_t word_count;
    uint32_t bound, entry, execution_model, first, visits;
    uint32_t *walk;
    PoolSpvId *ids;
    PoolIndexNode nodes[PI_NODE_LIMIT];
    PoolIndexResult results[PI_RESULT_LIMIT];
    uint16_t node_count, result_count, sampler_count;
    PoolSamplerResult sampler_nodes[PI_RESULT_LIMIT];
    const GfxShaderRes *domain_source;
    const PoolSpvId *domain_handle;
    uint32_t domain_maximum, domain_base_word;
    uint16_t domain_address;
    uint8_t domain_inexact, domain_mixed_handles, sampler_modes;
    const GfxShaderRes *resources[GFX_RES_KINDS];
    int counts[GFX_RES_KINDS];
} PoolIndexBuilder;
static uint32_t pi_op(const uint32_t *inst) { return inst[0] & 0xffffu; }
static uint32_t pi_len(const uint32_t *inst) { return inst[0] >> 16; }
static PoolSpvId *pi_id(PoolIndexBuilder *p,uint32_t id) { return id && id<p->bound ? &p->ids[id] : NULL; }
static const uint32_t *pi_insn(PoolIndexBuilder *p,uint32_t id) {
    PoolSpvId *d=pi_id(p,id);
    return d ? d->insn : NULL;
}

static uint32_t pi_result_operand(uint32_t op) {
    switch (op) {
    case PX_TYPE_BOOL: case PX_TYPE_INT: case PX_TYPE_FLOAT: case PX_TYPE_VECTOR: case PX_TYPE_IMAGE: case PX_TYPE_SAMPLER:
    case PX_TYPE_ARRAY:
    case PX_TYPE_RUNTIME_ARRAY: case PX_TYPE_STRUCT: case PX_TYPE_POINTER: case PX_LABEL: case PX_EXT_IMPORT:
        return 1;
    case PX_TRUE: case PX_FALSE: case PX_CONST: case PX_CONST_NULL: case PX_FN: case PX_PARAM: case PX_VAR:
    case PX_LOAD: case PX_CHAIN: case PX_IN_CHAIN: case PX_COPY: case PX_BITCAST: case PX_ADD: case PX_SUB: case PX_MUL:
    case PX_NOT: case PX_SELECT: case PX_EQ: case PX_NE: case PX_SGT: case PX_RSHIFT: case PX_LSHIFT: case PX_OR: case PX_XOR:
    case PX_AND: case PX_EXT: case PX_COMPOSITE_EXTRACT: case PX_SAMPLED_IMAGE:
        return 2;
    default: return 0;
    }
}

static int pi_scalar(PoolIndexBuilder *p,uint32_t type) {
    const uint32_t *t=pi_insn(p,type);
    if (!t) return 0;
    if (pi_op(t)==PX_TYPE_BOOL) return 1;
    if ((pi_op(t)==PX_TYPE_INT || pi_op(t)==PX_TYPE_FLOAT) && pi_len(t)>2 && t[2]==32) return 32;
    return 0;
}

static int pi_visit_block(PoolIndexBuilder *p,uint32_t block,uint32_t stop,uint32_t *pending) {
    PoolSpvId *label=pi_id(p,block);
    if (!label || !label->insn || pi_op(label->insn)!=PX_LABEL || label->function!=p->entry) return 0;
    if (block!=stop && label->visited!=p->visits) {
        label->visited=p->visits;
        p->walk[(*pending)++]=block;
    }
    return 1;
}

// A single store can be outside the entry block. Its value consists only of exact non-invocation inputs,
// but it must execute before every load that supplies a descriptor index. Search for any path that
// reaches the load without crossing the store. Unsupported terminators leave the whole-kind fallback.
static int pi_store_dominates(PoolIndexBuilder *p,const PoolSpvId *ptr,const PoolSpvId *load) {
    if (ptr->stores!=1 || ptr->escaped || ptr->function!=p->entry || load->function!=p->entry ||
        !ptr->store_block || !load->block) return 0;
    if (ptr->store_block==load->block) return ptr->store<load->insn;
    if (ptr->store_block==p->first) return 1;
    if (!p->walk) {
        p->walk=malloc((size_t)p->bound*sizeof *p->walk);
        if (!p->walk) return 0;
    }
    if (!++p->visits) return 0;
    uint32_t pending=0;
    if (!pi_visit_block(p,p->first,ptr->store_block,&pending)) return 0;
    while (pending) {
        uint32_t at=p->walk[--pending];
        if (at==load->block) return 0;
        const uint32_t *end=pi_id(p,at)->terminator;
        if (!end) return 0;
        uint32_t n=pi_len(end);
        switch (pi_op(end)) {
        case 249: // OpBranch
            if (n!=2 || !pi_visit_block(p,end[1],ptr->store_block,&pending)) return 0;
            break;
        case 250: // OpBranchConditional
            if ((n!=4 && n!=6) || !pi_visit_block(p,end[2],ptr->store_block,&pending) ||
                !pi_visit_block(p,end[3],ptr->store_block,&pending)) return 0;
            break;
        case 251: { // OpSwitch with one 32-bit literal for each label
            const PoolSpvId *selector=n>=3 ? pi_id(p,end[1]) : NULL;
            if (!selector || pi_scalar(p,selector->type)!=32 || !(n&1) ||
                !pi_visit_block(p,end[2],ptr->store_block,&pending)) return 0;
            for (uint32_t i=4;i<n;i+=2)
                if (!pi_visit_block(p,end[i],ptr->store_block,&pending)) return 0;
            break;
        }
        case 252: case 253: case 254: case 255: case 4416: // Return, discard, or unreachable
            break;
        default: return 0;
        }
    }
    return 1;
}

static uint16_t pi_node(PoolIndexBuilder *p,PoolIndexNode n) {
    for (uint16_t i=1;i<p->node_count;++i) {
        const PoolIndexNode *t=&p->nodes[i];
        if (t->op==n.op && t->a==n.a && t->b==n.b && t->c==n.c && t->constant==n.constant &&
            t->resource==n.resource && t->resource_kind==n.resource_kind) return i;
    }
    if (p->node_count>=PI_NODE_LIMIT) return 0;
    p->nodes[p->node_count]=n;
    return p->node_count++;
}
static uint16_t pi_literal(PoolIndexBuilder *p,uint32_t value) {
    return pi_node(p,(PoolIndexNode){.op=PI_VALUE,.constant=value});
}

static int pi_struct_offset(PoolIndexBuilder *p,uint32_t type,uint32_t member,uint32_t *offset) {
    // SPIR-V decorations precede definitions. Resolve only the literal Offset on a known struct member.
    for (uint64_t k=5;k<p->word_count;) {
        const uint32_t *i=p->words+k;
        if (pi_op(i)==PX_MEMBER_DECORATE && pi_len(i)==5 && i[1]==type && i[2]==member && i[3]==PI_MEMBER_OFFSET) {
            *offset=i[4];
            return 1;
        }
        k+=pi_len(i);
    }
    return 0;
}

static uint16_t pi_expression(PoolIndexBuilder *,uint32_t id,uint32_t depth);

// Uniform and StorageBuffer access chains share SPIR-V's explicit layouts. Only exact scalar 32-bit loads
// from reflected read-only buffers qualify; array/vector strides are on their real decorated type.
static uint16_t pi_buffer_load(PoolIndexBuilder *p,const uint32_t *chain,uint32_t depth) {
    uint32_t length=pi_len(chain);
    if (length<5 || length>4+PI_CHAIN_LIMIT) return 0;
    PoolSpvId *base=pi_id(p,chain[3]);
    if (!base || !base->insn || pi_op(base->insn)!=PX_VAR || pi_len(base->insn)<4 || base->set ||
        !base->binding || (base->insn[3]!=PI_STORAGE_UNIFORM && base->insn[3]!=PI_STORAGE_STORAGE_BUFFER)) return 0;
    int kind=-1;
    const GfxShaderRes *resource=NULL;
    for (int k=GFX_RES_UBO;k<=GFX_RES_SSBO;++k)
        for (int i=0;i<p->counts[k];++i) if ((uint32_t)p->resources[k][i].binding+1==base->binding) {
            if (resource) return 0;
            kind=k;
            resource=&p->resources[k][i];
        }
    if (!resource || (kind==GFX_RES_SSBO && ((resource->flags&1) || resource->cbuf_slot==0x40))) return 0;
    const uint32_t *pointer=pi_insn(p,base->type);
    if (!pointer || pi_op(pointer)!=PX_TYPE_POINTER || pi_len(pointer)!=4) return 0;
    uint32_t type=pointer[3];
    uint16_t offset=pi_literal(p,0);
    for (uint32_t x=4;offset && x<length;++x) {
        const uint32_t *t=pi_insn(p,type);
        if (!t) return 0;
        uint16_t add=0;
        if (pi_op(t)==PX_TYPE_STRUCT) {
            const uint32_t *member=pi_insn(p,chain[x]);
            uint32_t at,off;
            if (!member || pi_op(member)!=PX_CONST || pi_len(member)!=4) return 0;
            at=member[3];
            if (at>=pi_len(t)-2 || !pi_struct_offset(p,type,at,&off)) return 0;
            type=t[2+at];
            add=pi_literal(p,off);
        } else {
            uint32_t step=pi_id(p,type)->stride, count=0;
            if (pi_op(t)==PX_TYPE_VECTOR && pi_len(t)==4 && pi_scalar(p,t[2])==32) {
                step=4;
                count=t[3];
            } else if (pi_op(t)==PX_TYPE_ARRAY && pi_len(t)==4) {
                const uint32_t *n=pi_insn(p,t[3]);
                if (!n || pi_op(n)!=PX_CONST || pi_len(n)!=4 || !n[3]) return 0;
                count=n[3];
            } else if (pi_op(t)!=PX_TYPE_RUNTIME_ARRAY) return 0;
            if (!step) return 0;
            type=t[2];
            uint16_t index=pi_expression(p,chain[x],depth+1);
            if (!index) return 0;
            // Checked address arithmetic is separate from modulo arithmetic in the shader value slice.
            add=pi_node(p,(PoolIndexNode){.op=PX_IN_CHAIN,.a=index,.constant=step,.b=0});
            if (add && count)
                add=pi_node(p,(PoolIndexNode){.op=PX_COMPOSITE_EXTRACT,.a=add,.constant=count,.b=index});
        }
        if (!add) return 0;
        offset=pi_node(p,(PoolIndexNode){.op=PX_CHAIN,.a=offset,.b=add});
    }
    if (!offset || pi_scalar(p,type)!=32) return 0;
    return pi_node(p,(PoolIndexNode){.op=PI_U32,.a=offset,.resource=resource,.resource_kind=(uint8_t)kind});
}

static uint16_t pi_expression(PoolIndexBuilder *p,uint32_t id,uint32_t depth) {
    PoolSpvId *d=pi_id(p,id);
    if (!d || !d->insn || !pi_scalar(p,d->type) || depth>=PI_RECURSION_LIMIT || d->visiting) return 0;
    if (d->node) return d->node;
    d->visiting=1;
    const uint32_t *i=d->insn;
    uint32_t op=pi_op(i), length=pi_len(i);
    uint16_t out=0,a,b,c;
    switch (op) {
    case PX_TRUE: case PX_FALSE: case PX_CONST_NULL:
        out=pi_literal(p,op==PX_TRUE);
        break;
    case PX_CONST:
        if (length==4) out=pi_literal(p,i[3]);
        break;
    case PX_LOAD: {
        if (length!=4) break;
        PoolSpvId *ptr=pi_id(p,i[3]);
        if (!ptr || !ptr->insn) break;
        uint32_t storage=pi_op(ptr->insn);
        if (storage==PX_VAR && pi_len(ptr->insn)>=4 && ptr->insn[3]==PI_STORAGE_FUNCTION) {
            if (pi_store_dominates(p,ptr,d))
                out=pi_expression(p,ptr->store[2],depth+1);
        } else if (storage==PX_CHAIN || storage==PX_IN_CHAIN) {
            out=pi_buffer_load(p,ptr->insn,depth+1);
        }
        break;
    }
    case PX_COPY: case PX_BITCAST:
        if (length==4 && pi_id(p,i[3]) && pi_scalar(p,pi_id(p,i[3])->type)==pi_scalar(p,d->type))
            out=pi_expression(p,i[3],depth+1);
        break;
    case PX_NOT:
        if (length==4 && (a=pi_expression(p,i[3],depth+1)))
            out=pi_node(p,(PoolIndexNode){.op=PX_NOT,.a=a});
        break;
    case PX_SELECT:
        if (length==6 && (a=pi_expression(p,i[3],depth+1)) && (b=pi_expression(p,i[4],depth+1)) &&
            (c=pi_expression(p,i[5],depth+1))) out=pi_node(p,(PoolIndexNode){.op=PX_SELECT,.a=a,.b=b,.c=c});
        break;
    case PX_EXT: {
        if (length!=7 || i[4]!=38) break;   // GLSL.std.450 UMin
        const uint32_t *ext=pi_insn(p,i[3]);
        static const char name[]="GLSL.std.450";
        if (ext && pi_op(ext)==PX_EXT_IMPORT && (pi_len(ext)-2)*4>=sizeof name && !memcmp(ext+2,name,sizeof name) &&
            (a=pi_expression(p,i[5],depth+1)) && (b=pi_expression(p,i[6],depth+1)))
            out=pi_node(p,(PoolIndexNode){.op=PI_UMIN,.a=a,.b=b});
        break;
    }
    case PX_ADD: case PX_SUB: case PX_MUL: case PX_EQ: case PX_NE: case PX_RSHIFT: case PX_LSHIFT:
    case PX_OR: case PX_XOR: case PX_AND:
        if (length==5 && (a=pi_expression(p,i[3],depth+1)) && (b=pi_expression(p,i[4],depth+1)))
            out=pi_node(p,(PoolIndexNode){.op=(uint16_t)op,.a=a,.b=b});
        break;
    default: break;
    }
    d->visiting=0;
    d->node=out;
    return out;
}

static int pi_local_reference(uint32_t op,uint32_t operand,uint32_t length) {
    // These are literals, annotations or declaration result IDs, not uses of a descriptor or local pointer.
    if (operand==pi_result_operand(op)) return 0;
    switch (op) {
    case PX_ENTRY: case PX_DECORATE: case PX_MEMBER_DECORATE:
    case 5: case 6: case 7: case 8: case 10: case 16: case 17:
    case 3: case 4: case 14:       // handled before the scan
        return 0;
    case PX_EXT_IMPORT: case PX_CONST: case PX_CONST_NULL: return 0;
    case 21: case 22: case 31: return 0;        // Integer, float and opaque type literals.
    case 23: case 24: case 25: return operand == 2; // Vector, matrix and image component type.
    case 32: return operand == 3;             // OpTypePointer's storage class is a literal.
    case 39: return operand == 1;             // OpTypeForwardPointer's storage class is a literal.
    case PX_VAR: return operand>=4;
    case PX_LOAD: case PX_ARRAY_LENGTH: return operand==3;
    case PX_STORE: return operand<=2;
    case PX_EXT: return operand>=5;
    case PX_COMPOSITE_EXTRACT: return operand==3;
    case 79: return operand==3 || operand==4;     // OpVectorShuffle
    case 80: return operand>=3;                   // OpCompositeConstruct
    case 82: return operand==3 || operand==4;      // OpCompositeInsert
    case 87: case 88: case 89: case 90: case 91: case 92: case 93: case 94: case 95: case 96: case 97:
        return operand!=((op==89 || op==90 || op==93 || op==94 || op==97)?6u:5u);
    case 246: return operand<3;                    // OpLoopMerge
    case 247: return operand==1;                   // OpSelectionMerge
    case 250: return operand<=3;                   // OpBranchConditional
    case 251: return operand<3 || (operand&1)==0;   // OpSwitch 32-bit generated labels. Split-wide falls back.
    default: return operand<length;
    }
}

static const uint32_t *pi_domain_bits(PoolIndexBuilder *p, uint32_t id) {
    for (unsigned depth = 0; depth < PI_RECURSION_LIMIT; ++depth) {
        const PoolSpvId *value = pi_id(p, id);
        if (!value || !value->insn || value->inexact || pi_scalar(p, value->type) != 32) return NULL;
        const uint32_t *op = value->insn;
        if (pi_op(op) != PX_COPY && pi_op(op) != PX_BITCAST) return op;
        if (pi_len(op) != 4) return NULL;
        id = op[3];
    }
    return NULL;
}

static int pi_domain_constant(PoolIndexBuilder *p, uint32_t id, uint32_t value) {
    const uint32_t *op = pi_domain_bits(p, id);
    return op && ((pi_op(op) == PX_CONST && pi_len(op) == 4 && op[3] == value) ||
        (pi_op(op) == PX_CONST_NULL && pi_len(op) == 3 && value == 0));
}

// The translated pool index saturates the low 20 texture bits or high 12 sampler bits.
// A private scalar initialized to zero can keep zero or receive its sole store from the table.
static const PoolSpvId *pi_domain_handle(PoolIndexBuilder *p, uint32_t index, uint32_t maximum, int sampler) {
    const uint32_t *clamp = pi_domain_bits(p, index);
    if (!clamp || pi_op(clamp) != PX_EXT || pi_len(clamp) != 7 || clamp[4] != 38 ||
        !pi_domain_constant(p, clamp[6], maximum)) return NULL;
    const uint32_t *extended = pi_insn(p, clamp[3]);
    static const char glsl[] = "GLSL.std.450";
    if (!extended || pi_op(extended) != PX_EXT_IMPORT || pi_len(extended) < 2 ||
        (pi_len(extended) - 2) * 4 < sizeof glsl ||
        memcmp(extended + 2, glsl, sizeof glsl)) return NULL;
    const uint32_t *extract = pi_domain_bits(p, clamp[5]);
    if (!extract || pi_op(extract) != (sampler ? PX_RSHIFT : PX_AND) || pi_len(extract) != 5 ||
        !pi_domain_constant(p, extract[4], sampler ? 20 : UINT32_C(0xfffff))) return NULL;
    const uint32_t *load = pi_domain_bits(p, extract[3]);
    if (!load || pi_op(load) != PX_LOAD || pi_len(load) != 4) return NULL;
    const PoolSpvId *local = pi_id(p, load[3]);
    if (!local || !local->insn || pi_op(local->insn) != PX_VAR || pi_len(local->insn) != 5 ||
        local->insn[3] != PI_STORAGE_FUNCTION || local->function != p->entry || local->escaped || local->inexact ||
        local->stores != 1 ||
        !pi_domain_constant(p, local->insn[4], 0)) return NULL;
    return local;
}

static const GfxShaderRes *pi_domain_storage_source(PoolIndexBuilder *p, const uint32_t *read,
    uint32_t *address, uint32_t *base_word) {
    if (!read || pi_op(read) != PX_LOAD || pi_len(read) != 4) return NULL;
    const uint32_t *chain = pi_insn(p, read[3]);
    if (!chain || pi_op(chain) != PX_CHAIN || pi_len(chain) != 6) return NULL;
    const PoolSpvId *base = pi_id(p, chain[3]);
    if (!base || !base->insn || pi_op(base->insn) != PX_VAR || pi_len(base->insn) != 4 || !base->binding || base->set ||
        (base->insn[3] != PI_STORAGE_UNIFORM && base->insn[3] != PI_STORAGE_STORAGE_BUFFER)) return NULL;
    const uint32_t *pointer = pi_insn(p, base->type);
    const uint32_t *element = pi_insn(p, chain[1]);
    if (!pointer || pi_op(pointer) != PX_TYPE_POINTER || pi_len(pointer) != 4 || pointer[2] != base->insn[3] ||
        !element || pi_op(element) != PX_TYPE_POINTER || pi_len(element) != 4 || element[2] != pointer[2] ||
        element[3] != read[1]) return NULL;
    const uint32_t *structure = pi_insn(p, pointer[3]);
    const uint32_t *member = pi_insn(p, chain[4]);
    uint32_t offset;
    if (!structure || pi_op(structure) != PX_TYPE_STRUCT || !member || pi_op(member) != PX_CONST || pi_len(member) != 4 ||
        member[3] >= pi_len(structure) - 2 || !pi_struct_offset(p, pointer[3], member[3], &offset) || (offset & 3)) return NULL;
    const PoolSpvId *array = pi_id(p, structure[2 + member[3]]);
    if (!array || !array->insn || pi_op(array->insn) != PX_TYPE_RUNTIME_ARRAY || pi_len(array->insn) != 3 ||
        array->stride != 4 || array->insn[2] != read[1]) return NULL;
    const GfxShaderRes *source = NULL;
    for (int j = 0; j < p->counts[GFX_RES_SSBO]; ++j) {
        const GfxShaderRes *resource = &p->resources[GFX_RES_SSBO][j];
        if ((uint32_t)resource->binding + 1 != base->binding) continue;
        if (source || resource->set || resource->ptr || resource->cbuf_slot != 0 || resource->count != 1 || (resource->flags & 1))
            return NULL;
        source = resource;
    }
    *address = chain[5];
    *base_word = offset / 4;
    return source;
}


static uint16_t pi_domain_known_offset(PoolIndexBuilder *p, uint32_t id, unsigned depth) {
    PoolSpvId *value = pi_id(p, id);
    if (!value || !value->insn || !pi_scalar(p, value->type) ||
        depth >= PI_RECURSION_LIMIT || value->visiting) return 0;
    if (value->inexact) {
        p->domain_inexact = 1;
        return 0;
    }
    if (value->node) return value->node;
    value->visiting = 1;
    const uint32_t *i = value->insn;
    const uint32_t length = pi_len(i), op = pi_op(i);
    uint16_t node = 0;
    switch (op) {
    case PX_CONST:
        if (length == 4 && pi_scalar(p, value->type) == 32) node = pi_literal(p, i[3]);
        break;
    case PX_CONST_NULL:
        if (length == 3 && pi_scalar(p, value->type) == 32) node = pi_literal(p, 0);
        break;
    case PX_COPY: case PX_BITCAST: {
        const PoolSpvId *source = length == 4 ? pi_id(p, i[3]) : NULL;
        if (source && pi_scalar(p, source->type) == 32 && pi_scalar(p, value->type) == 32)
            node = pi_domain_known_offset(p, i[3], depth + 1);
        break;
    }
    case PX_LOAD: {
        const PoolSpvId *ptr = length == 4 ? pi_id(p, i[3]) : NULL;
        if (!ptr || !ptr->insn || pi_scalar(p, value->type) != 32) break;
        if (pi_op(ptr->insn) == PX_VAR && pi_len(ptr->insn) >= 4 && ptr->insn[3] == PI_STORAGE_FUNCTION) {
            if (pi_store_dominates(p, ptr, value)) node = pi_domain_known_offset(p, ptr->store[2], depth + 1);
            break;
        }
        const uint32_t *chain = ptr->insn;
        if ((pi_op(chain) != PX_CHAIN && pi_op(chain) != PX_IN_CHAIN) || pi_len(chain) < 5) break;
        const PoolSpvId *base = pi_id(p, chain[3]);
        if (!base || !base->binding || base->set) break;
        // Resolve only statically addressed original C0 words. All other encountered buffers stay unknown.
        int exact = 1;
        for (uint32_t axis = 4; axis < pi_len(chain); ++axis) {
            const uint32_t *constant = pi_insn(p, chain[axis]);
            if (!constant || pi_op(constant) != PX_CONST || pi_len(constant) != 4) exact = 0;
        }
        if (!exact) break;
        for (int j = 0; j < p->counts[GFX_RES_UBO]; ++j) {
            const GfxShaderRes *resource = &p->resources[GFX_RES_UBO][j];
            if (!resource->set && !resource->slot && resource->count == 1 &&
                (uint32_t)resource->binding + 1 == base->binding) {
                node = pi_buffer_load(p, chain, depth + 1);
                break;
            }
        }
        break;
    }
    case PX_ADD: case PX_SUB: case PX_MUL: case PX_AND: case PX_RSHIFT: case PX_LSHIFT:
        if (length == 5 && pi_scalar(p, value->type) == 32) {
            const uint16_t a = pi_domain_known_offset(p, i[3], depth + 1);
            const uint16_t b = pi_domain_known_offset(p, i[4], depth + 1);
            if (a || b) node = pi_node(p, (PoolIndexNode){.op = (uint16_t)op, .a = a, .b = b});
        }
        break;
    default: break;
    }
    value->node = node;
    value->visiting = 0;
    return node;
}

static int pi_domain_store_is_local(PoolIndexBuilder *p, const uint32_t *instruction) {
    const PoolSpvId *id = pi_id(p, instruction[1]);
    const uint32_t *ptr = id && id->type ? pi_insn(p, id->type) : NULL;
    return ptr && pi_op(ptr) == PX_TYPE_POINTER && pi_len(ptr) == 4 &&
        (ptr[2] == 3 || ptr[2] == 6 || ptr[2] == PI_STORAGE_FUNCTION);
}

static int pi_compute_domain_pointer_use(PoolIndexBuilder *p, const PoolSpvId *id, uint32_t op, uint32_t operand) {
    const uint32_t *ptr = pi_insn(p, id->type);
    if (!ptr || pi_op(ptr) != PX_TYPE_POINTER || pi_len(ptr) != 4 ||
        (ptr[2] != PI_STORAGE_UNIFORM && ptr[2] != PI_STORAGE_STORAGE_BUFFER)) return 1;
    return (op == PX_STORE && operand == 1) ||
        ((op == PX_LOAD || op == PX_CHAIN || op == PX_IN_CHAIN || op == PX_ARRAY_LENGTH) && operand == 3);
}

static int pi_domain_store_allowed(PoolIndexBuilder *p, const uint32_t *instruction) {
    if (pi_domain_store_is_local(p, instruction)) return 1;
    // Only the selected compute binder proves retained input/output interval disjointness.
    // Accept an ordinary store through a direct access chain rooted at a fixed reflected output.
    if (p->execution_model != 5 || pi_len(instruction) != 3) return 0;
    const uint32_t *chain = pi_insn(p, instruction[1]);
    if (!chain || (pi_op(chain) != PX_CHAIN && pi_op(chain) != PX_IN_CHAIN) || pi_len(chain) < 5) return 0;
    const PoolSpvId *root = pi_id(p, chain[3]);
    if (!root || !root->insn || pi_op(root->insn) != PX_VAR || pi_len(root->insn) != 4 ||
        root->set || !root->binding) return 0;
    const uint32_t storage = root->insn[3];
    const uint32_t *ptr = pi_insn(p, root->type), *element = pi_insn(p, chain[1]);
    if ((storage != PI_STORAGE_STORAGE_BUFFER && storage != PI_STORAGE_UNIFORM) ||
        !ptr || pi_op(ptr) != PX_TYPE_POINTER || pi_len(ptr) != 4 || ptr[2] != storage ||
        !element || pi_op(element) != PX_TYPE_POINTER || pi_len(element) != 4 || element[2] != storage) return 0;
    const uint32_t *structure = pi_insn(p, ptr[3]);
    if (!structure || pi_op(structure) != PX_TYPE_STRUCT || pi_len(structure) < 3) return 0;
    const GfxShaderRes *output = NULL;
    for (int i = 0; i < p->counts[GFX_RES_SSBO]; ++i) {
        const GfxShaderRes *resource = &p->resources[GFX_RES_SSBO][i];
        if (resource->set || (uint32_t)resource->binding + 1 != root->binding) continue;
        if (output || resource->ptr || resource->cbuf_slot || resource->count != 1 || !(resource->flags & 1))
            return 0;
        output = resource;
    }
    return output != NULL;
}

static int pi_domain_pool_type(PoolIndexBuilder *p, const PoolSpvId *pool, const GfxShaderRes *resource) {
    const uint32_t *ptr = pi_insn(p, pool->type);
    if (!ptr || pi_op(ptr) != PX_TYPE_POINTER || pi_len(ptr) != 4 || ptr[2] != PI_STORAGE_UNIFORM_CONSTANT) return 0;
    const uint32_t *array = pi_insn(p, ptr[3]);
    if (!array || pi_op(array) != PX_TYPE_ARRAY || pi_len(array) != 4 ||
        !pi_domain_constant(p, array[3], resource->count)) return 0;
    const uint32_t *image = pi_insn(p, array[2]);
    if (!image || pi_op(image) != PX_TYPE_IMAGE || pi_len(image) != 9 || image[7] != 1 || image[8] || image[6]) return 0;
    const uint32_t *sampled = pi_insn(p, image[2]);
    if (!sampled || pi_op(sampled) != PX_TYPE_FLOAT || pi_len(sampled) != 3 || sampled[2] != 32) return 0;
    const uint16_t type = resource->type;
    const uint16_t dimensions[] = {1, 3, 4, 5};
    return image[3] < sizeof dimensions / sizeof dimensions[0] && (type & 0xff) == dimensions[image[3]] &&
        image[4] == !!(type & 0x400) && image[5] == !!(type & 0x100) && !(type & 0x200);
}

static int pi_sampler_variable(PoolIndexBuilder *p, PoolSpvId *id, const GfxShaderRes *resource) {
    if (resource->count != 4096 || resource->set || id->set || pi_op(id->insn) != PX_VAR ||
        pi_len(id->insn) != 4 || id->insn[3] != PI_STORAGE_UNIFORM_CONSTANT) return 0;
    const uint32_t *ptr = pi_insn(p, id->type);
    if (!ptr || pi_op(ptr) != PX_TYPE_POINTER || pi_len(ptr) != 4 || ptr[2] != PI_STORAGE_UNIFORM_CONSTANT) return 0;
    const uint32_t *array = pi_insn(p, ptr[3]);
    if (!array || pi_op(array) != PX_TYPE_ARRAY || pi_len(array) != 4 ||
        !pi_domain_constant(p, array[3], resource->count)) return 0;
    const uint32_t *sampler = pi_insn(p, array[2]);
    return sampler && pi_op(sampler) == PX_TYPE_SAMPLER && pi_len(sampler) == 2;
}

static unsigned pi_sampled_pool_image_mode(PoolIndexBuilder *p, uint32_t id) {
    const uint32_t *load = pi_insn(p, id);
    if (!load || pi_op(load) != PX_LOAD || pi_len(load) != 4) return 0;
    const uint32_t *image = pi_insn(p, load[1]);
    const uint32_t *chain = pi_insn(p, load[3]);
    if (!image || pi_op(image) != PX_TYPE_IMAGE || pi_len(image) != 9 || image[4] > 1 ||
        !chain || (pi_op(chain) != PX_CHAIN && pi_op(chain) != PX_IN_CHAIN) || pi_len(chain) != 5) return 0;
    const PoolSpvId *base = pi_id(p, chain[3]);
    if (!base || !base->pool) return 0;
    return image[4] ? GFX_POOL_SAMPLER_COMPARE : GFX_POOL_SAMPLER_ORDINARY;
}

static int pi_depth_comparison(uint32_t op) {
    // ImageSampleDref{Implicit,Explicit}Lod, projected equivalents, and ImageDrefGather.
    return op == 89 || op == 90 || op == 93 || op == 94 || op == 97;
}

static int pi_ordinary_sampling(uint32_t op) {
    return op == 87 || op == 88 || op == 91 || op == 92 || op == 96;
}

// Every sampler-pool reference must have a typed sampling use. A storage-table read also
// needs the same captured handle. Scalar sampler indices have their own exact input expressions.
static int pi_sampler_domains(PoolIndexBuilder *p, int domain) {
    if (domain && (!p->domain_handle || p->domain_mixed_handles)) return 0;
    const GfxShaderRes *resource = NULL;
    for (int j = 0; j < p->counts[GFX_RES_TEX]; ++j) {
        const GfxShaderRes *candidate = &p->resources[GFX_RES_TEX][j];
        if (candidate->pool != 2) continue;
        if (resource) return 0;
        resource = candidate;
    }
    if (!resource) return 0;
    int pools = 0;
    for (uint32_t j = 1; j < p->bound; ++j) {
        PoolSpvId *id = &p->ids[j];
        if (!id->insn || id->binding != (uint32_t)resource->binding + 1) continue;
        if (pools || !pi_sampler_variable(p, id, resource)) return 0;
        id->sampler = PI_SAMPLER_POOL;
        ++pools;
    }
    if (!pools) return 0;
    for (uint64_t at = 5; at < p->word_count;) {
        const uint32_t *i = p->words + at;
        const uint32_t op = pi_op(i), len = pi_len(i);
        if ((op == PX_CHAIN || op == PX_IN_CHAIN) && len == 5) {
            const PoolSpvId *base = pi_id(p, i[3]);
            if (base && base->sampler == PI_SAMPLER_POOL) {
                PoolSpvId *out = pi_id(p, i[2]);
                if (domain) {
                    if (pi_domain_handle(p, i[4], resource->count - 1, 1) != p->domain_handle) return 0;
                } else {
                    out->sampler_node = pi_expression(p, i[4], 0);
                    if (!out->sampler_node) return 0;
                }
                pi_id(p, i[2])->sampler = PI_SAMPLER_ELEMENT;
            }
        } else if (op == PX_LOAD && len == 4) {
            const PoolSpvId *base = pi_id(p, i[3]);
            if (base && base->sampler == PI_SAMPLER_ELEMENT) {
                PoolSpvId *out = pi_id(p, i[2]);
                out->sampler_node = base->sampler_node;
                out->sampler = PI_SAMPLER_LOAD;
            }
        } else if (op == PX_SAMPLED_IMAGE && len == 5) {
            const PoolSpvId *sampler = pi_id(p, i[4]);
            if (sampler && sampler->sampler == PI_SAMPLER_LOAD) {
                const unsigned mode = pi_sampled_pool_image_mode(p, i[3]);
                if (!mode) return 0;
                PoolSpvId *out = pi_id(p, i[2]);
                out->sampler = PI_SAMPLER_COMBINED;
                out->sampler_mode = mode;
                out->sampler_node = sampler->sampler_node;
            }
        }
        at += len;
    }
    for (uint64_t at = 5; at < p->word_count;) {
        const uint32_t *i = p->words + at;
        const uint32_t op = pi_op(i), len = pi_len(i);
        for (uint32_t x = 1; x < len; ++x) {
            const PoolSpvId *id = pi_id(p, i[x]);
            if (!id || !id->sampler || !pi_local_reference(op, x, len)) continue;
            switch (id->sampler) {
            case PI_SAMPLER_POOL:
                if ((op != PX_CHAIN && op != PX_IN_CHAIN) || len != 5 || x != 3) return 0;
                break;
            case PI_SAMPLER_ELEMENT:
                if (op != PX_LOAD || len != 4 || x != 3) return 0;
                break;
            case PI_SAMPLER_LOAD:
                if (op != PX_SAMPLED_IMAGE || len != 5 || x != 4) return 0;
                break;
            case PI_SAMPLER_COMBINED: {
                const unsigned mode = id->sampler_mode;
                if (x != 3 || !(mode == GFX_POOL_SAMPLER_COMPARE ? pi_depth_comparison(op) : pi_ordinary_sampling(op))) return 0;
                p->sampler_modes |= mode;
                if (domain) break;
                uint16_t seen = 0;
                while (seen < p->sampler_count && p->sampler_nodes[seen].node != id->sampler_node) ++seen;
                if (seen == p->sampler_count) {
                    if (seen == PI_RESULT_LIMIT) return 0;
                    p->sampler_nodes[p->sampler_count++] = (PoolSamplerResult){.node=id->sampler_node};
                }
                p->sampler_nodes[seen].mode |= mode;
                break;
            }
            }
        }
        at += len;
    }
    return p->sampler_modes != 0;
}

#include "shader_pool_guard.h"

static ShaderPoolIndices *shader_pool_indices_build_mode(const uint32_t *words,uint64_t count,
    const GfxShaderRes *const resources[GFX_RES_KINDS],const int counts[GFX_RES_KINDS], int domain) {
    if (!words || count<5 || words[0]!=0x07230203 || words[3]>262144u) return NULL;
    int pooled=0;
    for (int i=0;i<counts[GFX_RES_TEX];++i) pooled+=resources[GFX_RES_TEX][i].pool==1;
    if (!pooled) return NULL;
    PoolIndexBuilder p={.words=words,.word_count=count,.bound=words[3],.node_count=1};
    memcpy(p.resources,resources,sizeof p.resources);
    memcpy(p.counts,counts,sizeof p.counts);
    p.ids=calloc(p.bound,sizeof *p.ids);
    if (!p.ids) return NULL;
    ShaderPoolIndices *plan=NULL;
    uint32_t function=0,block=0;
    for (uint64_t k=5;k<count;) {
        const uint32_t *i=words+k;
        uint32_t op=pi_op(i),len=pi_len(i),rd=pi_result_operand(op);
        if (!len || k+len>count || (rd && rd>=len)) goto done;
        if (op==PX_ENTRY) {
            if (p.entry || len<4) goto done;
            p.execution_model=i[1];
            p.entry=i[2];
        }
        if (op==PX_FN) function=i[2],block=0;
        if (op==PX_LABEL) {
            block=i[1];
            if (function==p.entry && !p.first) p.first=block;
        }
        if (op==PX_END) function=block=0;
        if (rd) {
            PoolSpvId *id=pi_id(&p,i[rd]);
            if (!id || id->insn) goto done;
            id->insn=i;
            id->function=function;
            id->block=block;
            id->type=rd==2?i[1]:0;
        }
        if (op == PX_GROUP_DECORATE || op == PX_GROUP_MEMBER_DECORATE) p.domain_inexact = 1;
        if (op == PX_DECORATE && len == 3 && (i[2] == PI_RELAXED_PRECISION ||
            i[2] == PI_NO_SIGNED_WRAP || i[2] == PI_NO_UNSIGNED_WRAP)) {
            PoolSpvId *id = pi_id(&p, i[1]);
            if (!id) goto done;
            id->inexact = 1;
        }
        if (op==PX_DECORATE && len==4) {
            PoolSpvId *id=pi_id(&p,i[1]);
            if (!id) goto done;
            if (i[2]==PI_DESCRIPTOR_BINDING && i[3]<UINT16_MAX) id->binding=i[3]+1;
            if (i[2]==PI_DESCRIPTOR_SET) id->set=i[3];
            if (i[2]==PI_ARRAY_STRIDE) id->stride=i[3];
        }
        if (op==PX_STORE && len>=3) {
            PoolSpvId *ptr=pi_id(&p,i[1]);
            if (!ptr) goto done;
            if (ptr->stores<UINT16_MAX) ++ptr->stores;
            ptr->store=i;
            ptr->store_block=block;
            if (function!=p.entry || !block || len!=3) ptr->escaped=1;
        }
        if ((op>=249 && op<=255) || op==4416) {
            PoolSpvId *label=pi_id(&p,block);
            if (!label || label->terminator) goto done;
            label->terminator=i;
        }
        k+=len;
    }
    if (!p.entry || !p.first) goto done;
    for (uint32_t id=1;id<p.bound;++id) {
        PoolSpvId *v=&p.ids[id];
        if (!v->insn || !v->binding || v->set || pi_op(v->insn)!=PX_VAR || pi_len(v->insn)<4) continue;
        for (int j=0;j<counts[GFX_RES_TEX];++j) if (resources[GFX_RES_TEX][j].pool==1 &&
            (uint32_t)resources[GFX_RES_TEX][j].binding+1==v->binding) {
            if (v->insn[3]!=PI_STORAGE_UNIFORM_CONSTANT) goto done;
            v->pool=1;
            --pooled;
        }
    }
    if (pooled) goto done;
    int in_function = 0;
    for (uint64_t k=5;k<count;) {
        const uint32_t *i=words+k;
        uint32_t op=pi_op(i),len=pi_len(i);
        if (op == PX_FN) in_function = 1;
        if (op == PX_END) in_function = 0;
        if (domain && ((op == PX_STORE && !pi_domain_store_allowed(&p, i)) || op == 63 || op == 64 || op == 99 ||
            (op >= 227 && op <= 242) || op == 318 || op == 319 || op == 5614 || op == 5615 || op == 6035)) goto done;
        for (uint32_t x=1;x<len;++x) {
            PoolSpvId *id=pi_id(&p,i[x]);
            if (!id || !id->insn || !pi_local_reference(op,x,len)) continue;
            if (domain && p.execution_model == 5 && (in_function || op == PX_VAR) &&
                !pi_compute_domain_pointer_use(&p, id, op, x)) goto done;
            if (pi_op(id->insn)!=PX_VAR) continue;
            if (id->pool) {
                if ((op!=PX_CHAIN && op!=PX_IN_CHAIN) || x!=3 || len!=5) goto done;
            } else if (pi_len(id->insn)>=4 && id->insn[3]==PI_STORAGE_FUNCTION &&
                !((op==PX_LOAD && x==3) || (op==PX_STORE && x==1))) id->escaped=1;
        }
        k+=len;
    }
    for (uint64_t k=5;k<count;) {
        const uint32_t *i=words+k;
        uint32_t op=pi_op(i),len=pi_len(i);
        PoolSpvId *base=(op==PX_CHAIN||op==PX_IN_CHAIN)&&len==5?pi_id(&p,i[3]):NULL;
        if (base && base->pool) {
            const GfxShaderRes *pool = NULL;
            for (int j=0;j<p.counts[GFX_RES_TEX];++j)
                if (resources[GFX_RES_TEX][j].pool==1 && (uint32_t)resources[GFX_RES_TEX][j].binding+1==base->binding)
                    pool = &resources[GFX_RES_TEX][j];
            if (!pool) goto done;
            const uint16_t type = pool->type;
            uint16_t index = 0;
            if (domain) {
                if (pool->count < 2 || !pi_domain_pool_type(&p, base, pool)) goto done;
                const uint32_t maximum = pool->count - 1;
                uint32_t address, base_word;
                const PoolSpvId *handle = pi_domain_handle(&p, i[4], maximum, 0);
                const uint32_t *read = handle ? pi_domain_bits(&p, handle->store[2]) : NULL;
                const GfxShaderRes *source = pi_domain_storage_source(&p, read, &address, &base_word);
                if (!source || (p.domain_source && (p.domain_source != source || p.domain_maximum != maximum))) goto done;
                const uint16_t narrowed = pi_domain_known_offset(&p, address, 0);
                if (!p.domain_source) {
                    p.domain_address = narrowed;
                    p.domain_base_word = base_word;
                    p.domain_handle = handle;
                } else {
                    if (p.domain_address != narrowed || p.domain_base_word != base_word) p.domain_address = 0;
                    if (p.domain_handle != handle) p.domain_mixed_handles = 1;
                }
                p.domain_source = source;
                p.domain_maximum = maximum;
            } else {
                index=pi_expression(&p,i[4],0);
                if (!index) goto done;
            }
            uint16_t found=0;
            while (found<p.result_count && (p.results[found].node!=index || p.results[found].type!=type)) ++found;
            if (found==p.result_count) {
                if (p.result_count==PI_RESULT_LIMIT) goto done;
                p.results[p.result_count++]=(PoolIndexResult){index,type};
            }
        }
        k+=len;
    }
    if (!p.result_count) goto done;
    const uint16_t texture_nodes = p.node_count;
    if (p.domain_inexact || !pi_sampler_domains(&p, domain)) {
        p.node_count = texture_nodes;
        p.sampler_count = 0;
        p.sampler_modes = 0;
    }
    const uint16_t inactive_guard = domain ? pi_guard_build(&p) : 0;
    plan=malloc(sizeof *plan+(size_t)p.node_count*sizeof *plan->nodes);
    if (plan) {
        plan->node_count=p.node_count;
        plan->result_count=p.result_count;
        plan->domain_source=p.domain_source;
        plan->domain_maximum=p.domain_maximum;
        plan->domain_address=p.domain_inexact ? 0 : p.domain_address;
        plan->domain_base_word=p.domain_base_word;
        plan->inactive_guard=inactive_guard;
        plan->sampler_modes=p.sampler_modes;
        plan->texture_node_count=texture_nodes;
        plan->sampler_count=p.sampler_count;
        memcpy(plan->sampler_nodes,p.sampler_nodes,p.sampler_count*sizeof *p.sampler_nodes);
        memcpy(plan->results,p.results,p.result_count*sizeof *p.results);
        memcpy(plan->nodes,p.nodes,p.node_count*sizeof *p.nodes);
    }
done:
    free(p.walk);
    free(p.ids);
    return plan;
}

static ShaderPoolIndices *shader_pool_indices_build(const uint32_t *words,uint64_t count,
    const GfxShaderRes *const resources[GFX_RES_KINDS],const int counts[GFX_RES_KINDS]) {
    ShaderPoolIndices *scalar = shader_pool_indices_build_mode(words, count, resources, counts, 0);
    return scalar ? scalar : shader_pool_indices_build_mode(words, count, resources, counts, 1);
}


typedef struct { uint32_t value; unsigned bits; } PoolKnownLow;

static unsigned pi_trailing_zeros(uint32_t value) { return value ? (unsigned)__builtin_ctz(value) : 32; }
static uint32_t pi_mask_low(unsigned bits) { return bits ? UINT32_MAX >> (32 - bits) : 0; }
static unsigned pi_min_unsigned(unsigned a, unsigned b) { return a < b ? a : b; }

static PoolKnownLow pi_domain_low_bits(const ShaderPoolIndices *plan, GfxShaderPoolReadWord read, void *context) {
    PoolKnownLow known[PI_NODE_LIMIT] = {{0}};
    if (!plan->domain_address) return known[0];
    for (uint16_t i = 1; i <= plan->domain_address; ++i) {
        const PoolIndexNode *node = &plan->nodes[i];
        const PoolKnownLow a = known[node->a], b = known[node->b];
        PoolKnownLow result = {0};
        switch (node->op) {
        case PI_VALUE: result = (PoolKnownLow){node->constant, 32}; break;
        case PI_U32:
            if (read && a.bits == 32 && node->resource_kind == GFX_RES_UBO && !node->resource->slot &&
                read(context, GFX_RES_UBO, node->resource, a.value, &result.value) == 1) result.bits = 32;
            break;
        case PX_CHAIN:
            if (a.bits == 32 && b.bits == 32 && b.value <= UINT32_MAX - a.value)
                result = (PoolKnownLow){a.value + b.value, 32};
            break;
        case PX_IN_CHAIN:
            if (a.bits == 32 && a.value <= UINT32_MAX / node->constant)
                result = (PoolKnownLow){a.value * node->constant, 32};
            break;
        case PX_COMPOSITE_EXTRACT:
            if (b.bits == 32 && b.value < node->constant) result = a;
            break;
        case PX_ADD: case PX_SUB:
            result = (PoolKnownLow){node->op == PX_ADD ? a.value + b.value : a.value - b.value,
                                  pi_min_unsigned(a.bits, b.bits)};
            break;
        case PX_MUL:
            // Expand (a + stepA) * (b + stepB). Each unknown term has a proved power of two factor.
            result = (PoolKnownLow){a.value * b.value, pi_min_unsigned(32, pi_min_unsigned(a.bits + b.bits,
                pi_min_unsigned(a.bits + pi_trailing_zeros(b.value), b.bits + pi_trailing_zeros(a.value))))};
            break;
        case PX_AND: {
            const uint32_t a_known = pi_mask_low(a.bits), b_known = pi_mask_low(b.bits);
            const uint32_t mask = (a_known & b_known) | (a_known & ~a.value) | (b_known & ~b.value);
            result = (PoolKnownLow){a.value & b.value, pi_trailing_zeros(~mask)};
            break;
        }
        case PX_RSHIFT:
            if (b.bits == 32 && b.value < 32) result = (PoolKnownLow){a.value >> b.value,
                a.bits == 32 ? 32 : a.bits > b.value ? a.bits - b.value : 0};
            break;
        case PX_LSHIFT:
            if (b.bits == 32 && b.value < 32) result = (PoolKnownLow){a.value << b.value,
                pi_min_unsigned(32, a.bits + b.value)};
            break;
        default: break;
        }
        result.value &= pi_mask_low(result.bits);
        known[i] = result;
    }
    return known[plan->domain_address];
}

static void pool_sampler_domain_slot(GfxShaderPoolSamplers *samplers, uint32_t slot, unsigned mode) {
    const uint64_t bit = UINT64_C(1) << (slot % 64);
    if (mode & GFX_POOL_SAMPLER_COMPARE) samplers->compare[slot / 64] |= bit;
    if (mode & GFX_POOL_SAMPLER_ORDINARY) samplers->ordinary[slot / 64] |= bit;
}

static int shader_pool_domain_evaluate(const ShaderPoolIndices *plan, GfxShaderPoolReadDomain read,
    GfxShaderPoolReadWord read_word, void *context, GfxShaderPoolIndex *out, uint32_t capacity,
    uint32_t *count, int *comparison_sampler_zero, GfxShaderPoolSamplers *samplers) {
    if (samplers) *samplers = (GfxShaderPoolSamplers){0};
    if (count) *count = 0;
    if (comparison_sampler_zero) *comparison_sampler_zero = 0;
    if (!plan || !plan->domain_source || !read || !out || !count || capacity < plan->result_count) return 0;
    const uint8_t *bytes = NULL;
    uint32_t size = 0;
    const int result = read(context, plan->domain_source, &bytes, &size);
    if (result != 1) return result;
    if (!bytes || !size || (size & 3)) return 0;
    const PoolKnownLow address = pi_domain_low_bits(plan, read_word, context);
    const uint32_t mask = pi_mask_low(address.bits);
    const uint32_t residue = (address.value + plan->domain_base_word) & mask;
    int only_compare_zero = plan->sampler_modes == GFX_POOL_SAMPLER_COMPARE;
    // A robust out-of-range storage read and the local initializer add a possible zero handle.
    for (uint32_t offset = 0;; offset += 4) {
        uint32_t word = 0;
        if (offset < size) {
            if (((offset / 4) & mask) != residue) continue;
            memcpy(&word, bytes + offset, sizeof word);
        }
        if (word >> 20) only_compare_zero = 0;
        if (samplers) pool_sampler_domain_slot(samplers, word >> 20, plan->sampler_modes);
        uint32_t index = word & UINT32_C(0xfffff);
        if (index > plan->domain_maximum) index = plan->domain_maximum;
        for (uint16_t site = 0; site < plan->result_count; ++site) {
            const uint16_t type = plan->results[site].type;
            uint32_t seen = 0;
            while (seen < *count && (out[seen].index != index || out[seen].type != type)) ++seen;
            if (seen != *count) continue;
            if (*count == capacity) {
                if (samplers) *samplers = (GfxShaderPoolSamplers){0};
                return *count = 0, 0;
            }
            out[(*count)++] = (GfxShaderPoolIndex){.index = index, .type = type};
        }
        if (offset == size) {
            if (comparison_sampler_zero) *comparison_sampler_zero = only_compare_zero;
            return 1;
        }
    }
}

static int shader_pool_domain_eval(const ShaderPoolIndices *plan, GfxShaderPoolReadDomain read,
    GfxShaderPoolReadWord read_word, void *context,
    GfxShaderPoolIndex *out, uint32_t capacity, uint32_t *count, int *comparison_sampler_zero) {
    return shader_pool_domain_evaluate(plan, read, read_word, context, out, capacity, count, comparison_sampler_zero, NULL);
}

static int shader_pool_indices_evaluate(const ShaderPoolIndices *plan,GfxShaderPoolReadWord read,void *context,
    GfxShaderPoolIndex *out,uint32_t capacity,uint32_t *count,GfxShaderPoolSamplers *samplers) {
    if (samplers) *samplers = (GfxShaderPoolSamplers){0};
    if (count) *count=0;
    if (!plan || plan->domain_source || !read || !out || !count || capacity<plan->result_count) return 0;
    uint32_t values[PI_NODE_LIMIT]={0};
    uint8_t ready[PI_NODE_LIMIT]={0};
    ready[0]=1;  // Unused operands do not stand for a read.
    const uint16_t nodes = samplers ? plan->node_count : plan->texture_node_count;
    for (uint16_t i=1;i<nodes;++i) {
        const PoolIndexNode *n=&plan->nodes[i];
        if (!ready[n->a]) continue;
        uint32_t a=values[n->a],v=0;
        uint16_t second=n->op==PX_SELECT && !a ? n->c : n->b;
        if (!ready[second]) continue;
        uint32_t b=values[second];
        switch (n->op) {
        case PI_VALUE: v=n->constant; break;
        case PI_U32: {
            int captured=read(context,n->resource_kind,n->resource,a,&v);
            if (captured==2) continue;
            if (captured!=1) return 0;
            break;
        }
        case PI_UMIN: v=a<b?a:b; break;
        case PX_CHAIN: if (b>UINT32_MAX-a) return 0; v=a+b; break;
        case PX_IN_CHAIN: if (a>UINT32_MAX/n->constant) return 0; v=a*n->constant; break;
        case PX_COMPOSITE_EXTRACT: if (b>=n->constant) return 0; v=a; break;
        case PX_ADD: v=a+b; break;
        case PX_SUB: v=a-b; break;
        case PX_MUL: v=a*b; break;
        case PX_LSHIFT: if (b>=32) return 0; v=a<<b; break;
        case PX_RSHIFT: if (b>=32) return 0; v=a>>b; break;
        case PX_NOT: v=!a; break;
        case PX_EQ: v=a==b; break;
        case PX_NE: v=a!=b; break;
        case PX_OR: v=a|b; break;
        case PX_XOR: v=a^b; break;
        case PX_AND: v=a&b; break;
        case PX_SELECT: v=b; break;
        default: return 0;
        }
        values[i]=v;
        ready[i]=1;
    }
    for (uint16_t i=0;i<plan->result_count;++i)
        if (!ready[plan->results[i].node]) return 2;
    if (samplers) {
        for (uint16_t i = 0; i < plan->sampler_count; ++i) {
            const uint16_t node = plan->sampler_nodes[i].node;
            if (!ready[node]) return 2;
            if (values[node] >= 4096) return 0;
        }
        for (uint16_t i = 0; i < plan->sampler_count; ++i) {
            const PoolSamplerResult *result = &plan->sampler_nodes[i];
            pool_sampler_domain_slot(samplers, values[result->node], result->mode);
        }
    }
    for (uint16_t i=0;i<plan->result_count;++i)
        out[i]=(GfxShaderPoolIndex){.type=plan->results[i].type,.index=values[plan->results[i].node]};
    *count=plan->result_count;
    return 1;
}


static int shader_pool_indices_eval(const ShaderPoolIndices *plan,GfxShaderPoolReadWord read,void *context,
    GfxShaderPoolIndex *out,uint32_t capacity,uint32_t *count) {
    return shader_pool_indices_evaluate(plan,read,context,out,capacity,count,NULL);
}
