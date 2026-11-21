// Synthetic structural writer coverage: no game shader bytes.
#include "../runtime/resource_core_inventory.c"
#include <assert.h>
#include <stdio.h>
#include <string.h>
typedef struct { uint32_t w[512], n; } Module;
static void emit(Module *m, uint32_t op, const uint32_t *args, uint32_t n) {
    m->w[m->n++] = (n + 1) << 16 | op;
    memcpy(m->w + m->n, args, n * sizeof *args); m->n += n;
}
#define OP(m, op, ...) do { const uint32_t a[] = {__VA_ARGS__}; emit(m,op,a,sizeof a/sizeof a[0]); } while(0)
static Module base(void) {
    Module m = {{0x07230203,0x10000,0,128,0},5};
    OP(&m,21,1,32,0); OP(&m,43,1,2,0); OP(&m,43,1,3,2);
    OP(&m,30,4,1); OP(&m,32,5,12,4); OP(&m,32,6,12,1);
    OP(&m,59,5,10,12); OP(&m,59,5,11,12);
    OP(&m,71,10,34,0); OP(&m,71,10,33,7);
    OP(&m,71,11,34,0); OP(&m,71,11,33,8);
    return m;
}
typedef struct { unsigned bindings, count, gpa, image; } Coverage;
static int covered(void *ctx, const RcWriteBinding *b) {
    Coverage *c=ctx;
    if(b->kind==RC_INVENTORY_GPA)return c->gpa;
    if(b->kind==RC_INVENTORY_IMAGE)return c->image && b->binding==9 && b->count<=c->count;
    return b->set==0 && b->binding<32 && (c->bindings&(1u<<b->binding)) && b->count<=c->count;
}
static int check(Module *m,Coverage *c){return rc_write_inventory_covered(m->w,m->n,covered,c);}
int main(void) {
    Coverage c={(1u<<7)|(1u<<8),1,1,1};Module m=base();
    OP(&m,65,6,20,10,2); OP(&m,62,20,2);assert(check(&m,&c));
    OP(&m,65,6,21,11,2); OP(&m,62,21,2);c.bindings=1u<<7;assert(!check(&m,&c));
    c.bindings|=1u<<8;assert(check(&m,&c)); // Equal BUFFER class cannot hide a missing second binding.
    m=base();OP(&m,169,5,20,2,10,11);OP(&m,62,20,2);assert(check(&m,&c));
    c.bindings=1u<<7;assert(!check(&m,&c));c.bindings|=1u<<8;
    m=base();OP(&m,245,5,20,10,40,21,41);OP(&m,245,5,21,20,40,11,41);OP(&m,62,20,2);
    assert(check(&m,&c));c.bindings=1u<<7;assert(!check(&m,&c));c.bindings|=1u<<8;
    m=base();OP(&m,55,5,20);OP(&m,62,20,2);assert(!check(&m,&c)); // Unknown parameter provenance.
    m=base();OP(&m,61,5,20,10);OP(&m,62,20,2);assert(!check(&m,&c)); // Pointer fetched from memory.
    m=base();OP(&m,28,30,4,3);OP(&m,32,31,12,30);OP(&m,59,31,32,12);
    OP(&m,71,32,34,0);OP(&m,71,32,33,7);OP(&m,65,5,33,32,2);OP(&m,62,33,2);
    assert(!check(&m,&c));c.count=2;assert(check(&m,&c));c.count=1;
    m=base();OP(&m,25,30,1,1,0,0,0,2,0);OP(&m,32,31,0,30);OP(&m,59,31,32,0);
    OP(&m,71,32,34,0);OP(&m,71,32,33,9);OP(&m,61,30,33,32);OP(&m,99,33,2,2);
    assert(check(&m,&c));c.image=0;assert(!check(&m,&c));c.image=1;
    OP(&m,32,34,11,1);OP(&m,60,34,35,32,2,2);OP(&m,234,1,36,35,2,2,2);
    assert(check(&m,&c));
    m=base();OP(&m,32,30,5349,1);OP(&m,55,30,31);OP(&m,62,31,2);
    assert(check(&m,&c));c.gpa=0;assert(!check(&m,&c));c.gpa=1;
    m=base();OP(&m,32,30,5402,1);OP(&m,59,30,31,5402);OP(&m,62,31,2);OP(&m,5294,2,2,2,31);
    assert(check(&m,&c));OP(&m,6000,2);assert(!check(&m,&c));
    m=base();m.w[5]=0;assert(!check(&m,&c));assert(!rc_write_inventory_covered(NULL,0,covered,&c));
    puts("resource inventory: missing same-class binding, array coverage, Phi/Select roots, image atomics, GPA and unknown effects passed");
}
