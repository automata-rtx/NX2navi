// Independent consumer-oriented coverage checks for guarded address recipes.
// Include the synthetic module builder and baseline cases, not a second evaluator.
#ifndef RC_EFFECT_RECIPES
#define RC_EFFECT_RECIPES 1
#endif
#define main effects_baseline_main
#include "resource_core_effects_check.c"
#undef main

static int contains(const RcEffectRange *ranges,size_t count,uint64_t first,uint64_t bytes) {
    for(size_t i=0;i<count;i++)
        if(ranges[i].kind==RC_EFFECT_SSBO && ranges[i].begin<=first &&
           first-ranges[i].begin<=ranges[i].size && bytes<=ranges[i].size-(first-ranges[i].begin))return 1;
    return 0;
}
static int require_possible(Module *m,uint32_t index,const char *name,int complete) {
    RcEffectInputs in={0};RcEffectRange ranges[64];size_t count=0;
    RcEffectResult result=run(m,&in,ranges,64,&count);
    int valid=result==RC_EFFECT_UNKNOWN || (result==RC_EFFECT_COMPLETE && contains(ranges,count,(uint64_t)index*4,4));
    if(complete)valid&=result==RC_EFFECT_COMPLETE;
    printf("recipe adversarial %s: result%d ranges%zu expected-index%u %s\n",name,result,count,index,valid?"safe":"FAILED");
    return valid;
}
static int transformed_undefined(void) {
    int ok=1;
    for(unsigned mode=0;mode<3;mode++) {
        Module m=base();uint32_t z=lit(&m,0),mask=lit(&m,31),limit=lit(&m,16),entry=id(&m);
        start(&m,entry);
        uint32_t input=id(&m),value=id(&m),index=id(&m),condition=id(&m),chosen=id(&m);
        if(mode==2) {uint32_t ptr=id(&m);OP(&m,65,8,ptr,10,z,z);OP(&m,61,1,input,ptr);}
        else OP(&m,1,1,input);
        OP(&m,mode==1?205:204,1,value,input);
        OP(&m,199,1,index,value,mask);
        OP(&m,176,3,condition,index,limit);
        OP(&m,169,1,chosen,condition,index,z);
        store_index(&m,z,chosen);finish(&m);
        // Undefined values propagate and each consumption may differ. OpSelect avoids a conditional branch on
        // undefined control, and every possible index stays in [0,31], so this does not rely on out-of-bounds behavior.
        ok&=require_possible(&m,mode==2?15:31,mode==0?"undefined bit-reverse select":mode==1?"undefined bit-count select":"initialized opaque select",mode==2);
    }
    return ok;
}
static int bypass_and_independent_choices(void) {
    int ok=1;
    Module m=base();OP(&m,32,15,7,1);
    uint32_t z=lit(&m,0),limit=lit(&m,16),large=lit(&m,500),entry=id(&m),yes=id(&m),no=id(&m),merge=id(&m);
    start(&m,entry);OP(&m,59,15,16,7);
    uint32_t ptr=id(&m),x=id(&m),condition=id(&m);
    OP(&m,65,8,ptr,10,z,z);OP(&m,61,1,x,ptr);OP(&m,176,3,condition,x,limit);
    OP(&m,247,merge,0);OP(&m,250,condition,yes,no);
    OP(&m,248,yes);OP(&m,62,16,x);OP(&m,249,merge);
    OP(&m,248,no);OP(&m,62,16,large);OP(&m,249,merge);
    OP(&m,248,merge);uint32_t joined=id(&m);OP(&m,61,1,joined,16);store_index(&m,z,joined);finish(&m);
    ok&=require_possible(&m,500,"bypass arm retains distant write",1);

    m=base();OP(&m,32,15,7,1);z=lit(&m,0);limit=lit(&m,1);uint32_t c100=lit(&m,100),c1000=lit(&m,1000);entry=id(&m);
    start(&m,entry);OP(&m,59,15,16,7);OP(&m,59,15,17,7);ptr=id(&m);OP(&m,65,8,ptr,10,z,z);
    for(unsigned k=0;k<2;k++) {
        x=id(&m);condition=id(&m);yes=id(&m);no=id(&m);merge=id(&m);
        OP(&m,61,1,x,ptr);OP(&m,176,3,condition,x,limit);OP(&m,247,merge,0);OP(&m,250,condition,yes,no);
        OP(&m,248,yes);OP(&m,62,16+k,z);OP(&m,249,merge);
        OP(&m,248,no);OP(&m,62,16+k,k?c1000:c100);OP(&m,249,merge);OP(&m,248,merge);
    }
    uint32_t a=id(&m),b=id(&m),sum=id(&m);OP(&m,61,1,a,16);OP(&m,61,1,b,17);OP(&m,128,1,sum,a,b);store_index(&m,z,sum);finish(&m);
    for(uint32_t index=0;index<=1100;index+=index==0?100:index==100?900:100)
        ok&=require_possible(&m,index,"independent reaching choices",1);
    return ok;
}
static int loop_iteration_identity(void) {
    Module m=base();OP(&m,32,15,7,1);
    uint32_t z=lit(&m,0),large=lit(&m,500),limit=lit(&m,16),entry=id(&m),head=id(&m),body=id(&m),cont=id(&m),exit=id(&m);
    start(&m,entry);OP(&m,59,15,16,7);OP(&m,59,15,17,7);OP(&m,62,16,z);OP(&m,62,17,large);OP(&m,249,head);
    OP(&m,248,head);uint32_t current=id(&m),condition=id(&m);OP(&m,61,1,current,16);OP(&m,176,3,condition,current,limit);
    OP(&m,246,exit,cont,0);OP(&m,250,condition,body,exit);
    OP(&m,248,body);uint32_t previous=id(&m);OP(&m,61,1,previous,17);store_index(&m,z,previous);OP(&m,62,17,current);OP(&m,249,cont);
    OP(&m,248,cont);uint32_t next=id(&m);OP(&m,61,1,next,11);OP(&m,62,16,next);OP(&m,249,head);
    OP(&m,248,exit);finish(&m);
    return require_possible(&m,500,"previous iteration value not current guard",0);
}
static int explicit_initializers(void) {
    int ok=1;
    for(unsigned initialized=0;initialized<2;initialized++) {
        Module m=base();OP(&m,32,15,7,1);uint32_t null=id(&m);OP(&m,46,1,null);uint32_t zero=lit(&m,0);
        uint32_t one=lit(&m,1),three=lit(&m,3),entry=id(&m),yes=id(&m),merge=id(&m);
        start(&m,entry);
        if(initialized)OP(&m,59,15,16,7,null);else OP(&m,59,15,16,7);
        uint32_t input=id(&m),condition=id(&m);OP(&m,61,1,input,11);OP(&m,176,3,condition,input,one);
        OP(&m,247,merge,0);OP(&m,250,condition,yes,merge);
        OP(&m,248,yes);OP(&m,62,16,three);OP(&m,249,merge);
        OP(&m,248,merge);uint32_t value=id(&m);OP(&m,61,1,value,16);store_index(&m,zero,value);finish(&m);
        RcEffectInputs in={0};RcEffectRange ranges[8];size_t count=0;RcEffectResult result=run(&m,&in,ranges,8,&count);
        int valid=initialized ? result==RC_EFFECT_COMPLETE&&contains(ranges,count,0,4)&&contains(ranges,count,12,4) : result==RC_EFFECT_UNKNOWN;
        printf("recipe initializer %s: %s\n",initialized?"explicit null covers bypass":"absent stays uncertain",valid?"safe":"FAILED");ok&=valid;
    }
    Module m=base();OP(&m,32,15,7,1);uint32_t null=id(&m);OP(&m,46,1,null);uint32_t zero=lit(&m,0);
    uint32_t limit=lit(&m,16),entry=id(&m),head=id(&m),body=id(&m),cont=id(&m),exit=id(&m);
    start(&m,entry);OP(&m,59,15,16,7,null);OP(&m,249,head);OP(&m,248,head);
    uint32_t value=id(&m),condition=id(&m);OP(&m,61,1,value,16);OP(&m,176,3,condition,value,limit);
    OP(&m,246,exit,cont,0);OP(&m,250,condition,body,exit);OP(&m,248,body);store_index(&m,zero,value);OP(&m,249,cont);
    OP(&m,248,cont);uint32_t next=id(&m);OP(&m,61,1,next,11);OP(&m,62,16,next);OP(&m,249,head);OP(&m,248,exit);finish(&m);
    ok&=require_possible(&m,15,"initializer not reapplied at loop backedge",1);
    return ok;
}
static int undefined_integer_operations(void) {
    int ok=1;
    for(unsigned mode=0;mode<4;mode++) {
        Module m=base();uint32_t zero=lit(&m,0),one=lit(&m,1),width=lit(&m,32),mask=lit(&m,31),limit=lit(&m,16),entry=id(&m);
        start(&m,entry);uint32_t ptr=id(&m),input=id(&m),value=id(&m),index=id(&m),condition=id(&m),chosen=id(&m);
        OP(&m,65,8,ptr,10,zero,zero);OP(&m,61,1,input,ptr);
        OP(&m,mode&1?196:134,1,value,input,mode==0?zero:mode==1?width:mode==2?one:zero);
        OP(&m,199,1,index,value,mask);OP(&m,176,3,condition,index,limit);OP(&m,169,1,chosen,condition,index,zero);
        store_index(&m,zero,chosen);finish(&m);
        ok&=require_possible(&m,mode<2?31:15,mode==0?"zero division select":mode==1?"oversized shift select":mode==2?"valid division select":"valid shift select",mode>=2);
    }
    return ok;
}
static int repeated_branch_versions(void) {
    int ok=1;
    for(unsigned distinct=0;distinct<2;distinct++) {
        Module m=base();OP(&m,32,15,7,1);OP(&m,32,18,7,3);
        uint32_t z=lit(&m,0),three=lit(&m,3),large=lit(&m,500),limit=lit(&m,16);
        uint32_t entry=id(&m),yes=id(&m),merge=id(&m),write=id(&m),exit=id(&m);
        start(&m,entry);OP(&m,59,15,16,7,large);OP(&m,59,18,17,7);
        uint32_t ptr=id(&m),input=id(&m),value=id(&m),predicate=id(&m);
        OP(&m,65,8,ptr,10,z,z);OP(&m,61,1,input,ptr);OP(&m,345,1,value,three,input,z);
        OP(&m,176,3,predicate,value,limit);OP(&m,62,17,predicate);
        uint32_t first=id(&m);OP(&m,61,3,first,17);OP(&m,247,merge,0);OP(&m,250,first,yes,merge);
        OP(&m,248,yes);OP(&m,62,16,three);OP(&m,249,merge);OP(&m,248,merge);
        if(distinct) {
            input=id(&m);value=id(&m);predicate=id(&m);
            OP(&m,61,1,input,ptr);OP(&m,345,1,value,three,input,z);OP(&m,176,3,predicate,value,limit);OP(&m,62,17,predicate);
        }
        uint32_t second=id(&m);OP(&m,61,3,second,17);OP(&m,247,exit,0);OP(&m,250,second,write,exit);
        OP(&m,248,write);uint32_t index=id(&m);OP(&m,61,1,index,16);store_index(&m,z,index);OP(&m,249,exit);OP(&m,248,exit);finish(&m);
        RcEffectInputs in={0};RcEffectRange ranges[16];size_t count=0;RcEffectResult result=run(&m,&in,ranges,16,&count);
        int valid=result==RC_EFFECT_UNKNOWN||(result==RC_EFFECT_COMPLETE&&contains(ranges,count,12,4)&&(!distinct||contains(ranges,count,2000,4)));
        if(!distinct&&RC_EFFECT_RECIPES)valid&=result==RC_EFFECT_COMPLETE&&!contains(ranges,count,2000,4);
        printf("recipe repeated actual branch %s: result%d %s\n",distinct?"distinct captured versions":"same reaching store",result,valid?"safe":"FAILED");ok&=valid;
    }
    return ok;
}
static int repeated_select_and_loop(void) {
    Module m=base();uint32_t z=lit(&m,0),three=lit(&m,3),large=lit(&m,500),limit=lit(&m,16),entry=id(&m);
    start(&m,entry);uint32_t input=id(&m),value=id(&m),condition=id(&m),first=id(&m),second=id(&m);
    OP(&m,1,1,input);OP(&m,204,1,value,input);OP(&m,176,3,condition,value,limit);
    OP(&m,169,1,first,condition,three,large);OP(&m,169,1,second,condition,first,z);store_index(&m,z,second);finish(&m);
    int ok=require_possible(&m,500,"repeated undefined select is not branch identity",0);

    m=base();OP(&m,32,15,7,1);OP(&m,32,18,7,3);uint32_t no=id(&m);OP(&m,42,3,no);
    z=lit(&m,0);three=lit(&m,3);large=lit(&m,500);limit=lit(&m,16);uint32_t two=lit(&m,2);
    entry=id(&m);uint32_t head=id(&m),body=id(&m),set=id(&m),join=id(&m),write=id(&m),after=id(&m),cont=id(&m),exit=id(&m);
    start(&m,entry);OP(&m,59,15,16,7,large);OP(&m,59,18,17,7,no);OP(&m,249,head);
    OP(&m,248,head);uint32_t iteration=id(&m),more=id(&m);OP(&m,61,1,iteration,11);OP(&m,176,3,more,iteration,two);
    OP(&m,246,exit,cont,0);OP(&m,250,more,body,exit);OP(&m,248,body);OP(&m,62,16,large);
    uint32_t ptr=id(&m);input=id(&m);value=id(&m);condition=id(&m);
    OP(&m,65,8,ptr,10,z,z);OP(&m,61,1,input,ptr);OP(&m,345,1,value,three,input,z);OP(&m,176,3,condition,value,limit);
    OP(&m,247,join,0);OP(&m,250,condition,set,join);OP(&m,248,set);OP(&m,62,16,three);OP(&m,249,join);OP(&m,248,join);
    uint32_t prior=id(&m);OP(&m,61,3,prior,17);OP(&m,247,after,0);OP(&m,250,prior,write,after);
    OP(&m,248,write);uint32_t index=id(&m);OP(&m,61,1,index,16);store_index(&m,z,index);OP(&m,249,after);OP(&m,248,after);
    OP(&m,62,17,condition);OP(&m,249,cont);OP(&m,248,cont);OP(&m,249,head);OP(&m,248,exit);finish(&m);
    ok&=require_possible(&m,500,"previous loop predicate is not current version",0);
    return ok;
}
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    int ok=1;ok&=transformed_undefined();ok&=bypass_and_independent_choices();ok&=loop_iteration_identity();
    ok&=explicit_initializers();ok&=undefined_integer_operations();ok&=repeated_branch_versions();ok&=repeated_select_and_loop();
    return ok?0:1;
}
