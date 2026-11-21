"""Exercise the real registration mutation guard with a concurrently resolving recorder."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from bind_recipe_check import function

backend = (ROOT / "runtime/vk_backend.c").read_text()
registration = (ROOT / "runtime/vk_texture_registrations.h").read_text()
texture = (ROOT / "runtime/vk_texture.c").read_text()
guard = function(texture, "vk_texture_register_in_pool_ordered")
guard = guard[:guard.index("    int id = texture_pool_claim")]
prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "resource_core_registry.c"
typedef struct VkTextureRegistrationScope {
    RcRegistryScope horizon;uint64_t publication;uint32_t references;
} VkTextureRegistrationScope;
typedef struct {void *mem; uint64_t serial;} tex_t;
typedef int view_t;
typedef struct {tex_t descriptor;uint64_t publication;RcRegistryHandle handle;} TextureRegistration;
typedef struct {TextureRegistration *registration;} TexSlot;
static TexSlot tslots[2];
static RcRegistry *texture_registrations;
static uint64_t texture_registration_publication=1,texture_recording_revision,texture_recording_next;
static VkTextureRegistrationScope *texture_recording_scope[2];
static RcRegistry *texture_registration_store(void) {return texture_registrations;}
static CRITICAL_SECTION texture_lock,rl;
static CONDITION_VARIABLE rcv;
static HANDLE waiting,signalled;
static _Thread_local int tex_depth,held_rl;
static uint64_t r_registration_issued,r_registration_done,r_registration_publication;
enum { DR_REGISTRATION, PERF_DRAIN };
static unsigned dr_calls[1],dr_tcalls[1];
static void lock(void){EnterCriticalSection(&texture_lock);tex_depth++;}
static void unlock(void){assert(tex_depth);tex_depth--;LeaveCriticalSection(&texture_lock);}
static void rt_fatal(const char *message,...){fputs(message,stderr);abort();}
static void plat_mutex_lock(CRITICAL_SECTION *m){assert(m==&rl&&!tex_depth&&!held_rl);EnterCriticalSection(m);held_rl=1;}
static void plat_mutex_unlock(CRITICAL_SECTION *m){assert(m==&rl&&held_rl);held_rl=0;LeaveCriticalSection(m);}
static void plat_cond_broadcast(CONDITION_VARIABLE *c){assert(held_rl);WakeAllConditionVariable(c);}
static void plat_cond_wait(CONDITION_VARIABLE *c,CRITICAL_SECTION *m){
    assert(m==&rl&&held_rl&&!tex_depth);
    assert(SetEvent(waiting));
    assert(SleepConditionVariableCS(c,m,5000));
}
static uint64_t perf_begin(void){return 1;}
static int core_away_unlocked(void){assert(held_rl&&!tex_depth);return 1;}
static void core_back_unlocked(int *away){assert(held_rl&&*away);}
static void wait_counted(int why,uint64_t start){assert(why==DR_REGISTRATION&&start==1);}
static int vk_texture_pool_slot_locked(uint64_t owner,int logical){
    assert(owner==1&&logical>=0&&logical<2);return tslots[logical].registration?logical:-1;
}
static int texture_registration_same(const TexSlot *slot,const tex_t *t,const view_t *v){
    (void)v;return slot->registration->descriptor.serial==t->serial;
}
static TextureRegistration records[3];
static int assigned;
static void publish(int logical,const tex_t *t){
    assert(tex_depth&&assigned<3);
    TextureRegistration *record=&records[assigned++];
    assert(rc_registry_register(texture_registration_store(),(uint64_t)logical+1,t->serial,
        (RcSourceRange){1,0x1000,64},t->serial*10,&record->handle)==RC_REGISTRY_OK);
    record->publication=++texture_registration_publication;
    record->descriptor=*t;tslots[logical].registration=record;
}
'''
parts = [
    "void vk_backend_registration_record_progress(void);",
    *(function(registration, name) for name in (
        "texture_registration_scope_new", "texture_recording_revision_update", "vk_texture_registration_record_begin",
        "vk_texture_registration_record_end", "vk_texture_registration_record_revision",
        "texture_registration_cutoff_locked", "texture_registration_pending")),
    *(function(backend, name) for name in (
        "replay_registration_cutoff", "vk_backend_registration_record_progress",
        "replay_registration_pending", "replay_registration_wait")),
    guard + "    publish(logical,t);unlock();return 1;\n}",
]
main = r'''
static const tex_t previous={ (void *)0x1000,17 },replacement={ (void *)0x1000,23 },other={ (void *)0x2000,29 };
static DWORD WINAPI mutate(void *argument){
    (void)argument;
    for(;;){
        uint64_t publication,recording;
        uint64_t cutoff=replay_registration_cutoff(&publication,&recording);
        if(vk_texture_register_in_pool_ordered(1,0,&replacement,NULL,publication))break;
        replay_registration_wait(cutoff,recording);
    }
    assert(SetEvent(signalled));return 0;
}
int main(void){
    InitializeCriticalSection(&texture_lock);
    InitializeCriticalSection(&rl);
    InitializeConditionVariable(&rcv);
    waiting=CreateEvent(NULL,TRUE,FALSE,NULL);signalled=CreateEvent(NULL,TRUE,FALSE,NULL);assert(waiting&&signalled);
    texture_registrations=rc_registry_create((RcRegistryConfig){8,1,4});assert(texture_registrations);
    assert(vk_texture_register_in_pool_ordered(1,0,&previous,NULL,0));
    VkTextureRegistrationScope *base=vk_texture_registration_record_begin(NULL,0);
    uint64_t active=vk_texture_registration_record_revision();assert(active);
    HANDLE writer=CreateThread(NULL,0,mutate,NULL,0,NULL);assert(writer);
    assert(WaitForSingleObject(waiting,5000)==WAIT_OBJECT_0);
    assert(WaitForSingleObject(signalled,0)==WAIT_TIMEOUT);
    assert(vk_texture_register_in_pool_ordered(1,1,&other,NULL,0)); // Brand-new registration is unrelated.
    RcRegistryRecord record;
    assert(rc_registry_resolve(texture_registrations,1,base->horizon,&record)==RC_REGISTRY_OK);
    assert(record.interpretation==170);
    VkTextureRegistrationScope *continuation=vk_texture_registration_record_begin(NULL,1);
    assert(vk_texture_registration_record_revision()!=active);
    vk_texture_registration_record_end(base,0); // Still resolving cold reads after an actual event resume.
    assert(WaitForSingleObject(signalled,0)==WAIT_TIMEOUT);
    vk_texture_registration_record_end(continuation,1); // The recorder returns from a wait before its guest signal.
    assert(WaitForSingleObject(signalled,5000)==WAIT_OBJECT_0);
    assert(WaitForSingleObject(writer,5000)==WAIT_OBJECT_0);
    assert(!vk_texture_registration_record_revision());
    assert(rc_registry_resolve(texture_registrations,1,base->horizon,&record)==RC_REGISTRY_OK);
    assert(record.interpretation==170); // Retained historical incarnation stays independent of live replacement.
    VkTextureRegistrationScope *fresh=vk_texture_registration_record_begin(NULL,0);
    assert(rc_registry_resolve(texture_registrations,1,fresh->horizon,&record)==RC_REGISTRY_OK);
    assert(record.interpretation==230);
    vk_texture_registration_record_end(fresh,0);
    assert(rc_registry_scope_end(texture_registrations,base->horizon)==RC_REGISTRY_OK);
    assert(rc_registry_scope_end(texture_registrations,continuation->horizon)==RC_REGISTRY_OK);
    assert(rc_registry_scope_end(texture_registrations,fresh->horizon)==RC_REGISTRY_OK);
    free(base);free(continuation);free(fresh);rc_registry_destroy(texture_registrations);
    CloseHandle(writer);CloseHandle(signalled);CloseHandle(waiting);
    DeleteCriticalSection(&rl);DeleteCriticalSection(&texture_lock);
    puts("texture recorder cutoff: active sync and resumed cold readers, unrelated publisher, concurrent completion and historical/live scopes passed");
    return 0;
}
'''
cc=Path(os.environ["CC"])
env=dict(os.environ,PATH=str(cc.parent)+os.pathsep+os.environ.get("PATH",""))
with tempfile.TemporaryDirectory(prefix="registration-recorder-",dir=ROOT/"build") as tmp:
    code=Path(tmp)/"check.c"
    code.write_text(prefix+"\n".join(parts)+main)
    exe=code.with_suffix(".exe")
    subprocess.run([str(cc),"-std=gnu11","-Wall","-Wextra","-Werror","-fsanitize=undefined",
                    "-fsanitize-trap=all","-I",str(ROOT/"runtime"),str(code),"-o",str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)
