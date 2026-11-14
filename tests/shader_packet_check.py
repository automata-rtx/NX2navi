"""Exercise production shader selection, answer-cache validity and ready packets without a GPU or game data."""
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r'^(?:static )?[^\n]*\b' + name + r'\(', source, re.M)
    if not match:
        raise RuntimeError(name)
    opening = source.index('{', match.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end] + '\n'


def main():
    cc = Path(os.environ['CC'])
    source = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    prefix = source[:source.index('static void prepared_stats_flush')]
    declarations = '''
static int shader_memo_on(void);
static uint8_t type_of_handle(uint64_t);
static uint8_t format_of_handle(uint64_t);
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];
static struct { const uint8_t *ctl; const GfxShader *s; } probe_sh[16];
static int n_probe_sh;
static unsigned resolve_calls;
static unsigned answer_calls;
static PlatMutex shader_lock = PLAT_MUTEX_INIT;
static VkFormat target_format(int, int *, int *);
static int format_type(VkFormat);
'''
    pipeline = (ROOT / 'runtime/vk_pipeline.c').read_text()
    selected = function(pipeline, 'vk_pipeline_answers').replace('{', '{\n    answer_calls++;', 1)
    selected += '\n'.join(function(source, name) for name in (
        'ctl_nvn2', 'ctl_stage_len', 'ctl_local', 'find_stage_scan', 'find_stage', 'token_of', 'slot_still',
        'slotm_idx', 'slot_mask_of', 'slot_mask_learn', 'answer_slot', 'answers_for'))
    selected += function(source, 'shader_for_resolve').replace('{', '{\n    resolve_calls++;', 1)
    sys.path.insert(0, str(ROOT / 'tools'))
    from check_content import GENERATED_MARKER
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    work = ROOT / 'build/test/shader_packet'
    work.mkdir(parents=True, exist_ok=True)
    (work / 'shader_selection_under_test.h').write_text('// ' + marker + '\n' + prefix + declarations + selected +
                                                     '\n#include "vk_shader_packet.h"\n')
    # Execute the production selection-to-side-effect boundaries. GPU command emission and motion-list storage
    # are sinks in this fixture; no copied implementation of the deferred return or its ordering is used.
    caller_source = (Path(os.environ['SHADER_CALLER_REFERENCE']) / 'runtime/vk_exec_draw.c').read_text() \
        if os.environ.get('SHADER_CALLER_REFERENCE') else source
    gates = ''
    for name, start, stop in (
        ('op_dispatch', '    int present;', '    if (knob_log_cs_binds())'),
        ('op_dispatch_ind', '    int present;', '    if (!cs || !native_tensor(ex))'),
        ('mfb_draw', '    int has_ms,', '    if (!has_ms)')):
        body = function(caller_source, name)
        region = body[body.index(start):body.index(stop, body.index(start))]
        gates += 'static void gate_' + name + '(VkExec *ex) {\n' + region + '\n route_advanced++;\n}\n'
    # Locate the definition explicitly: this function also has an earlier forward declaration.
    motion_source = caller_source[caller_source.index('static int mvobj_pick(VkExec *ex, const GfxShader **vs_out, uint32_t *banks_out, uint64_t t0) {'):]
    motion = function(motion_source, 'mvobj_pick')
    region = motion[motion.index('        cx->sf_mvprev = 1;'):motion.index('        if (!v || pending)')]
    gates += 'static int gate_motion(VkExec *ex) { DrawCtx *cx=ex->dc; int has, pending=0;\n' + region + '\n return v && !pending;\n}\n'
    gates += function(caller_source, 'mvobj_draw')
    indexed = function(caller_source, 'op_draw_idx')
    first = min(indexed.index('    vkCmdDrawIndexed('), indexed.index('    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_BYTE) mvobj_draw'))
    counter = '    vk_exec_frame.draws++;'
    assert counter in indexed[:first]
    gates += 'static void gate_indexed(VkExec *ex, const p_draw_idx *d, VkMemSpan sp) {\n' + counter + '\n' + indexed[first:]
    (work / 'shader_caller_gates.h').write_text('// ' + marker + '\n' + gates)
    exe = work / 'shader_packet_check.exe'
    env = dict(os.environ)
    env['PATH'] = str(cc.parent) + os.pathsep + env.get('PATH', '')
    subprocess.check_call([str(cc), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                           '-Wno-unused-variable', '-Wno-missing-field-initializers', '-fsanitize=undefined', '-fsanitize-trap=all',
                           '-I', str(ROOT / 'runtime'), '-I', str(work), '-I', str(cc.parent.parent / 'include/vma'),
                           str(ROOT / 'tests/shader_packet_check.c'), '-o', str(exe)], env=env)
    subprocess.check_call([str(exe)], env=env)


if __name__ == '__main__':
    main()
