"""Exercise production shader selection, answer-cache validity and ready packets without a GPU or game data."""
import argparse
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLATE = ROOT.parents[1] / 'build' / 'xlate' / ('nx2navi_xlate.exe' if os.name == 'nt' else 'nx2navi_xlate')


def function(source, name):
    match = re.search(r'^(?:static )?[\w *]+\b' + name + r'\([^;{}]*\)\s*{', source, re.M)
    if not match:
        raise RuntimeError(name)
    opening = match.end() - 1
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end] + '\n'


def translation_probe(exe, xlate, env):
    # Runtime answers supply the same format names used by the production JSON writer.
    gfx = (ROOT / 'runtime/gfx_shader.c').read_text()
    table = gfx[gfx.index('static const char *const fmt_names[]'):].split('};', 1)[0]
    formats = [None] + re.findall(r'"([^"\n]+)"', table)
    default = re.search(r'fmt_names\[a->img_format\[i\]\] : "([^"]+)"', gfx).group(1)
    codes = json.loads(subprocess.check_output([str(exe), '--format-codes'], env=env))
    assert len(codes) == 4
    with tempfile.TemporaryDirectory(prefix='shader-packet-xlate-') as directory:
        work = Path(directory)
        binary = work / 'surface-store.bin'
        # SM86 MOV writes two data words and two zero coordinates. SUST.P writes RG at (0,0)
        # through image slot 0 (driver c[0] 0x120). The program uses an unconditional EXIT.
        pt = 7 << 12

        def mov(register, value):
            return (0x802 | pt | register << 16, value, 0xF00, 0)

        store = (0xB9B | pt | 4 << 24, (0x120 // 4) << 8 | 3 << 29, 0x00100300, 0)
        exit_ = (0x794D, 0, 0x03800000, 0)
        instructions = (mov(0, 0x3F000000), mov(1, 0x3F800000), mov(4, 0), mov(5, 0), store, exit_)
        binary.write_bytes(b''.join(struct.pack('<4I', *inst) for inst in instructions))
        names = ('float', 'uint', 'sint', 'explicit')
        jobs = [{'key': kind, 'stage': 'cs', 'code': str(binary), 'debug': False, 'local': [1, 1, 1],
                 'size': binary.stat().st_size, 'sm86': True,
                 'img': {'0': {'type': '2d', 'fmt': formats[code] if code else default}}}
                for kind, code in zip(names, codes)]
        job = work / 'jobs.json'
        job.write_text(json.dumps(jobs), encoding='utf-8')
        subprocess.run([str(xlate), str(job), str(work)], env=env, check=True)
        for name in names:
            reflection = json.loads((work / f'{name}.res.json').read_text(encoding='utf-8'))
            assert reflection['img_queried'] == [0]
            assert len(reflection['images']) == 1 and reflection['images'][0]['handle'] == 0x120 // 4
        subprocess.check_call([str(exe), '--modules', *(str(work / f'{kind}.spv') for kind in names)], env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xlate', type=Path, help='Run the synthetic SM86 translator checks with this executable')
    args = parser.parse_args()
    cc = Path(os.environ['CC'])
    source = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    prefix = source[:source.index('static void prepared_stats_flush')]
    declarations = '''
static int shader_memo_on(void);
static int gfx_type_of(const VkTexImage *, const view_t *, int);
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
    table = source.index('static const struct { uint16_t nvn; uint8_t code; VkFormat vk; } integer_storage_formats[]')
    selected += source[table:source.index('static int int_view_format(', table)]
    selected += source[source.index('enum ShaderAnswerResource'):source.index('static int answer_of_handle(')]
    selected += '\n'.join(function(source, name) for name in (
        'int_view_format', 'texel_bytes', 'storage_integer_view', 'storage_image_format_answer',
        'ctl_nvn2', 'ctl_stage_len', 'ctl_local', 'find_stage_scan', 'find_stage', 'token_of', 'slot_still',
        'slotm_idx', 'slot_mask_of', 'slot_mask_learn', 'answer_of_handle', 'answer_slot', 'shader_answer_local',
        'answers_for'))
    selected += function(source, 'shader_for_resolve_answers_mode').replace('{', '{\n    resolve_calls++;', 1)
    selected += function(source, 'shader_for_resolve_answers')
    selected += function(source, 'shader_for_resolve')
    selected += '\n#include "rc_native_image_store.h"\nenum { SPV_OP_FUNCTION = 54 };\n'
    selected += function(pipeline, 'spirv_storage_image_extended')
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
    # These caller checks use the fixture's current registration and single default texture pool.
    gates = '''
uint64_t vk_texture_registration_scope_token(void) { return 0; }
uint64_t vk_exec_texture_pool(void) { return 0; }
uint64_t vk_texture_pool_slot_token(uint64_t pool, int logical) {
    assert(!pool);
    return vk_texture_slot_token(logical);
}
'''
    # Use the actual compatibility selection through command emission sinks for both dispatch forms.
    for name, payload, locals_ in (
        ('op_dispatch', 'p_dispatch', 'int contact_work=0;'),
        ('op_dispatch_ind', 'p_dispatch_ind', 'VkMemSpan ip={0};'),
    ):
        body = function(caller_source, name)
        region = body[body.index('    const GfxShader *cs = NULL;'):]
        gates += (f'static void gate_{name}(VkExec *ex, const {payload} *d) {{\n'
                  f'    int retry=0, aa_specialized=0; {locals_}\n' + region)

    # Transformed mesh now resolves stages during ordered draw preparation, before a ready manifest emits.
    mesh = function(caller_source, 'mfb_draw_prepare')
    ordered_gate = mesh[mesh.index('    if (!ordered)'):mesh.index('    if (!draw->fallback)')]
    region = mesh[mesh.index('        result = graphics_stage_select'):mesh.index('        MfbTransformKey key;')]
    stages = caller_source[caller_source.index('typedef struct {\n    VkExec *ex;\n    int stage, present;'):]
    gates += stages[:stages.index('static int graphics_stage_parked(')]
    gates += function(caller_source, 'graphics_stage_parked') + function(caller_source, 'graphics_stage_select')
    gates += ('static VkExecGraphicsResult gate_mesh_stages(VkExec *ex, int ordered) {\n'
              '    struct { const GfxShader *ms, *ts; } selected, *mfb=&selected;\n'
              '    VkExecGraphicsResult result;\n' + ordered_gate + region + '    return result;\n}\n')

    motion = function(caller_source, 'mvobj_pick')
    region = motion[motion.index('        cx->sf_mvprev = 1;'):motion.index('        if (!v || pending)')]
    gates += 'static int gate_motion(VkExec *ex) { DrawCtx *cx=ex->dc; int has, pending=0;\n' + region + '\n return v && !pending;\n}\n'
    gates += function(caller_source, 'mvobj_draw')
    indexed = function(caller_source, 'op_draw_idx')
    first = indexed.index('    if (!graphics_storage_begin(')
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
                           str(ROOT / 'tests/shader_packet_check.c'), str(ROOT / 'runtime/rc_native_image_store.c'), '-o', str(exe)], env=env)
    subprocess.check_call([str(exe)], env=env)
    xlate = args.xlate or DEFAULT_XLATE
    if args.xlate or xlate.is_file():
        translation_probe(exe, xlate.resolve(strict=True), env)
    else:
        print('shader_packet_check: SM86 translation probe requires --xlate or the built translator')


if __name__ == '__main__':
    main()
