"""CPU-only public NVN dispatch and projected ReadTexels contract."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument("--overlay",type=Path)
args=parser.parse_args()
runtime=args.root / "runtime"
overlay=(args.overlay / "runtime") if args.overlay else runtime

def load(name):
    modified=overlay / name
    return (modified if modified.exists() else runtime/name).read_text(encoding="utf-8")

def extract(source,name):
    start=re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b"+name+r"\([^;{}]*\)\s*\{",source,re.M)
    if not start:
        raise ValueError("Missing production function " + name)
    depth=1
    end=start.end()
    while depth:
        depth += (source[end]=="{")-(source[end]=="}")
        end+=1
    return source[start.start():end]

def typedef(source,tag):
    stop=source.index("} "+tag+";")+len("} "+tag+";")
    return source[source.rindex("typedef struct {",0,stop):stop]

tex=load("vk_texture.c")
alias_layout=load("vk_texture_alias_bytes.h").split("enum { VK_TEXTURE_ALIAS_MAX_PLANES",1)[1]
alias_types=tex[tex.index("typedef struct {\n    VkBuffer buffer;\n    VkBufferCopy region;"):
                tex.index("static int image_alias_guest_mapping_current(uint64_t address",tex.index("struct VkTextureAliasRead {"))]
layout=load("tex_deswizzle.c")
box_layout="\n".join((typedef(layout,"TexelBoxLayout"),typedef(layout,"TexelRanges"),
    *(extract(layout,name) for name in ("texel_box_layout","texel_range","nvn_texels_box","nvn_texels_box_ranges","nvn_texels_slice_ranges"))))
texture_names=("region_of","read_aspect","texels_read_record","texels_read_copy_rows",
    "image_alias_guest_mapping_current","image_alias_placement_current","image_alias_source_current",
    "preserve_pool_contents_handed","image_alias_stage_buffer","native_texture_output_contains",
    "image_alias_compare_handles","image_alias_source_storage","image_alias_source_index",
    "image_alias_capture_source_table","image_alias_capture_sources","image_alias_fragment_equal",
    "image_alias_snapshot","image_alias_read_candidate_with","image_alias_read_current_with","image_alias_read_current_at",
    "image_alias_record_copies","image_alias_compose_guest","cpu_texel_region","cpu_texel_layout_handed",
    "cpu_texel_source_current","vk_texture_cpu_read_candidate","cpu_texel_cut_current",
    "cpu_texel_alias_capture_locked","cpu_texel_alias_sources_locked","cpu_texel_alias_layouts_handed_locked",
    "cpu_texel_alias_stage","cpu_texel_read_composed","vk_texture_cpu_read_texels")
texture_source="\n".join([*(typedef(tex,name) for name in ("Region","TexRead","TextureCpuReadSource")),
    *(extract(tex,name) for name in texture_names)]).replace("rc_registry_runtime(", "fixture_registry(")
memory=extract(load("memory.c"),"mem_backing_write_hold")
backend="\n".join(extract(load("vk_backend.c"),name) for name in ("replay_drain","vk_be_read_texels"))
header=load("vk.h")
enum=re.search(r"typedef enum \{\s*VK_TEXTURE_CPU_READ_OUTSIDE[^}]+}\s*VkTextureCpuReadResult;",header).group(0)
nvn=load("nvn_api_obj.c")
public="\n".join([extract(nvn,name) for name in ("texel_strides","read_texels")]+
    re.findall(r"^NVN_API\(nvnTextureReadTexels(?:Strided)?\).+",nvn,re.M))
compiler=os.environ.get("CC") or shutil.which("clang")
if not compiler:raise SystemExit("Clang required (set CC)")
clang=Path(compiler).resolve()
env=dict(os.environ,PATH=str(clang.parent)+os.pathsep+os.environ.get("PATH",""))
with tempfile.TemporaryDirectory(prefix="public-cpu-read-") as temp:
    build=Path(temp)
    for name,value in (("texture",texture_source),("memory",memory),("backend",backend),("abi",enum),("nvn",public),
        ("alias_layout","enum { VK_TEXTURE_ALIAS_MAX_PLANES"+alias_layout),("alias_types",alias_types),
        ("box_layout",box_layout)):
        (build/f"fixture_{name}.h").write_text(value,encoding="utf-8")
    c=(args.overlay / "tests" / "public_cpu_read_fixture.c") if args.overlay else Path(__file__).with_name("public_cpu_read_fixture.c")
    if not c.exists():
        c=Path(__file__).with_name("public_cpu_read_fixture.c")
    registry=overlay / "resource_core_registry.c"
    if not registry.exists():
        registry=runtime / "resource_core_registry.c"
    registry_headers=overlay if (overlay/"resource_core_registry.h").exists() else runtime
    exe=build/("check.exe" if os.name=="nt" else "check")
    subprocess.run([str(clang),"-std=gnu11","-O1","-Wall","-Wextra","-Werror","-fsanitize=undefined",
        "-fsanitize-trap=all","-Wno-unused-function","-DRC_REGISTRY_SINGLE_THREADED=1","-I"+str(build),
        "-I"+str(registry_headers),"-I"+str(runtime),str(c),str(registry),"-o",str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)
