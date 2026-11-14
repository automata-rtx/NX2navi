"""CPU-only checks of extracted descriptor comparisons and pool selection; no Vulkan device or game data."""
import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Frozen vk_exec_draw.c for the optional push-decision differential")
    args = parser.parse_args()
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Set CC to MSYS2 UCRT64 Clang")
    work = ROOT / "build/test/descriptor_reuse"
    work.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "runtime/vk_exec_draw.c").read_text()
    content_root = next(p for p in (ROOT, *ROOT.parents) if (p / "tools/check_content.py").is_file())
    sys.path.insert(0, str(content_root / "tools"))
    from check_content import GENERATED_MARKER
    selected = "\n".join(function(source, name) for name in
                         ("descriptor_packet_equal", "descriptor_packet_match", "pool_sync_draw"))
    (work / "descriptor_under_test.h").write_text("// " + GENERATED_MARKER + "\n" + selected)
    decision = "// " + GENERATED_MARKER + "\n"
    if args.baseline:
        for label, text in (("baseline", args.baseline.read_text()), ("candidate", source)):
            start = text.index("            uint32_t n = (gfx || prepared) && push_skip_on() ? dbytes : 0;")
            stop = text.index("                BS.push_skip++;", start)
            block = text[start:stop]
            block = block[:block.rfind("            if (")]
            result = "n && state->pushed[s] && equal" if label == "baseline" else "equal"
            decision += f"static int push_decision_{label}(DescriptorState *state, DsSig *sig, int gfx, int prepared, "
            decision += "const DescriptorPacket *packet, const VkDescSlot *ds, uint32_t dbytes, int push_equal) {\n"
            decision += "unsigned s = 0; (void)push_equal;\n" + block + "return " + result + ";\n}\n"
    (work / "descriptor_decision_under_test.h").write_text(decision)
    env = dict(os.environ)
    env["PATH"] = str(Path(cc).resolve().parent) + os.pathsep + env.get("PATH", "")
    results = []
    for diagnostic in (0, 1):
        exe = work / (f"descriptor_reuse_check_{diagnostic}" + (".exe" if os.name == "nt" else ""))
        command = [cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-fsanitize=undefined", "-fsanitize-trap=all", "-DNX2NAVI_REPLAY_WORK=1",
                   f"-DNX2NAVI_REPLAY_WORK_DESCRIPTORS={diagnostic}",
                   f"-DDESCRIPTOR_DIFFERENTIAL={int(bool(args.baseline))}",
                   "-I", str(ROOT / "runtime"), "-I", str(work),
                   str(ROOT / "tests/descriptor_reuse_check.c"), "-o", str(exe)]
        subprocess.run(command, check=True, env=env)
        result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, env=env)
        results.append({"descriptor_diagnostics": diagnostic, "output": result.stdout.strip(), "passed": True})
    (work / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
