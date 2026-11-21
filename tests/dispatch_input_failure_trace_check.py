"""Exercise the original-read dispatch conflict and deferred value-only trace."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[1] / "runtime")
    parser.add_argument("--expect-legacy", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    function = runpy.run_path(str(root / "tests/ordered_graphics_feedback_check.py"))["function"]
    draw = (args.sources / "vk_exec_draw.c").read_text(encoding="utf-8")
    replay = (args.sources / "vk_exec.c").read_text(encoding="utf-8")
    header = (args.sources / "vk_exec.h").read_text(encoding="utf-8")
    traced = "VK_DISPATCH_TRACE_BINDING" in header
    assert traced != args.expect_legacy
    basics = ("dispatch_storage_sample_view_disjoint", "dispatch_storage_output_disjoint")
    names = ("dispatch_storage_conflict",)
    if traced:
        a = header.index("enum {\n    VK_DISPATCH_TRACE_BINDING")
        b = header.index("} VkExecDispatchInputTrace;", a) + len("} VkExecDispatchInputTrace;")
        types = header[a:b]
        names += ("dispatch_trace_binding", "dispatch_trace_reflection", "dispatch_trace_image",
                  "dispatch_trace_effect", "dispatch_trace_sample", "dispatch_trace_begin",
                  "dispatch_storage_input_conflict", "dispatch_storage_peer_conflict", "dispatch_storage_target_trace")
        fmt = ("sampled_receipt_hex", "dispatch_trace_operand_text", "vk_exec_dispatch_input_trace_log")
        terminal = function(replay, "prepare_dispatch_inputs")
        assert terminal.index("vk: required dispatch input") < terminal.index("vk_exec_dispatch_input_trace_log")
        transfer = function(draw, "dispatch_compat_inputs_transfer")
        assert transfer.index("!dispatch_storage_outputs_disjoint") < transfer.index("vk_exec_dispatch_input_trace_log")
        inventory = function(draw, "dispatch_storage_inventory_capture")
        assert "if (!optional_bound)\n                    dispatch_storage_target_trace" in inventory
        start = inventory.index("                if (optional_bound && result != VK_EXEC_GRAPHICS_INVALIDATED)")
        end = inventory.index("                return result;", start) + len("                return result;")
        optional = inventory[start:end]
    else:
        optional = ""
        types = ""
        fmt = ()
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="dispatch-input-trace-") as folder:
        out = Path(folder)
        (out / "types.inc").write_text(types, encoding="utf-8")
        (out / "optional.inc").write_text(optional, encoding="utf-8")
        (out / "capture.inc").write_text("\n\n".join(function(draw, name) for name in names + basics), encoding="utf-8")
        (out / "logger.inc").write_text("\n\n".join(function(replay, name) for name in fmt), encoding="utf-8")
        exe = out / "check.exe"
        fixture = root / "tests/dispatch_input_failure_trace_fixture.c"
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-sign-compare",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I" + str(out), f"-DTRACED={int(traced)}",
                        f"-DHELD_FOOTPRINT={int('dispatch_storage_material_sample_disjoint(output, sample, dispatch->read_position)' in draw)}",
                        str(fixture), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()
