"""Exercise the prospective sampled provider at its original read with real registry winners."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, functions


def main():
    cc = Path(os.environ.get("CC") or shutil.which("clang") or "").resolve()
    if not cc.is_file():
        raise SystemExit("Clang is required (set CC)")
    source = Path(os.getenv("ORDERED_SAMPLE_CHECK_SOURCE", ROOT / "runtime" / "vk_texture.c")).read_text(encoding="utf-8")
    production = functions(source, "image_alias_refuse", "image_alias_refuse_layout", "image_alias_finish_upload_layout", "image_alias_read_prepare_locked",
                           "image_alias_sample_prospective_locked", "image_alias_sample_entry_stale",
                           "image_alias_sample_select_locked", "vk_texture_ordered_sample_select_locked")
    template = Path(__file__).with_name("ordered_sample_provider_fixture.c").read_text(encoding="utf-8")
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="ordered-sample-provider-") as temp:
        fixture = Path(temp) / "check.c"
        exe = Path(temp) / ("check.exe" if os.name == "nt" else "check")
        fixture.write_text(template.replace("/* Extract provider */", production), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                        str(fixture), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()
