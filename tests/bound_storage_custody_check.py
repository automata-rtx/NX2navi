from pathlib import Path
import os
import re
import subprocess
import tempfile

tests = Path(__file__).resolve().parent
source = tests.parent / "runtime/vk_texture.c"
fixture = tests / "bound_storage_custody_fixture.c"
text = source.read_text(encoding="utf-8")
names=("storage_target_capture_bound_locked","vk_texture_storage_target_bound_plan",
       "vk_texture_storage_target_bound_commit","vk_texture_storage_target_release")
def function(name):
    found=re.search(r"^[^\n]*\b"+name+r"\([^;]*?\) \{",text,re.M)
    assert found, name
    depth=1
    at=found.end()
    while depth:
        depth+=(text[at]=="{")-(text[at]=="}")
        at+=1
    return text[found.start():at]
start=text.index("struct VkTextureStorageTarget {")
end=text.index("\n};", start)+3
with tempfile.TemporaryDirectory(prefix="bound-storage-host-") as temp:
    p=Path(temp)
    (p/"target.inc").write_text(text[start:end],encoding="utf-8")
    (p/"bound.inc").write_text("\n".join(map(function,names)),encoding="utf-8")
    exe=p/"bound.exe"
    subprocess.run([os.environ["CC"],"-std=c11","-Wall","-Wextra","-Werror","-O1","-fsanitize=undefined",
                    "-fsanitize-trap=all","-I"+temp,str(fixture),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=15)
