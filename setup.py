import os
import subprocess
from setuptools import Extension, setup

def is_foma_installed():
    test_program_c = """
    #include <fomalib.h>
    int main(void) {
        return 0;
    }
    """
    try:
        compiler = os.environ.get("CC", "cc")
        subprocess.run(
            [compiler, "-x", "c", "-", "-o", "/dev/null", "-lfoma"],
            input=test_program_c.encode('utf-8'),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("INFO: Foma library found. Building with Foma support.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Foma library not found. Building without Foma support.")
        return False

sources = [
    "src/lib.c",
    "src/bpe.c",
    "src/core.c",
    "src/hash.c",
    "src/hashmap.c",
    "src/helper.c",
    "src/string.c",
    "src/pretokenizer.c",
    "src/bbpe.c",
]

include_dirs = ["include"]
extra_compile_args = ["-O3", "-march=native", "-funroll-loops", "-Iinclude"]
extra_link_args = ["-flto","-lpcre2-8"]

if is_foma_installed():
    extra_compile_args.append("-DUSE_FOMA")
    extra_link_args.append("-lfoma")

ext_modules = [
    Extension(
        name="_hutoken",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=ext_modules,
)
