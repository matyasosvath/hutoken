from setuptools import setup, Extension


setup(
    name="hutoken",
    description = 'Fast BPT tokeniser for Hungarian language.',
    author = 'Mátyás Osváth',
    author_email = 'osvath.matyas@hun-ren.nytud.hu',
    version="0.1.0",
    py_modules=["hutoken"],
    ext_modules=[
        Extension(
            "_hutoken",
            ["src/lib.c", "src/bpe.c", "src/core.c", "src/hash.c", "src/hashmap.c", "src/helper.c"],
            extra_compile_args=["-O3", "-march=native", "-funroll-loops", "-Iinclude"],
            extra_link_args=["-flto", "-lfoma"]
        )
    ]
)
