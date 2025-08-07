from setuptools import setup, Extension


setup(
    name="hutoken",
    description = 'Fast BPT tokeniser for Hungarian language.',
    author = 'Mátyás Osváth',
    author_email = 'osvath.matyas@hun-ren.nytud.hu',
    version="0.1.0.dev6",
    py_modules=["hutoken"],
    ext_modules=[
        Extension(
            "_hutoken",
            ["src/lib.c", "src/bpe.c", "src/core.c", "src/hash.c", "src/hashmap.c", "src/helper.c", "src/string.c", "src/pretokenizer.c", "src/bbpe.c"],
            include_dirs=["include"],
            extra_compile_args=["-O3", "-march=native", "-funroll-loops", "-Iinclude"],
            extra_link_args=["-flto", "-lfoma"]
        )
    ],
    python_requires=">=3.8",
    install_requires=["transformers>=4.53.0",],
    extras_requires={
        "dev": ["tiktoken", "pytest", "timeit"]
    }
)
