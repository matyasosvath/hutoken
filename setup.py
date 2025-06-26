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
            ["src/lib.c"],
            extra_compile_args=["-O3", "-march=native", "-funroll-loops", "-lpcre2-8"],
            extra_link_args=["-flto"]
        )
    ]
)
