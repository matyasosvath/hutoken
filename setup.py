from distutils.core import setup, Extension


setup(
    name="hutoken",
    description = 'Fast BPT tokeniser for Hungarian language.',
    author = 'Mátyás Osváth',
    author_email = 'osvath.matyas@hun-ren.nytud.hu',
    version="0.0.1",
    ext_modules=[
        Extension(
            "hutoken",
            ["src/lib.c"],
            extra_compile_args=["-O3", "-march=native", "-funroll-loops"],
            extra_link_args=["-flto"]
        )
    ]
)
