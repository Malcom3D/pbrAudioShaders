from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os

ext_modules = [
    Pybind11Extension(
        "faust_module",
        [
            "src/faust_module.cpp",
            "src/DspFaust.cpp",
        ],
        include_dirs=["src"],
        cxx_std=14,
    ),
]

setup(
    name="faust_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
