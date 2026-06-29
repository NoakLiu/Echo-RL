"""Build EchoRL C++ extension modules."""

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
KERNEL_CPP = ROOT / "echo_rl" / "kernels" / "cpp"

ext_modules = [
    Pybind11Extension(
        "echo_rl.kernels._echo_kernels",
        [
            str(KERNEL_CPP / "echo_kernels.cpp"),
            str(KERNEL_CPP / "pybind_module.cpp"),
        ],
        include_dirs=[str(KERNEL_CPP / "include")],
        cxx_std=17,
    ),
]

setup(
    name="echo-rl",
    version="1.1.0",
    description="EchoRL: Bandwidth-Efficient RL with Latent Planning",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"],
    },
)
