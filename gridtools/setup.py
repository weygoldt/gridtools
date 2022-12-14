from setuptools import find_packages, setup

packages = [
    "gridtools",
    "gridtools.utils",
    "gridtools.preprocessing",
    "gridtools.annotation",
]

setup(
    name="gridtools",
    version="0.0.1",
    author="Patrick Weygoldt",
    author_email="weygoldt@pm.me",
    description="",
    packages=find_packages(exclude=["tests*", "exceptions"]),
    entry_points={"console_scripts": ["prepro=gridtools.preprocessing.prepro:main"]},
    install_requires=["sklearn", "scipy", "numpy", "matplotlib", "thunderfish"],
)
