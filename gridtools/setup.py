from setuptools import find_packages, setup

packages = ['gridtools',
            'gridtools.toolbox',
            'gridtools.prepro']

setup(
    name="gridtools",
    version="0.0.1",
    author="Patrick Weygoldt",
    author_email="weygoldt@pm.me",
    description="",
    packages=find_packages(exclude=["tests*", "exceptions"]),
    entry_points = {
        'console_scripts': ['prepro=gridtools.prepro.prepro:main']},
    install_requires = ['sklearn', 'scipy', 'numpy', 'matplotlib', 'thunderfish'],
)
