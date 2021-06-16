from distutils.core import setup

from setuptools import find_packages

setup(
    name="doodad",
    version="0.2.1dev",
    packages=find_packages(),
    license="MIT License",
    long_description=open("README.md").read(),
)
