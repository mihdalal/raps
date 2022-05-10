from distutils.core import setup

from setuptools import find_packages

setup(
    name="d4rl",
    version="1.1",
    install_requires=[
        "gym",
        "numpy",
        "mujoco-py",
        "h5py",
        "termcolor",  # adept_envs dependency
        "click",  # adept_envs dependency
        "dm_control @ git+https://github.com/deepmind/dm_control@main#egg=dm_control",
        "mjrl @ git+git://github.com/aravindr93/mjrl@master#egg=mjrl",
    ],
    packages=find_packages(),
)
