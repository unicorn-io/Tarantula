import setuptools
from pathlib import Path

setuptools.setup(
    name="hex_gym",
    version="1.0.0",
    description="A OpenAi Gym Environment for a hexapod",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include="hex_gym*"),
    install_requires=['gym', 'pybullet', 'numpy']
)