import os
import sys
import setuptools
from setuptools.command.install import install

CURRENT_DIR = os.getcwd()
REQUIREMENTS = 'requirements.txt'
requires = [line.strip('\n') for line in open(REQUIREMENTS).readlines()]
with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = "0.0.1"


setuptools.setup(
    name="Adversarials",
    version=VERSION,
    author='Domnan Diretnan',
    author_email="diretnandomnan@gmail.com",
    description="easy wrapper for initializing several GAN networks in keras",
    url="https://github.com/deven96/Simple_GAN",
    packages=setuptools.find_packages(),
    install_requires=requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords='keras GAN GANs networks adversarial',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    package_data={
        '': ['*.*'],
    },
    include_package_data=True,
)