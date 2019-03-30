#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "click",
    "h5py",
    "plumbum",
    "torch>=1.0.0",
    "numpy",
    "scipy",
    "torchvision",
    "fs",
    "tqdm",
    "scikit-image",
    "allennlp",
    "tensorboardX",
    "attrs",
]

setup_requirements = []

test_requirements = ["flake8", "nose", "coverage"]

setup(
    author="Richard Weiss",
    author_email="richardweiss@richardweiss.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="MAC: memory, attention and composition",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="mac",
    name="mac",
    packages=find_packages("src", include=["mac"]),
    package_dir={"": "src"},
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.com/ririw/mac",
    version="0.1.0",
    zip_safe=False,
    entry_points={"console_scripts": ["mac-learn=mac.cli:main"]},
)
