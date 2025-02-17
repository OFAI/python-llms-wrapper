#!/usr/bin/env python
# encoding: utf-8
"""Packaging script for the llms_wrapper library."""
import sys
import os
import re
from setuptools import setup, find_packages

if sys.version_info < (3, 11):
    sys.exit("ERROR: ragability requires Python 3.11+")

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md")) as f:
    readme = f.read()


def versionfromfile(*filepath):
    infile = os.path.join(here, *filepath)
    with open(infile) as fp:
        version_match = re.search(
            r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string in {}.".format(infile))


version = versionfromfile("llms_wrapper/version.py")


setup(
    name="llms_wrapper",
    version=version,
    author="Johann Petrak",
    author_email="johann.petrak@gmail.com",
    # url="",
    description="Package for simple usage of diverse LLMs and configuring them using human readable config files",
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=[
    ],
    install_requires=[
        "litellm",
        "hjson",
        "loguru",
        "docstring_parser", 
    ],
    extras_require={
        "dev": [ "pdoc3", "pytest", "pytest-cov", "ipython", "jupyter", "jupyterlab", "ipykernel", "notebook" ],
    },
    python_requires=">=3.11",
    tests_require=["pytest", "pytest-cov"],
    platforms="any",
    packages=find_packages(),
    # test_suite="tests",
    entry_points={"console_scripts": [
       "llms_wrapper_test=llms_wrapper.llms_wrapper_test:main",
    ]},
    classifiers=[
        # "Development Status :: 6 - Mature",
        # "Development Status :: 5 - Production/Stable",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    project_urls={
       # "Documentation": "",
       #  "Source": "",
    },
)
