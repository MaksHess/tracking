#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

requirements = [
    "h5py",
    "numpy",
    "xarray",
    "polars",
    "pandas",
    "tqdm",
    "toolz",
    "napari[all]",
    "spatial_image",
    "imageio",
    "zarr",
    "scikit-learn",
    "colorcet",
    "pyarrow",
    "btrack",
    "dask",
]

test_requirements = []

setup(
    author="Max Timo Hess",
    author_email="max.hess@mls.uzh.ch",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="""Cell tracking for BIO325""",
    # entry_points={
    #     "console_scripts": [
    #         "zfish=zfish.cli:main",
    #     ],
    # },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    # include_package_data=True,
    # keywords="tracking",
    # name="tracking",
    packages=find_packages(include=["tracking", "tracking.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/MaksHess/tracking",
    version="0.1.0",
    zip_safe=False,
)
