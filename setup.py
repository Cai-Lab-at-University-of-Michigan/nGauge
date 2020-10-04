"""Setup script for nMeasure"""

import os.path
from setuptools import setup

# The text of the README file
HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

# This call to setup() does all the work
setup(
    name="nMeasure",
    version="0.1",
    description="Perform morphology measurement on neuron SWC files",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Cai-Lab-at-University-of-Michigan/nGauge/",
    author="Logan Walker",
    author_email="logan.walker@me.com",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["nMeasure"],
    include_package_data=True,
    install_requires=[
        "scipy", "numpy"
    ],
    #entry_points={"console_scripts": ["realpython=reader.__main__:main"]},
)
