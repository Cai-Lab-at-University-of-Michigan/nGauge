"""Setup script for nGauge"""

import os.path
from setuptools import setup

import ngauge

# The text of the README file
HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

# This call to setup() does all the work
setup(
    name="nGauge",
    version=ngauge.__version__,
    description="Perform morphology measurement on neuron SWC files",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Cai-Lab-at-University-of-Michigan/nGauge/",
    author="Logan Walker",
    author_email="logan.walker@me.com",
    license="GPLv3",
    classifiers=[
	"Development Status :: 3 - Alpha",
	"Intended Audience :: Science/Research",
	"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
	"Natural Language :: English",

        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
	"Programming Language :: Python :: 3 :: Only",

	"Topic :: Scientific/Engineering :: Visualization",
	"Topic :: Scientific/Engineering :: Image Processing",
	"Topic :: Scientific/Engineering :: Bio-Informatics",
	"Topic :: Scientific/Engineering"
    ],
    packages=["ngauge"],
    include_package_data=True,
    install_requires=[
        "scipy", "numpy"
    ],
    #entry_points={"console_scripts": ["realpython=reader.__main__:main"]},
)
