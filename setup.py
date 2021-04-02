"""Setup script for nGauge"""

import os.path
from setuptools import setup

# The text of the README file
HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()


### Get version number
version = ''
for line in open('ngauge/__init__.py'):
    if '__version__' in line:
        line = line.split( '=' )[1]
        line = line.replace( '"', '' )
        line = line.replace( "'", '' )
        line = line.strip()

        version = line
        break

if not version:
    raise ValueError( "No Version Detected!!!" )
###

# This call to setup() does all the work
setup(
    name="nGauge",
    version=version,
    description="Perform morphology measurement on neuron SWC files",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Cai-Lab-at-University-of-Michigan/nGauge/",
    author="Logan Walker",
    author_email="loganaw@umich.edu",
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
        "scipy", "numpy", "shapely"
    ],
    #entry_points={"console_scripts": ["realpython=reader.__main__:main"]},
)
