# nGauge
`nGauge` is a Python library which provides a collection of tools for the measurement, quantification, and visualization of neuron morphology. The library structure makes automation of data analysis pipelines easier, leading to faster and more reproducible results.

## Example Usage
For example code and tutorials, please see the GitHub folder labeled ["tutorials"](tutorials/README.md).

## Installation

### Recommended Method (`pip`)

If you have  `pip` and `python 3.x` installed already, nGauge can be installed by running the following command
in the terminal:

`pip install ngauge`

Which will: <https://pypi.org/project/nGauge/>

### To install from source:

`pip install .`

### Using with Blender

By default, many versions (such as the version packaged with Ubuntu) of Blender
support packages installed to the system Python library; in this case, no additional
steps are required.

For Blender versions which do not share Python with the system-level install (such
as Windows and `snap` versions) can be used with one of these two methods:

1. If the Python version of the system and Blender are the same, the system
library path can be appended to the Blender path (`import sys; sys.path += ['<location>']`).
2. It is also possible to directly install packages to the Blender Python libary
using `pip`, however support for this install method can not be provided.
  * For example, please see: `https://blender.stackexchange.com/questions/56011/how-to-install-pip-for-blenders-bundled-python`.

## Citation

nGauge has been submitted for publication. 

## Contact
 * Logan Walker <loganaw@umich.edu>
 * Dawen Cai <dwcai@umich.edu>

## License

This code is licensed under the GNU General Public License v3.0, a copy of which is available in [LICENSE](LICENSE).
