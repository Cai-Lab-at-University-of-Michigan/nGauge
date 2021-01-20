# nGauge
`nGauge` is a Python library which provides a collection of tools for the measurement, quantification, and visualization of neuron morphology. The library structure makes automation of data analysis pipelines easier, leading to faster and more reproducible results.

## Installation

### To install with `pip` on a `python 3.x` installation:

`pip install ngauge`

See repository information here: <https://pypi.org/project/nGauge/>

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

## Example Usage

## Contact
Logan Walker <loganaw@umich.edu>
Jennifer Williams <jenwill@umich.edu>

## License

This code is licensed under the GNU General Public License v3.0, a copy of which is available in [LICENSE](LICENSE).
