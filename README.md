# nGauge
`nGauge` is a Python library which provides a collection of tools for the measurement, quantification, and visualization of neuron morphology. The library structure makes automation of data analysis pipelines easier, leading to faster and more reproducible results.

## Example Usage
For example code and tutorials, please see the GitHub folder labeled ["tutorials"](tutorials/README.md).

## Installation

### Recommended Method (`pip`)

If you have  `pip` and `python 3.x` installed already, nGauge can be installed by running the following command
in the terminal: `pip install ngauge`. This will install the current release version of nGauge from the
PyPi repository: <https://pypi.org/project/nGauge/>.

### To install from source:

To install the development (source) version of nGauge to your local computer, follow these steps:

1. Clone this GitHub repository: `git clone https://github.com/Cai-Lab-at-University-of-Michigan/nGauge.git`
2. Change directory into the newly downloaded folder: `cd nGauge`
3. Run the `pip` local installation script: `pip install .`

### Using with Blender

Additional steps are required to utilize 

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

nGauge has been submitted for publication, but the preprint is currently available on bioRxiv: https://www.biorxiv.org/content/10.1101/2021.05.13.443832v2.abstract

### Cite As:

```
nGauge: Integrated and extensible neuron morphology analysis in Python
Logan A Walker, Jennifer S Williams, Ye Li, Douglas H Roossien, Nigel S Michki, Dawen Cai
bioRxiv 2021.05.13.443832; doi: https://doi.org/10.1101/2021.05.13.443832
```

## Contact
 * Logan Walker <loganaw@umich.edu>
 * Dawen Cai <dwcai@umich.edu>

## License

This code is licensed under the GNU General Public License v3.0, a copy of which is available in [LICENSE](LICENSE).
