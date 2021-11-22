"""
nGauge

A python library for neuron morphology.

https://github.com/Cai-Lab-at-University-of-Michigan/nGauge
"""

__version__ = "1.0.0"
__author__ = (
    "Logan A Walker <loganaw@umich.edu> and Jennifer S Willams <jenwill@umich.edu>"
)
__credits__ = "The University of Michigan"

import ngauge.util

__num_types__ = util.__num_types__

from ngauge.util import *
from ngauge.TracingPoint import TracingPoint
from ngauge.Neuron import Neuron
