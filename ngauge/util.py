"""
Provides mathematical functions used in the nGauge package
"""

from numpy import cos, sin
import numpy as np

def rotation_matrix(a, b, c):
    """Creates a standard rotation matrix of three angles for rotation
    calculations

    :param a: The first rotation angle
    :type a: numeric

    :param b: The second rotation angle
    :type b: numeric

    :param c: The third rotation angle
    :type c: numeric

    :return: The rotation matrix
    :rtype: `numpy.array` (3x3)"""

    return np.array(
        [
            [
                cos(a) * cos(b),
                cos(a) * sin(b) * sin(c) - sin(a) * cos(c),
                cos(a) * sin(b) * cos(c) + sin(a) * sin(c),
            ],
            [
                sin(a) * cos(b),
                sin(a) * sin(b) * sin(c) + cos(a) * cos(c),
                sin(a) * sin(b) * cos(c) - cos(a) * sin(c),
            ],
            [-1.0 * sin(b), cos(b) * sin(c), cos(b) * cos(c)],
        ]
    )
