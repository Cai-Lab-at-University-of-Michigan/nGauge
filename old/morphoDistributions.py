import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy import stats
import morphoStats as ms
from src.morpho import Neuron as n
from src.morpho import TracingPoint as t
from shapely.geometry import LineString, Point, MultiLineString
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
import math as m
import unittest





def get_linear_reg_vector(n, tip, totalX, totalY, totalZ, count):
    """Calculates directional vector for the linear regression of the tip

    :param n: neuron all points are from
    :type n: :class:`Neuron`

    :param tip: terminal point the root angle is being calculated on
    :type tip: :class:`TracingPoint`

    :params totalX, totalY, totalZ: sum of all x, y, and z values, respectively, in n
    :type totalX, totalY, totalZ: `int`

    :param count: total nodes in n
    :type count: `int`

    :returns: directional vector
    :rtype: `list`
    """
    Vx = (totalX - tip.x) / (count - 1)
    Vy = (totalY - tip.y) / (count - 1)
    Vz = (totalZ - tip.z) / (count - 1)
    length = (Vx ** 2 + Vy ** 2 + Vz ** 2) ** 0.5
    if length == 0:
        length = 1
    Ux = Vx / length
    Uy = Vy / length
    Uz = Vz / length
    somaVec = [Ux, Uy, Uz]
    return somaVec


def angle_bn_two_vectors(somaVec, rootVec, tip):
    """Calculates the angle between two vectors both starting at the tip

    :param somaVec: linear regression vector for the tip
    :type somaVec: `list` of `float`

    :param rootVec: directional vector from root to tip
    :type rootVec: `list` of `float`

    :param tip: starting location for both directional vectors
    :type tip: :class:`TracingPoint`

    :returns: angle in degrees between [0, 180]
    :rtype: `float`
    """
    somaX = somaVec[0] - tip.x
    somaY = somaVec[1] - tip.y
    somaZ = somaVec[2] - tip.z
    rootX = rootVec[0] - tip.x
    rootY = rootVec[1] - tip.y
    rootZ = rootVec[2] - tip.z
    magSoma = (somaX ** 2 + somaY ** 2 + somaZ ** 2) ** 0.5
    magRoot = (rootX ** 2 + rootY ** 2 + rootZ ** 2) ** 0.5
    dotProduct = (somaX * rootX) + (somaY * rootY) + (somaZ * rootZ)
    calc = dotProduct / (magSoma * magRoot)
    if calc >= 1:
        return 0
    return m.acos(dotProduct / (magSoma * magRoot)) * (180 / m.pi)


def root_angles(n, bins=20):
    """Creates a histogram of all root angles in a neuron

    :param n: neuron
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :note: used a linear regression model to get directional vector, instead of finding path to soma
    :returns: histogram of all root angles
    :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> root_angles(neuron)
        (array([3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                 99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
    """

    # finds the total x, y, and z values for averages
    q = deque(n)
    totalX = totalY = totalZ = count = 0
    while q:
        i = q.pop()
        totalX += i.x
        totalY += i.y
        totalZ += i.z
        count += 1
        q.extend(i)

    # each tip node will have a root angle
    tips = ms.get_tip_nodes(n)
    out = []
    while tips:
        root = tip = tips.pop()
        # somaVec is the linear regression vector for the tip
        somaVec = get_linear_reg_vector(n, tip, totalX, totalY, totalZ, count)
        while root.parent:
            root = root.parent
        # rootVec is the directional vector from root to tip
        rootVec = [(root.x - tip.x), (root.y - tip.y), (root.z - tip.z)]
        out.append(angle_bn_two_vectors(somaVec, rootVec, tip))
    return np.histogram(out, bins=bins, range=(0, 180))


def euler_root_angles(n, bins=20):
    """Creates a histogram of all Euler root angles in a neuron

    :param n: neuron
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :note: used a linear regression model to get directional vector, instead of finding path to soma
    :returns: histogram of all root angles
    :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> euler_root_angles(neuron)
        (array([2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                 99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
    """
    # finds the total x, y, and z values for averages
    q = deque(n)
    totalX = totalY = totalZ = count = 0
    while q:
        i = q.pop()
        totalX += i.x
        totalY += i.y
        totalZ += i.z
        count += 1
        q.extend(i)

    # each tip node will have a Euler root angle
    tips = ms.get_tip_nodes(n)
    out = []
    while tips:
        root = tip = tips.pop()
        # somaVec is the linear regression vector for the tip
        somaVec = get_linear_reg_vector(n, tip, totalX, totalY, totalZ, count)
        while root.parent:
            root = root.parent
        # rootVec is the directional vector from root to tip
        rootVec = [(root.x - tip.x), (root.y - tip.y), (root.z - tip.z)]
        r = R.from_rotvec([somaVec, rootVec])
        s, r = r.as_euler("xyz", degrees=False)
        somaVec = [somaVec[0] - s[0], somaVec[1] - s[1], somaVec[2] - s[2]]
        rootVec = [rootVec[0] - r[0], rootVec[1] - r[1], rootVec[2] - r[2]]
        out.append(angle_bn_two_vectors(somaVec, rootVec, tip))
    return np.histogram(out, bins=bins, range=(0, 180))






def path_angle_x_path_dist_helper(t, angles, dist, length=0):
    """Recursive helper function that will fill angles and dist parameters with their path angles
    and distances associated with each node

    :param t: starting point for neurite
    :type t: :class:`TracingPoint`

    :param angles: list to which path angles are written
    :type angles: `list`

    :param dist: list to which path distances are written
    :type dist: `list`

    :param length: path length before the present node
    :type length: `float`

    :returns: path length between present node and children
    :rtype: `float`
    """
    if len(t.children) == 0:
        return 0
    length += ms.euclidean_distance(t, t.parent)
    if len(t.children) > 1:
        for i in range(len(t.children)):
            dist.append(length)
            angles.append(ms.angle(t, t.parent, t.children[i]))
            path_angle_x_path_dist_helper(t.children[i], angles, dist, length)
        return length
    else:
        dist.append(length)
        angles.append(ms.angle(t, t.parent, t.children[0]))
        path_angle_x_path_dist_helper(t.children[0], angles, dist, length)
        return length
