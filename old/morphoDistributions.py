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


def branch_orders(n):
    """Creates list from 0, K with K being the max branch order in the neuron and each i in the
            list being the number of bifurcation points with that branch order

    :param n: neuron
    :type n: :class:`Neuron`

    :returns: number of bifurcation points for each branch order value
    :rtype: `list` of `int`

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> branch_orders(neuron)
        [0, 0, 4]
    """
    d = {}
    out = ms.get_branch_points(n)

    # no branch points
    if len(out) == 0:
        return [0] * (2)

    # determine how many of each branch order in neuron
    while out:
        i = out.pop()
        l = len(i.children)
        if d.get(l) == None:
            d[l] = 1
        else:
            d[l] += 1

    # fill and return vector with branch order counts
    maxOrder = max(d, key=d.get)
    x = [0] * (maxOrder + 1)
    for i in range(1, len(x)):
        if d.get(i) != None:
            x[i] = d[i]
    return x


def path_angles(n, bins=20):
    """Creates a histogram (an array of counts and an array of edges) of all path angles with
       default of 20 bins between [0, 180] degrees

    :param n: neuron for histogram to be created from
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: histogram of all path angles
    :rtype: `tuple` two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> path_angles(neuron)
        (array([0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 1, 1]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
    """
    out = ms.all_path_angles(n)
    return np.histogram(out, bins=bins, range=(0, 180))


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


def segment_lengths(n, bins=20):
    """Creates a histogram (an array of counts and an array of edges) of all euclidean segment lengths
       with default of 20 bins between 0 and maximum segment length

    :param n: neuron for histogram to be created from
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: histogram of all segment lengths
    :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> segment_lengths(neuron)
        (array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 2]),
         array([0.        , 0.12247449, 0.24494897, 0.36742346, 0.48989795,
                0.61237244, 0.73484692, 0.85732141, 0.9797959 , 1.10227038,
                1.22474487, 1.34721936, 1.46969385, 1.59216833, 1.71464282,
                1.83711731, 1.95959179, 2.08206628, 2.20454077, 2.32701526,
                2.44948974]))
    """
    out = ms.all_segment_lengths(n)
    return np.histogram(out, bins=bins, range=(0, out[-1]))


def thickness(n, bins=30):
    """Creates a histogram (an array of counts and an array of edges) of all nodes' radii, soma
       excluded, with default of 30 bins between 0 and maximum radii

    :param n: neuron for histogram to be created from
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: histogram of all thicknesses
    :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> thickness(neuron)
        (array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10]),
         array([0.        , 0.03333333, 0.06666667, 0.1       , 0.13333333,
                0.16666667, 0.2       , 0.23333333, 0.26666667, 0.3       ,
                0.33333333, 0.36666667, 0.4       , 0.43333333, 0.46666667,
                0.5       , 0.53333333, 0.56666667, 0.6       , 0.63333333,
                0.66666667, 0.7       , 0.73333333, 0.76666667, 0.8       ,
                0.83333333, 0.86666667, 0.9       , 0.93333333, 0.96666667,
                1.        ]))
    """
    q = deque(n)
    out = []
    while q:
        i = q.pop()
        out.append(i.r)
        q.extend(i)
    out.sort()
    return np.histogram(out, bins=bins, range=(0, out[-1]))


def path_length(t, out, length=0):
    """Recursive function determining path lengths for all branch points and tips to the soma
       for all neurites

    :param t: starting point for neurite
    :type t: :class:`TracingPoint`

    :param out: list to which path distances are written
    :type out: `list` of `float`

    :param length: path length before the present node
    :type length: `float`

    :returns: path length between present node and children
    :rtype: `float`
    """
    if len(t.children) == 0:
        out.append(length)
        return 0
    if len(t.children) == 2:
        out.append(length)
        return path_length(
            t.children[0], out, length + ms.euclidean_distance(t, t.children[0])
        ) + path_length(
            t.children[1], out, length + ms.euclidean_distance(t, t.children[1])
        )
    length += ms.euclidean_distance(t, t.children[0])
    return path_length(t.children[0], out, length)


def path_distance_to_soma(n, bins=20):
    """Creates a histogram (an array of counts and an array of edges) of the path length of
       each branch point and tip to the soma with default of 20 bins between 0 and maximum length

    :param n: neuron for histogram to be created from
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: histogram of all path distances
    :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> path_distance_to_soma(neuron)
        (array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 2]),
         array([0.        , 0.31975865, 0.6395173 , 0.95927595, 1.27903459,
                1.59879324, 1.91855189, 2.23831054, 2.55806919, 2.87782784,
                3.19758649, 3.51734513, 3.83710378, 4.15686243, 4.47662108,
                4.79637973, 5.11613838, 5.43589703, 5.75565568, 6.07541432,
                6.39517297]))
    """
    q = deque(n)
    out = []
    while q:
        i = q.pop()
        path_length(i, out, 0)
    out.sort()
    return np.histogram(out, bins=bins, range=(0, out[-1]))


def euclidean_distances_to_soma(n, bins=20):
    """Creates a histogram (an array of counts and an array of edges) of the Euclidean distance
    of each branch point and tip to the soma with default of 20 bins between 0 and maximum length

    :param n: neuron for histogram to be created from
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: histogram of euclidean distances
    :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> euclidean_distances_to_soma(neuron)
        (array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1]),
         array([0.   , 0.225, 0.45 , 0.675, 0.9  , 1.125, 1.35 , 1.575, 1.8  ,
                2.025, 2.25 , 2.475, 2.7  , 2.925, 3.15 , 3.375, 3.6  , 3.825,
                4.05 , 4.275, 4.5  ]))
    """
    q = deque(n)
    out = []
    while q:
        i = q.pop()
        tips = ms.get_tip_nodes(i)
        branches = ms.get_branch_points(i)
        tips += branches
        while tips:
            tip = tips.pop()
            out.append(ms.euclidean_distance(i, tip))
    return np.histogram(out, bins=bins, range=(0, out[-1]))


def sholl_intersection(a, steps=36, proj="xy"):
    """Creates a numpy array for the Sholl Intersection, which is all intersections at
       different radii a certain number of steps away from center of input

    :param a: center point
    :type a: either :class:`Neuron` or :class:`TracingPoint`

    :param steps: number of steps desired between center and maximal neurite point
    :type steps: `int`

    :param proj: which plane circle is located on
    :type: `string`

    :returns: `numpy.array` consisting of `tuples` with radii position (`float`) and
              number of intersections (`int`)
    :rtype: `numpy.ndarray`

    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> sholl_intersection(neuron)
        array([[0.15713484, 1.        ],
       [0.31426968, 1.        ],
       [0.47140452, 1.        ],
       [0.62853936, 1.        ],
       [0.7856742 , 1.        ],
       [0.94280904, 1.        ],
       [1.09994388, 2.        ],
       [1.25707872, 2.        ],
       [1.41421356, 2.        ],
       [1.5713484 , 2.        ],
       [1.72848324, 2.        ],
       [1.88561808, 2.        ],
       [2.04275292, 2.        ],
       [2.19988776, 2.        ],
       [2.3570226 , 2.        ],
       [2.51415744, 2.        ],
       [2.67129228, 2.        ],
       [2.82842712, 3.        ],
       [2.98556197, 4.        ],
       [3.14269681, 4.        ],
       [3.29983165, 2.        ],
       [3.45696649, 2.        ],
       [3.61410133, 2.        ],
       [3.77123617, 2.        ],
       [3.92837101, 2.        ],
       [4.08550585, 2.        ],
       [4.24264069, 3.        ],
       [4.39977553, 2.        ],
       [4.55691037, 1.        ],
       [4.71404521, 1.        ],
       [4.87118005, 1.        ],
       [5.02831489, 1.        ],
       [5.18544973, 1.        ],
       [5.34258457, 1.        ],
       [5.49971941, 1.        ],
       [5.65685425, 1.        ]])

       Example 1b - TracingPoint object:
           >>> neuron = from_swc("Example1.swc")
           >>> p1 = neuron.branches[0]
           >>> type(sholl_intersection(p1))
           numpy.ndarray
    """
    # determine center based on input
    if type(a) is n:
        center = a.soma_centroid()
        if proj == "xy":
            center = Point(center[0], center[1])
        elif proj == "yz":
            center = Point(center[1], center[2])
        elif proj == "xz":
            center = Point(center[0], center[2])
    elif type(a) is t:
        center = a
        if proj == "xy":
            center = Point(center.x, center.y)
        elif proj == "yz":
            center = Point(center.y, center.z)
        elif proj == "xz":
            center = Point(center.x, center.z)
        else:
            raise AttributeError("proj must be either xy, yz, or xz")
    else:
        raise AttributeError("input must be either a TracingPoint or Neuron object")

    # fills lines with all the segments in our neuron or tracing point
    out = ms.get_tip_nodes(a)
    coord = []
    while out:
        i = out.pop()
        if i.parent:
            if proj == "xy":
                coord.append(((i.x, i.y), (i.parent.x, i.parent.y)))
            elif proj == "yz":
                coord.append(((i.y, i.z), (i.parent.y, i.parent.z)))
            else:
                coord.append(((i.x, i.z), (i.parent.x, i.parent.z)))
            out.append(i.parent)
    lines = MultiLineString(coord)

    # determines how many intersections occur at each location
    maxPoint = Point(lines.bounds[2], lines.bounds[3])
    line = LineString([center, maxPoint])
    intersect = []
    for i in range(1, steps + 1):
        r = (line.length / steps) * i
        c = center.buffer(r).boundary
        i = c.intersection(lines)
        if type(i) is Point:
            intersect.append((r, 1))
        elif type(i) is LineString:
            intersect.append((r, 1))
        else:
            intersect.append((r, len(i)))
    return np.array(intersect)


def all_branch_orders(n):
    """Creates a list with all the branch orders of all bifurcation points in neuron

    :param n: neuron
    :type n: either :class:`Neuron` or :class:`TracingPoint`

    :returns: `list` with all branch orders
    :rtype: `list` of `int`

    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> all_branch_orders(neuron)
        [2, 2, 2, 2]

    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> all_branch_orders(p1)
        [2, 2, 2, 2]
    """
    q = ms.get_branch_points(n)
    out = []
    while q:
        i = q.pop()
        out.append(len(i.children))
    return out


def branch_angles_x_branch_orders(n, bins=20):
    """Creates a 2D histogram of branch angles as a function of branch orders (across all branch
            points) with default bins of 20 and range 0 to max branch order by 0 to 180 degrees

    :param n: neuron
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: 2D histogram of branch angles as a function of branch orders
    :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> branch_angles_x_branch_orders(neuron)
        (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 0., 1., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.]]),
         array([0.  , 0.15, 0.3 , 0.45, 0.6 , 0.75, 0.9 , 1.05, 1.2 , 1.35, 1.5 ,
                1.65, 1.8 , 1.95, 2.1 , 2.25, 2.4 , 2.55, 2.7 , 2.85, 3.  ]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                 99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
    """
    orders = all_branch_orders(n)
    angles = ms.all_branch_angles(n)
    return np.histogram2d(
        orders, angles, bins=bins, range=[[0, ms.max_branch_order(n)], [0, 180]]
    )


def branch_angles_x_path_distances(n, bins=20):
    """Creates a 2D histogram of branch angles as a function of path distances to the soma in
            microns (across all branch points) with default bins of 20

    :param n: neuron
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: 2D histogram of branch angles as a function of path distances
    :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> branch_angles_x_path_distances(neuron)
        (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.]]),
         array([0.        , 0.24747449, 0.49494897, 0.74242346, 0.98989795,
                1.23737244, 1.48484692, 1.73232141, 1.9797959 , 2.22727038,
                2.47474487, 2.72221936, 2.96969385, 3.21716833, 3.46464282,
                3.71211731, 3.95959179, 4.20706628, 4.45454077, 4.70201526,
                4.94948974]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                 99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
    """
    angles = ms.all_branch_angles(n)
    q = ms.get_branch_points(n)
    dist = []
    while q:
        i = q.pop()
        length = 0
        while i.parent:
            length += ms.euclidean_distance(i, i.parent)
            i = i.parent
        dist.append(length)
    distSorted = sorted(dist)
    maxDist = distSorted[-1]
    return np.histogram2d(dist, angles, bins=bins, range=[[0, maxDist], [0, 180]])


def path_angle_x_branch_order(n, bins=20):
    """Creates a 2D histogram of path angles as a function of branch orders (across all nodes)
       with default bins of 20

    :param n: neuron
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: 2D histogram of path angles as a function of branch orders
    :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges
    :note: bifurcation nodes will have multiple values in histogram associated with them due
           to multiple path angles

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> path_angle_x_branch_order(neuron)
        (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 2.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 2.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 1.]]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                 99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]),
         array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. ]))
    """
    q = deque(n)
    p = deque()
    angles = []
    orders = []
    while q:
        it = q.pop()
        p.extend(it)
    while p:
        i = p.pop()
        if i.children and len(i.children) > 1:
            for j in range(len(i.children)):
                angles.append(ms.angle(i, i.parent, i.children[j]))
                orders.append(len(i.children))
                p.append(i.children[j])
        elif i.children:
            angles.append(ms.angle(i, i.parent, i.children[0]))
            orders.append(len(i.children))
            p.append(i.children[0])
    orderSort = orders
    orderSort.sort()
    return np.histogram2d(
        angles, orders, bins=bins, range=[[0, 180], [0, orderSort[-1]]]
    )


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


def path_angle_x_path_distance(n, bins=20):
    """Creates a 2D histogram of path angles as a function of path distances to the soma in microns
       (across all nodes) with default bins of 20

    :param n: neuron
    :type n: :class:`Neuron`

    :param bins: number of bins for histogram to have
    :type bins: `int`

    :returns: 2D histogram of path angles as a function of path distances
    :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges
    :note: bifurcation nodes will have multiple values in histogram associated with them due
           to multiple path angles

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> path_angle_x_path_distance(neuron)
        (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
                 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                 0., 0., 1., 0.]]),
         array([0.        , 0.24747449, 0.49494897, 0.74242346, 0.98989795,
                1.23737244, 1.48484692, 1.73232141, 1.9797959 , 2.22727038,
                2.47474487, 2.72221936, 2.96969385, 3.21716833, 3.46464282,
                3.71211731, 3.95959179, 4.20706628, 4.45454077, 4.70201526,
                4.94948974]),
         array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                 99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
    """
    q = deque(n)
    p = deque()
    angles = []
    dist = []
    maxLen = 0
    while q:
        it = q.pop()
        p.extend(it)
    while p:
        i = p.pop()
        path_angle_x_path_dist_helper(i, angles, dist)
    sortDist = dist
    sortDist.sort()
    return np.histogram2d(dist, angles, bins=bins, range=[[0, sortDist[-1]], [0, 180]])


def thickness_x_branch_order(n, bins=20):
    """Creates 2D histogram of neurite radii as a function of branch orders (across all nodes)

    :param n: neuron
    :type n: :class:`Neuron`

    :returns: 2D histogram of thickness as a function of branch orders
    :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> thickness_x_branch_order(neuron)
        (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 6.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 4.]]),
         array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
                1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. ]),
         array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
                0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ]))
    """
    q = deque(n)
    maxRad = maxOrder = 0
    radii = []
    orders = []
    while q:
        i = q.pop()
        if i.r > maxRad:
            maxRad = i.r
        radii.append(i.r)
        if len(i.children) > maxOrder:
            maxOrder = len(i.children)
        if len(i.children) > 1:
            orders.append(len(i.children))
        else:
            orders.append(1)
        if i.children:
            q.extend(i)
    return np.histogram2d(orders, radii, bins=bins, range=[[0, maxOrder], [0, maxRad]])


def thickness_x_path_distance(n, bins=20):
    """Creates 2D histogram of neurite radii as a function of path distances to the soma in microns
            (across all nodes)

    :param n: neuron
    :type n: :class:`Neuron`

    :returns: 2D histogram of thickness as a function of path distances
    :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> thickness_x_path_distance(neuron)
        (array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 2.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 4.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 2.]]),
         array([0.        , 0.31975865, 0.6395173 , 0.95927595, 1.27903459,
                1.59879324, 1.91855189, 2.23831054, 2.55806919, 2.87782784,
                3.19758649, 3.51734513, 3.83710378, 4.15686243, 4.47662108,
                4.79637973, 5.11613838, 5.43589703, 5.75565568, 6.07541432,
                6.39517297]),
         array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
                0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ]))
    """
    q = deque(n)
    radii = []
    dist = []
    maxRadii = maxDist = 0
    while q:
        i2 = i = q.pop()
        length = 0
        if i.r > maxRadii:
            maxRadii = i.r
        radii.append(i.r)
        while i.parent:
            length += ms.euclidean_distance(i, i.parent)
            i = i.parent
        if length > maxDist:
            maxDist = length
        dist.append(length)
        q.extend(i2)
    return np.histogram2d(dist, radii, bins=bins, range=[[0, maxDist], [0, maxRadii]])
