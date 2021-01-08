import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy import stats
import morphoStats as s
import morphoDistributions as d
from src.morpho import Neuron as n
from src.morpho import TracingPoint as t
from shapely.geometry import LineString, Point, MultiLineString
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
import math as m


def is_inside(p, center, radius):
    """Returns the difference between the point's distance from the center and the 
       the radius' distance from the center. If returned value is 0, point in on the sphere.
       If returned value is less than 0, point is inside sphere. If returned value is 
       greater than 0, point is outside sphere.
       
       :param p: point being determined
       :type p: :class:`TracingPoint`
       
       :param center: coordinates of the center [x, y, z]
       :type center: `list` of `int` 
       
       :param radius: radius of the sphere
       :type radius: `int`
       
       :returns: value for difference between the point's distance from the center and the 
       the radius' distance from the center
       :rtype: `float`
    """
    x1 = (p.x - center[0]) ** 2
    y1 = (p.y - center[1]) ** 2
    z1 = (p.z - center[2]) ** 2
    dist = x1 + y1 + z1
    return dist - (radius * radius)


def euclidean_distance_list(p1, t1):
    """Returns euclidean distance between Tracing Point object and a list of coordinates
    
    :param p1: coordinates of point [x, y, z]
    :type p1: `list` of `int`
    
    :param t1: other point
    :type t1: :class:`TracingPoint` 
    
    :returns: euclidean distance
    :rtype: `float`
    """
    return ((p1[0] - t1.x) ** 2 + (p1[1] - t1.y) ** 2 + (p1[2] - t1.z) ** 2) ** 0.5


def find_radius(p1, p2, pi):
    """Returns the radius of a point connected between two other points
    
    :param p1: point one
    :type p1: :class:`TracingPoint`
    
    :param p2: point two
    :type p2: :class:`TracingPoint`
    
    :param pi: coordinates of point inbetween two other point parameters [x, y, z]
    :type pi: `list` of `int` 
    
    :returns: radius for inbetween point
    :rtype: `float`
    """
    p1i = euclidean_distance_list(pi, p1)
    p2i = euclidean_distance_list(pi, p2)
    p12 = s.euclidean_distance(p1, p2)
    if p1.r == p2.r:
        return p1.r
    if p1.r < p2.r:
        return (p2.r - p1.r) * (p2i / p12) + p1.r
    else:
        return (p1.r - p2.r) * (p1i / p2i) + p2.r


def find_intersection(p1, p2, c, r):
    """Determines coordinates for intersection point along sphere between two points
    
    :param p1: point one
    :type p1: :class:`TracingPoint`
    
    :param p2: point two
    :type p2: :class:`TracingPoint`
    
    :param c: coordinates for center of sphere [x, y, z]
    :type c: `list` of `int`
    
    :param r: radius of sphere
    :type r: `float`
    
    :returns: the coordinates for the point between param p1 and p2 [x, y, z, r]
    :rtype: `list` of `int` and `float`
    """
    a = (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p2.z) ** 2
    b = -2 * (
        (p2.x - p1.x) * (c[0] - p1.x)
        + (p2.y - p1.y) * (c[1] - p1.y)
        + (c[2] - p1.z) * (p2.z - p1.z)
    )
    c = (c[0] - p1.x) ** 2 + (c[1] - p1.y) ** 2 + (c[2] - p1.z) ** 2 - r ** 2
    if (b ** 2 - 4 * a * c) > 0:
        t1 = (-b + m.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        t2 = (-b - m.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        x1 = p1.x + (p2.x - p1.x) * t1
        y1 = p1.y + (p2.y - p1.y) * t1
        z1 = p1.z + (p2.z - p1.z) * t1
        x2 = p1.x + (p2.x - p1.x) * t2
        y2 = p1.y + (p2.y - p1.y) * t2
        z2 = p1.z + (p2.z - p1.z) * t2
        if (
            ((p1.x <= x1 <= p2.x) or (p2.x <= x1 <= p1.x))
            and ((p1.y <= y1 <= p2.y) or (p2.y <= y1 <= p1.y))
            and ((p1.z <= z1 <= p2.z) or (p2.z <= z1 <= p1.z))
        ):
            r = find_radius(p1, p2, [x1, y1, z1])
            return [x1, y1, z1, r]
        else:
            r = find_radius(p1, p2, [x2, y2, z2])
            return [x2, y2, z2, p1.r]


def dendrites_vol(a, r):
    """Calculates the volume of the dendrites within a given radius
    
    :param a: object dendrites are located on
    :type a: either :class:`Neuron` or :class:`TracingPoint`
    
    :param r: the radius of the sphere surrounding the center
    :type r: `float`
    
    :returns: volume of dendrites
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> dendrites_vol(neuron, 2)
        38.79386021086252
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> dendrites_vol(p1, 2)
        38.79386021086252
    """
    # finds center of neuron
    if type(a) is n:
        x, y, z = a.soma_centroid()
        c = [x, y, z]
    elif type(a) is t:
        c = a
        c = [c.x, c.y, c.z]

    # gathers all compartments of dendrites
    q = s.get_tip_nodes(a)
    volume = 0
    pairs = []
    while q:
        i = q.pop()
        while i.parent:
            pairs.append((i.parent, i))
            i = i.parent

    # adds the volume of each pair inside sphere
    for p1, p2 in pairs:
        p1i = is_inside(p1, c, r)
        p2i = is_inside(p2, c, r)
        if p1i <= 0 and p2i <= 0:
            if p1.r == p2.r:
                volume += m.pi * (p1.r) * (p1.r) * s.euclidean_distance(p1, p2)
            else:
                volume += (
                    (1 / 3)
                    * m.pi
                    * (p1.r ** 2 + p1.r * p2.r + p2.r ** 2)
                    * s.euclidean_distance(p1, p2)
                )
        elif p1i > 0 and p2i < 0:
            p = find_intersection(p1, p2, c, r)
            volume += (
                (1 / 3)
                * m.pi
                * (p2.r ** 2 + p2.r * p[3] + p[3] ** 2)
                * euclidean_distance_list(p, p2)
            )
        elif p1i < 0 and p2i > 0:
            p = find_intersection(p1, p2, c, r)
            volume += (
                (1 / 3)
                * m.pi
                * (p1.r ** 2 + p1.r * p[3] + p[3] ** 2)
                * euclidean_distance_list(p, p1)
            )
    return volume


def dendrites_vol520(a):
    """Calculates the volume of the dendrites inside a sphere with a radius of 520 um
    
    :param a: object dendrites are located on
    :type a: either :class:`TracingPoint` or :class:`Neuron`
    
    :returns: volume of dendrites
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> dendrites_vol520(neuron)
        86.74212718397749
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> dendrites_vol520(p1)
        86.74212718397749
    """
    return dendrites_vol(a, 520)


def vol90(a):
    """Calculates the volume of a neuron inside a sphere with a radius of 90 um
    
    :param a: object dendrites are located on
    :type a: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: volume of dendrites
    :rtype: `float`
        
    Example 1 - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> vol90(p1)
        86.74212718397749
        
    Example 2 - Potential ValueError - soma only contains 1-slice (1 point):
        >>> neuron = from_swc("Example1.swc")
        >>> vol90(neuron)
        ValueError
        
    Example 3 - Neuron object:
        >>> neuron = from_swc("public_data/OlfactoryBulb/MitralCells/IF04063.CNG.swc")
        >>> vol90(neuron)
        44896.83217932993
    """
    add = 0
    if type(a) is n:
        add = a.soma_volume()
    volume = dendrites_vol(a, 90)
    return volume + add


def inter(a, r):
    """Determines the number of intersection of a neuron with a sphere of radius r
    
    :param a: object dendrites are located on
    :type a: either :class:`Neuron` or :class:`TracingPoint`
    
    :param r: the radius of the sphere surrounding the center
    :type r: `float`
    
    :returns: number of intersections
    :rtype: `int`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> inter(neuron, 2)
        5
    
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> inter(p1, 4)
        3
    """
    # finds center of sphere
    if type(a) is n:
        x, y, z = a.soma_centroid()
        center = [x, y, z]
    elif type(a) is t:
        center = [a.x, a.y, a.z]

    # gathers all compartments of the neuron
    q = s.get_tip_nodes(a)
    count = 0
    pairs = []
    while q:
        i = q.pop()
        while i.parent:
            pairs.append((i.parent, i))
            i = i.parent

    # cycles through all pairs and adds any intersection points
    for p1, p2 in pairs:
        p1i = is_inside(p1, center, r)
        p2i = is_inside(p2, center, r)
        if p1i > 0 and p2i > 0:
            # check for tangency
            a = (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p2.z) ** 2
            b = -2 * (
                (p2.x - p1.x) * (center[0] - p1.x)
                + (p2.y - p1.y) * (center[1] - p1.y)
                + (center[2] - p1.z) * (p2.z - p1.z)
            )
            c = (
                (center[0] - p1.x) ** 2
                + (center[1] - p1.y) ** 2
                + (center[2] - p1.z) ** 2
                - r ** 2
            )
            if (b ** 2 - 4 * a * c) == 0:
                count += 1
        elif p1i >= 0 and p2i < 0:
            count += 1
        elif p1i < 0 and p2i >= 0:
            count += 1
        elif p1i == 0 and p2i > 0:
            count += 1
        elif p2i == 0 and p1i > 0:
            count += 1
        elif (p1.is_tip() and p1i == 0) or (p2.is_tip() and p2i == 0):
            # if tip node is along sphere then add one
            count += 1
    return count


def inter520(a):
    """Determines the number of intersection of a neuron with a sphere of radius 520 um
    
    :param a: object dendrites are located on
    :type a: either :class:`Neuron` or :class:`TracingPoint` 
    
    :returns: number of intersections
    :rtype: `int`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("public_data/OlfactoryBulb/MitralCells/IF04063.CNG.swc")
        >>> inter520(neuron)
        6
        
    Example 1b - TracingPoint object
        >>> neuron = from_swc("public_data/OlfactoryBulb/MitralCells/IF04063.CNG.swc")
        >>> p1 = neuron.branches[0]
        >>> inter520(p1)
        0
    """
    return inter(a, 520)


def dendrites_rs1_inter(a, steps=36, proj="xy"):
    """Calculates the correlation coefficient for the intersections of the semi-log
       Sholl analysis method at each radius
    
    :param a: center point
    :type a: either :class:`Neuron` or :class:`TracingPoint`
    
    :param steps: number of steps desired between center and maximal neurite point
    :type steps: `int`
    
    :param proj: which plane circle is located on
    :type: `string`
    
    :returns: correlation coefficient
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> dendrites_rs1_inter(neuron)
        -0.953491599893758
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> dendrites_rs1_inter(p1)
        -0.953491599893758
    """
    intersections = d.sholl_intersection(a, steps=steps, proj=proj)
    x = []
    y = []
    for r, i in intersections:
        x.append(r)
        y.append(m.log10(i / (m.pi * r ** 2)))
    x = np.array(x)
    y = np.array(y)
    result = stats.linregress(x, y)
    return result.rvalue
