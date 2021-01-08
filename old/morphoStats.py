from collections import defaultdict, deque
from src.morpho import TracingPoint as t
from src.morpho import Neuron as n
import SWCParser as parse
import numpy as np
import scipy.linalg
from glob import glob
import statistics as stat
import math as m

# definitions of functions are from the morphometrics statistics section of https://link.springer.com/content/pdf/10.1007/s12021-020-09461-z.pdf


def num_branch_points(n):
    """Determines the number of branch points in a neurite
    
    :param n: starting node in the neurite
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: count of branch points in neurite
    :rtype: `int`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> neuron
        {Neuron of 1 branches and 1 soma layers}
        >>> num_branch_points(neuron)
        4
    
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> p1
        TracingPoint(x=0.0, y=0.0, z=0.0, r=1.0, t=1, children=[TracingPoint(x=0.0, y=1.0, z=0.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...})])
        >>> num_branch_points(p1)
        4
    """
    count = 0
    if type(n) is t:
        if len(n.children) > 1:
            count += 1
    q = deque(n)
    while q:
        i = q.pop()
        if len(i.children) > 1:
            count += 1
        q.extend(i)
    return count


def num_tips(n):
    """Determines the number of tip nodes in a neurite
    
    :param n: starting node for a neurite
    :type n: either :class:`Neuron` or :class:`TracingPoint` 
    
    :returns: count of tip nodes in neurite
    :rtype: `int`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> num_tips(neuron)
        5
    
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> num_tips(p1)
        5
    """
    count = 0
    q = deque(n)
    while q:
        i = q.pop()
        if len(i.children) is 0:
            count += 1
        q.extend(i)
    return count


def cell_height(n):
    """Determines the distance of a cell in the z direction in microns
    
    :param n: cell to measure
    :type n: :class:`Neuron`
    
    :returns: distance in microns
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> cell_height(neuron)
        1.5
    """
    maxVar = float("-inf")
    minVar = float("inf")
    q = deque(n)
    while q:
        i = q.pop()
        maxVar = max(i.z, maxVar)
        minVar = min(i.z, minVar)
        q.extend(i)
    return maxVar - minVar


def cell_width(n):
    """Determines the distance of a cell in the x direction in microns
    
    :param n: cell to measure
    :type n: :class:`Neuron`
    
    :returns: distance in microns
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> cell_width(neuron)
        7.0
    """
    maxVar = float("-inf")
    minVar = float("inf")
    q = deque(n)
    while q:
        i = q.pop()
        maxVar = max(i.x, maxVar)
        minVar = min(i.x, minVar)
        q.extend(i)
    return maxVar - minVar


def cell_depth(n):
    """Determines the distance of a cell in the y direction in microns
    
    :param n: cell to measure
    :type n: :class:`Neuron`
    
    :returns: distance in microns
    :rtype: `float`
     
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> cell_height(neuron)
        1.5
    """
    maxVar = float("-inf")
    minVar = float("inf")
    q = deque(n)
    while q:
        i = q.pop()
        maxVar = max(i.y, maxVar)
        minVar = min(i.y, minVar)
        q.extend(i)
    return maxVar - minVar


def num_stems(n):
    """Determines number of neurites extending from the soma
    
    :param n: neuron object neurites extend from
    :type n: :class:`Neuron`
    
    :returns: number of neurites
    :rtype: `int`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> num_stems(neuron)
        1
    """
    return len(n.branches)


def avg_thickness(n):
    """Determines average radius in microns across all neurites
    
    :param n: neuron object neurites extend from
    :type n: :class:`Neuron`
    
    :returns: average radius in microns
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> avg_thickness(neuron)
        1.0
    """
    count = 0
    totalR = 0
    q = deque(n)
    while q:
        i = q.pop()
        totalR += i.r
        count += 1
        q.extend(i)
    return float(totalR / count)


def euclidean_distance(t1, t2):
    """Calculates euclidean distance between two points in 3D plane
    
    :param t1: initial point
    :type t1: :class:`TracingPoint`
    
    :param t2: ending point
    :type t2: :class:`TracingPoint`
    
    :returns: euclidean distance between points
    :rtype: `float`
    
    Example 1 - Create TracingPoint objects:
        >>> from src.morpho import TracingPoint as t
        >>> p1 = t(0, 0, 0, 1, 1)
        >>> p2 = t(1, 1, 1, 1, 1)
        >>> euclidean_distance(p1, p2)
        1.7320508075688772
        
    Example 2 - Use points in Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> neuron.branches
        [TracingPoint(x=0.0, y=0.0, z=0.0, r=1.0, t=1, children=[TracingPoint(x=0.0, y=1.0, z=0.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...})])]
        >>> p1 = neuron.branches[0]
        >>> p1.children
        [TracingPoint(x=0.0, y=1.0, z=0.0, r=1.0, t=3, children=[TracingPoint(x=2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...}), TracingPoint(x=-2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...})], parent=TracingPoint(x=0.0, y=0.0, z=0.0, r=1.0, t=1, children=[{ 1, truncated }]))]
        >>> p2 = p1.children[0]
        >>> euclidean_distance(p1, p2)
        1.0
    """
    return ((t2.x - t1.x) ** 2 + (t2.y - t1.y) ** 2 + (t2.z - t1.z) ** 2) ** 0.5


def neurite_path_length(t):
    """Calculates the total path length in microns of a neurite
    
    :param t: neurite starting node
    :type t: :class:`TracingPoint`
    
    :returns: total path length
    :rtype: `float`
    
    Example - Find specific length of section on Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0].children[0].children[0] #Accesses right branch of neuron structure
        >>> neurite_path_length(p1)
        5.863427917355878  
    """
    length = 0
    branchChildren = deque(t)
    while branchChildren:
        n = branchChildren.pop()
        length += euclidean_distance(n, n.parent)
        branchChildren.extend(n)
    return length


def total_length(n):
    """Calculates total path length in microns of all neurites
    
    :param n: neuron on which length is determined
    :type n: :class:`Neuron`
    
    :returns: total path length
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> total_length(neuron)
        14.762407402922234
    """
    totalLength = 0
    q = deque(n)
    while q:
        i = q.pop()
        totalLength += neurite_path_length(i)
    return totalLength


def get_tip_nodes(a):
    """Creates deque of all tip nodes neuron or neurite
    
    :param a: starting point from which to find tip nodes
    :type a: either :class:`Neuron` or :class:`TracingPoint
    
    :returns: tip nodes
    :rtype: `deque`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> len(get_tip_nodes(neuron))
        5

    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0].children[0].children[1] #Accesses left branch of neuron structure
        >>> dq = get_tip_nodes(p1)
        >>> dq
        deque([TracingPoint(x=-3.0, y=3.0, z=1.5, r=1.0, t=3, parent=TracingPoint(x=-2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...})),
       TracingPoint(x=-1.0, y=3.0, z=1.5, r=1.0, t=3, parent=TracingPoint(x=-2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...}))])
        >>> len(dq)
        2       
    """
    # determines type of input
    if type(a) is t:
        q = deque()
        q.append(a)
    if type(a) is n:
        q = deque(a)

    out = deque()
    while q:
        i = q.pop()
        q.extend(i)
        if len(i.children) is 0:
            out.append(i)
    return out


def max_neurite_length(n):
    """Determines the path length of the longest neurite from tip to soma
    
    :param n: starting point on which finding longest neurite
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: path length of longest neurite
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> max_neurite_length(neuron)
        6.395172972263274
    
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> max_neurite_length(p1)
        6.395172972263274
    """
    b = get_tip_nodes(n)
    maxLen = 0
    while b:
        i = b.pop()
        length = 0
        while i.parent != None:
            length += euclidean_distance(i, i.parent)
            i = i.parent
        maxLen = max(length, maxLen)
    return maxLen


def max_branch_order(n):
    """Determines the maximum number of branch points passed when tracing a neurite from the tip 
    
    :param n: starting point 
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: maximum number of branch points
    :rtype: `int`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> max_branch_order(neuron)
        3
    
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> max_branch_order(p1)
        3
    """
    out = get_tip_nodes(n)
    maxBranch = 0
    while out:
        i = out.pop()
        branches = 0
        while i.parent:
            if len(i.children) > 1:
                branches += 1
            i = i.parent
        maxBranch = max(branches, maxBranch)
    return maxBranch


def all_segment_lengths(n):
    """Creates sorted list of all segment lengths in a neuron
    
    :param n: neuron for segments to come from
    :type n: :class:`Neuron`
    
    :returns: all segment lengths sorted by least to greatest distance
    :rtype: `list`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> lengths = all_segment_lengths(neuron)
        >>> type(lengths)
        <class 'list'>
        >>> lengths
        [1.0,
         1.4177446878757824,
         1.445683229480096,
         1.5,
         1.5,
         1.5,
         1.5,
         2.449489742783178,
         2.449489742783178]
    """
    out = []
    q = deque(n)
    while q:
        i = q.pop()
        branchChildren = deque(i)
        while branchChildren:
            n = branchChildren.pop()
            distance = euclidean_distance(n, n.parent)
            out.append(distance)
            q.append(n)
    out.sort()
    return out


def max_segment(n):
    """Determines longest segment's euclidean length in microns
    
    :param n: starting point on which finding longest neurite
    :type n: :class:`Neuron`
    
    :returns: longest euclidean length
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> max_segment(neuron)
        2.449489742783178
    """
    out = all_segment_lengths(n)
    return out[-1]


def median_intermediate_segment(n):
    """Determines the median path length of intermediate segments in microns
    
    :param n: starting point 
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: median path length
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> median_intermediate_segment(neuron)
        1.5
    
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> median_intermediate_segment(p1)
        1.5
    """
    q = get_tip_nodes(n)
    out = []
    b = deque()

    # ignores tip nodes
    while q:
        it = q.pop()
        if it.parent not in b:
            b.append(it.parent)

    while b:
        i = b.pop()
        if i.parent:
            out.append(euclidean_distance(i, i.parent))
            b.append(i.parent)
    out.sort()
    return stat.median(out)


def median_terminal_segment(n):
    """Determines the median path length of terminal segments in microns
    
    :param n: starting point 
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: median path length
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> median_terminal_segment(neuron)
        1.5
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> median_terminal_segment(p1)
        1.5
    """
    q = get_tip_nodes(n)
    out = []
    while q:
        i = q.pop()
        out.append(euclidean_distance(i, i.parent))
    out.sort()
    return stat.median(out)


def angle(branchPoint, a, b):
    """Determines the angle between two paths in degrees between [0, 180]
    
    :param branchPoint: point connected to both a and b
    :type branchPoint: :class:`TracingPoint`
    
    :param a: one child of branching point
    :type a: :class:`TracingPoint`
    
    :param b: other child of branching point
    :type b: :class:`TracingPoint`
    
    :returns: angle
    :rtype: `float`
    
    Example 1 - Create TracingPoint objects:
        >>> from src.morpho import TracingPoint as t
        >>> p1 = t(0, 0, 0, 1, 1)
        >>> p2 = t(1, 0, 1, 1, 1)
        >>> ba = t(1, 0, 0, 1, 1)
        >>> angle(ba, p1, p2)
        90.0
        
    Example 2 - TracingPoints in a Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> ba = neuron.branches[0].children[0]
        >>> p1 = ba.children[0]
        >>> p2 = ba.children[1]
        >>> angle(ba, p1, p2)
        109.47122063449069
    """
    ax = a.x - branchPoint.x
    ay = a.y - branchPoint.y
    az = a.z - branchPoint.z
    bx = b.x - branchPoint.x
    by = b.y - branchPoint.y
    bz = b.z - branchPoint.z
    magA = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5
    magB = (bx ** 2 + by ** 2 + bz ** 2) ** 0.5
    if magA == 0 or magB == 0:
        return 180
    dotProduct = (ax * bx) + (ay * by) + (az * bz)
    rad = dotProduct / (magA * magB)
    rad = min(rad, 1)
    rad = max(rad, -1)
    return m.acos(rad) * (180 / m.pi)


def all_path_angles(n):
    """Calculates the path angle for all nodes in a neuron, starting at the node after the root node.
            
    :param n: neuron calculation performed on
    :type n: :class:`Neuron`      
    
    :returns: all the path angles in a neuron
    :rtype: `list` of `float`
    :note: branch points, as there are then multiple new branches diverging, 
            will have multiple path angles angles added to returned list
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> all_path_angles(neuron)
           [114.0948425521107,
             114.0948425521107,
             82.17876449350037,
             162.28452765633213,
             65.9051574478893,
             132.87598866663572,
             88.65276496096037,
             172.50550517788332]
    """
    q = deque(n)
    p = deque()
    out = []
    while q:
        i = q.pop()
        p.extend(i)
    while p:
        i = p.pop()
        if len(i.children) > 1:
            for j in range(len(i.children)):
                out.append(angle(i, i.parent, i.children[j]))
                p.append(i.children[j])
        elif i.children:
            out.append(angle(i, i.parent, i.children[0]))
            p.append(i.children[0])
    return out


def median_path_angle(n):
    """Determines the median path angle across a neuron in degrees between [0, 180]
    
    :param n: neuron 
    :type n: :class:`Neuron`
    
    :returns: median path angle
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> median_path_angle(neuron)
        114.0948425521107
    """
    out = all_path_angles(n)
    out.sort()
    return stat.median(out)


def max_path_angle(n):
    """Determines the maximal path angle across a neuron in degrees between [0, 180]
    
    :param n: neuron 
    :type n: :class:`Neuron`
    
    :returns: maximal path angle
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> max_path_angle(neuron)
        172.50550517788332
    """
    out = all_path_angles(n)
    out.sort()
    return out[(len(out) - 1)]


def get_branch_points(a):
    """Creates a deque of all branch points in a neuron or neurite 
    
    :param a: starting point for gathering branch points
    :type a:  either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: all branch points
    :rtype: `deque`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> len(get_branch_points(neuron))
        4
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0].children[0].children[1] #Accesses left subtree
        >>> get_branch_points(p1)
        deque([TracingPoint(x=-2.0, y=2.0, z=1.0, r=1.0, t=3, children=[TracingPoint(x=-1.0, y=3.0, z=1.5, r=1.0, t=3, parent={...}), TracingPoint(x=-3.0, y=3.0, z=1.5, r=1.0, t=3, parent={...})], parent=TracingPoint(x=0.0, y=1.0, z=0.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...}))])
        >>> len(get_branch_points(p1))
        1
    """
    # determines type of input
    if type(a) is t:
        q = deque()
        q.append(a)
    if type(a) is n:
        q = deque(a)

    out = deque()
    while q:
        i = q.pop()
        q.extend(i)
        if len(i.children) > 1:
            out.append(i)
    return out


def all_branch_angles(n):
    """Creates a list with angles of all the branch points in a neuron in degrees between [0, 180]
    
    :param n: starting point for gathering branch point angles
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: all branch angles
    :rtype: `list`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> all_branch_angles(neuron)
        [90.8386644299175, 83.62062979155719, 83.62062979155719, 109.47122063449069]
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> all_branch_angles(p1)
        [90.8386644299175, 83.62062979155719, 83.62062979155719, 109.47122063449069]
    """
    q = get_branch_points(n)
    out = []
    while q:
        i = q.pop()
        a = deque(i)
        s1 = a.pop()
        s2 = a.pop()
        out.append(angle(i, s1, s2))
        while a:
            s1 = a.pop()
            if a:
                s2 = a.pop()
            out.append(angle(i, s1, s2))
    return out


def min_branch_angle(n):
    """Determines the minimal branch point angle in degrees between [0, 180]
    
    :param n: starting point
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: minimal branch angle
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> min_branch_angle(neuron)
        83.62062979155719
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> min_branch_angle(p1)
        83.62062979155719
    """
    out = all_branch_angles(n)
    out.sort()
    return out[0]


def avg_branch_angle(n):
    """Determines the average branch point angle in degrees between [0, 180]
    
    :param n: starting point
    :type n: either :class:`Neuron` or :class:`TracingPoint1
    
    :returns: average branch angle
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> avg_branch_angle(neuron)
        91.88778616188064
        
    Example 1b - TracingPoint object
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> avg_branch_angle(p1)
        91.88778616188064
    """
    out = all_branch_angles(n)
    return sum(out) / len(out)


def max_branch_angle(n):
    """Determines the maximal branch point angle in degrees between [0, 180]
    
    :param n: starting point
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: maximal branch angle
    :rtype: `float`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> max_branch_angle(neuron)
        109.47122063449069
        
    Example 1b - TracingPoint object
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> max_branch_angle(p1)
        109.47122063449069
    """
    out = all_branch_angles(n)
    out.sort()
    return out[(len(out) - 1)]


def max_degree(n):
    """Determines the maximal number of neurites meeting at one branch point
    
    :param n: starting point
    :param n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: maximal converging neurites at one branch point
    :rtype: `int`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> max_degree(neuron)
        2
        
    Example 1b - TracingPoint object
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> max_degree(p1)
        2
    """
    out = get_branch_points(n)
    maxDeg = 1
    while out:
        i = out.pop()
        if len(i.children) > maxDeg:
            maxDeg = len(i.children)
    return maxDeg


def neurite_tortuosity(t):
    """Determine the log(tortuosity) of a neurite - tortuosity defined as the ratio between the path
       length (each segment's length combined) and the Euclidean distance from the root and tip node
    
    :param t: the tip of a neurite
    :type t: :class:`TracingPoint` in a :class:`Neuron`

    :returns: the log(tortuosity)
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> dq = get_tip_nodes(neuron)
        >>> dq[0]
        TracingPoint(x=-3.0, y=3.0, z=1.5, r=1.0, t=3, parent=TracingPoint(x=-2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...}))
        >>> neurite_tortuosity(dq[0])
        0.04134791479135339 
    """
    tip = t
    pathLength = 0
    while t.parent:
        pathLength += euclidean_distance(t, t.parent)
        t = t.parent
    euclideanStartToEnd = euclidean_distance(t, tip)
    return m.log10(pathLength / euclideanStartToEnd)


def all_neurites_tortuosities(n):
    """Creates a sorted list of the log(tortuosity) values of all the neurites in the input
    
    :param n: starting point for neurites or neuron
    :type n: either :class:`Neuron` or :class:`TracingPoint`
    
    :returns: all log(tortuosity) values for neurites sorted least to greatest
    :rtype: `list`
    
    Example 1a - Neuron object:
        >>> neuron = from_swc("Example1.swc")
        >>> all_neurites_tortuosities(neuron)
        [0.04134791479135339,
         0.05300604176745361,
         0.14956195330433844,
         0.15049238421642142,
         0.18919849587081047]
        
    Example 1b - TracingPoint object:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> all_neurites_tortuosities(p1)
        [0.04134791479135339,
         0.05300604176745361,
         0.14956195330433844,
         0.15049238421642142,
         0.18919849587081047] 
    """
    out = []
    neurites = get_tip_nodes(n)
    while neurites:
        i = neurites.pop()
        out.append(neurite_tortuosity(i))
    out.sort()
    return out


def max_tortuosity(n):
    """Determines the 99.5 percentile of log(tortuosity) across all neurites in a neuron
    
    :param n: neuron
    :type n: :class:`Neuron`
    
    :returns: 99.5 percentile of log(tortuosity)
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> max_tortuosity(neuron)
        0.1884243736377227
    """
    out = all_neurites_tortuosities(n)
    return np.percentile(all_neurites_tortuosities(n), 99.5)


def median_tortuosity(n):
    """Determines the medial log(tortuosity) accross all neurites in a neuron
    
    :param n: neuron
    :type n: :class:`Neuron`
    
    :returns: median log(tortuosity)
    :rtype: `float` 
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> median_tortuosity(neuron)
        0.14956195330433844
    """
    out = all_neurites_tortuosities(n)
    return stat.median(out)


def partition_asymmetry(t):
    """Calculates the partition asymmetry according to:
       http://cng.gmu.edu:8080/Lm/help/Partition_asymmetry.htm
    :note: only calculates tree asymmetry for bifurcation points, not multifurcations
       
    :param t: beginning bifurcation point 
    :type t: :class:`TracingPoint`
    
    :returns: partition asymmetry calculation
    :returns: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0].children[0] #Accesses first branch point
        >>> partition_asymmetry(p1)
        0.3333333333333333
    """
    l = t.children[0]
    r = t.children[1]
    numL = len(get_tip_nodes(l))
    numR = len(get_tip_nodes(r))
    if numL != numR:
        return abs(numL - numR) / (numL + numR - 2)
    return 0


def get_bif_points_more_three_leaves(t):
    """Creates a deque with all multifurcation points that have more than 3 leaves
    
    :param t: starting branch point 
    :type t: :class:`TracingPoint`
    
    :returns: all multifurcation points that have more than 3 leaves
    :rtype: `deque` of :class:`TracingPoint`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> p1 = neuron.branches[0]
        >>> get_bif_points_more_three_leaves(p1)
        deque([TracingPoint(x=0.0, y=1.0, z=0.0, r=1.0, t=3, children=[TracingPoint(x=2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...}), TracingPoint(x=-2.0, y=2.0, z=1.0, r=1.0, t=3, children=[{ 2, truncated }], parent={...})], parent=TracingPoint(x=0.0, y=0.0, z=0.0, r=1.0, t=1, children=[{ 1, truncated }]))])
        >>> len(get_bif_points_more_three_leaves(p1))
        1
    """
    out = deque()
    nodes = deque()
    nodes.append(t)
    while nodes:
        i = nodes.pop()
        if len(get_tip_nodes(i)) > 3 and len(i.children) > 1:
            out.append(i)
        nodes.extend(i)
    return out


def tree_asymmetry(n):
    """Calculates the tree asymmetry value by taking the sum of (wp · PSAD(p)) on all branch nodes,
       where wp ∈ {0, 1} and equals 1 if the branch node's subtree has more than 3 leaves. PSAD is 
       defined as PSAD(p) = m / 2(m − 1)(n − m) * sum for each subtree of p of |r - n / m| with m 
       m defined as the number of subtrees (or children) of a p, n being the total number of terminal
       segments for p, and r being the degree (or number of tip nodes) for a subtree. 
    
    :param n: neuron
    :type n: :class:`Neuron`
    
    :returns: tree asymmetry value
    :rtype: `float`
    
    Example:
        >>> neuron = from_swc("Example1.swc")
        >>> tree_asymmetry(neuron)
        0.3333333333333333
    """
    q = deque(n)
    allPSADs = []
    while q:
        i = q.pop()
        b = get_bif_points_more_three_leaves(i)
        while b:
            p = b.pop()
            m = len(p.children)
            n = len(get_tip_nodes(p))
            sumSubTrees = 0
            for c in p.children:
                sumSubTrees += abs(len(get_tip_nodes(c)) - (n / m))
            psad = (m / (2 * (m - 1) * (n - m))) * sumSubTrees
            allPSADs.append(psad)
    if len(allPSADs) == 0:
        return 0
    return sum(allPSADs) / len(allPSADs)
