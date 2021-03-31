from collections import defaultdict, deque
from src.morpho import TracingPoint as t
from src.morpho import Neuron as n
import SWCParser as parse

# definitions of functions are from the morphometrics statistics section of https://link.springer.com/content/pdf/10.1007/s12021-020-09461-z.pdf


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
