from collections import defaultdict, deque
import numpy as np
import math

from ngauge import __num_types__


class TracingPoint:
    """A class which defines a (X,Y,Z,R,T) tuple representing one point of a SWC file.

    :raises AttributeError: Raised on inproper typing of constructor variables
    :return: A new TracingPoint
    :rtype: TracingPoint
    """

    def __init__(self, x, y, z, r, t, children=None, fid=None, parent=None):
        """Create a TracingPoint"""

        if type(x) not in __num_types__:
            raise AttributeError(
                "X must be a numeric type ("
                + repr(__num_types__)
                + ") ["
                + str(x)
                + "]"
            )
        if type(y) not in __num_types__:
            raise AttributeError(
                "Y must be a numeric type ("
                + repr(__num_types__)
                + ") ["
                + str(y)
                + "]"
            )
        if type(z) not in __num_types__:
            raise AttributeError(
                "Z must be a numeric type ("
                + repr(__num_types__)
                + ") ["
                + str(z)
                + "]"
            )
        if type(r) not in __num_types__:
            raise AttributeError(
                "R must be a numeric type ("
                + repr(__num_types__)
                + ") ["
                + str(r)
                + "]"
            )
        if type(t) is not int:
            raise AttributeError("T must be a int [" + str(t) + "]")

        self.x = x
        """Stores the x coordinate of the TracingPoint"""
        self.y = y
        """Stores the y coordinate of the TracingPoint"""
        self.z = z
        """Stores the z coordinate of the TracingPoint"""
        self.r = r
        """Stores the radius of the TracingPoint"""
        self.t = t
        """Stores the type value of the TracingPoint"""

        self.file_id = fid
        """Stores the id used to load this TracingPoint from the SWC file"""

        self.parent = parent
        """Stores a symbolic link to the parent node of this TracingPoint"""

        self.children = []
        """Stores a list of all child nodes of this TracingPoint"""

        if children is None:
            self.children = []
        elif (
            type(children) is list
            and sum(1 for x in children if type(x) is not TracingPoint) == 0
        ):
            self.children = children
        else:
            raise TypeError("Children must be list of TracingPoint objects")

    def plot(self, ax=None, fig=None, color="blue", axis="z"):
        """Draws this TracingPoint as a figure

        :param ax: A matplotlib axis object to draw upon
        :type ax: matplotlib.pyplot.axis
        :param fig: A matplotlib figure object to draw upon
        :type fig: matplotlib.pyplot.figure

        :note: If no value for `ax` or `fig` is supplied, a new one will be generated

        :param color: The color to render the line drawing
        :type color: matplotlib color descriptor
        :param axis: The axis to perform the 3D to 2D projection upon
        :type axis: str

        :note: The returned matplotlib objects can be manipulated further to generate custom figures

        :return: a matplotlib `Figure` object which has been rendered.
        """
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import collections as mc

        if not ax and not fig:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")

        segments = []
        for node in self.select_nodes(lambda x: x.parent is not None):
            segments.append(
                (node.x, node.y, node.z, node.parent.x, node.parent.y, node.parent.z)
            )

        # for ix,iy,iz,jx,jy,jz in segments:
        #    ax.plot( [ix, jx], [iy, jy], color=color )
        lc = None
        if axis == "z":
            lc = mc.LineCollection(
                [[(ix, iy), (jx, jy)] for ix, iy, iz, jx, jy, jz in segments],
                color=color,
            )
        elif axis == "x":
            lc = mc.LineCollection(
                [[(iy, iz), (jy, jz)] for ix, iy, iz, jx, jy, jz in segments],
                color=color,
            )
        elif axis == "y":
            lc = mc.LineCollection(
                [[(ix, iz), (jx, jz)] for ix, iy, iz, jx, jy, jz in segments],
                color=color,
            )
        ax.add_collection(lc)
        ax.autoscale()

        return fig

    def add_child(self, toadd):
        """Add a child node to a given TracingPoint

        :param toadd: Node to add
        :type toadd: TracingPoint

        :raises AttributeError: Raised if :code:`type(toadd) != TracingPoint`

        :return: None"""
        if type(toadd) is not type(self):
            raise AttributeError("Can only add type TracingPoint to children")
        self.children.append(toadd)

    def __repr__(self, recurse=True):
        out = "TracingPoint("
        out += ", ".join(
            [
                "x=" + str(self.x),
                "y=" + str(self.y),
                "z=" + str(self.z),
                "r=" + str(self.r),
                "t=" + str(self.t),
            ]
        )
        if len(self.children) != 0:
            out += ", children=["

            if recurse:
                out += ", ".join([i.__repr__(recurse=False) for i in self])
            else:
                out += "{ %d, truncated }" % (len(self.children))
            out += "]"
        if self.parent is not None:
            out += ", parent="
            if recurse:
                out += self.parent.__repr__(recurse=False)
            else:
                out += "{...}"
        out += ")"

        return out

    # def __copy__(self):
    #    return self

    # def __setattr__(self, name, value):
    #    self.__dict__[name] = value

    def __iter__(self):
        return iter(self.children)

    def fix_parents(self):
        # TODO Document
        for node in self.select_nodes(lambda x: x.total_children() > 0):
            for child in node.children:
                child.parent = node

    def total_children(self):
        """
        :return: The count of all child nodes
        :rtype: `int`
        """
        return len(self.children)

    def total_child_nodes(self):
        """
        :return: The recursive count of all children-of-children
        :rtype: `int`
        """
        return len(self.get_all_nodes())

    def total_tip_nodes(self):
        """
        :return: The total number of tip nodes below this TracingPoint
        :rtype: `int`
        """
        return len(self.get_tip_nodes())

    def total_bif_nodes(self):
        """
        :return: The total number of bifurcation points below this TracingPoint
        :rtype: `int`
        """
        return len(self.get_bifurcation_nodes())

    def is_tip(self):
        """
        :returns: `True` if node is a tip
        :rtype: `bool`
        """
        return self.total_children() == 0

    def is_bif(self):
        """
        :returns: `True` if node is a bifurcation point
        :rtype: `bool`
        """
        return self.total_children() > 1

    def is_root(self):
        """
        :returns: `True` if node is a root (:code:`parent is None`)
        :rtype: `bool`
        """
        return self.parent is None

    def get_tip_nodes(self):
        """
        :returns: An array of TracingPoints representing the tip nodes
        :rtype: `list`
        """
        return self.select_nodes(lambda x: x.is_tip())

    def get_bifurcation_nodes(self):
        """
        :returns: An array of TracingPoints representing all bifurcation points
        :rtype: `list` of :class:`TracingPoint`
        """
        return self.select_nodes(lambda x: x.is_bif())

    def get_all_nodes(self):
        """
        :returns: An array of all nodes (inclusive) below this node
        :rtype: `list` of :class:`TracingPoint`
        """
        return self.select_nodes(lambda x: True)

    def get_path_to_root(self):
        """
        :returns: An array of nodes required to traverse to the root node (i.e. :code:`parent == None`)
        :rtype: `list` of :class:`TracingPoint`
        """
        out = [self]
        while out[-1].parent is not None:
            out.append(out[-1].parent)
        return out

    def path_dist_to_root(self):
        """
        :note: Uses result from :func:`get_path_to_root`
        :returns: The path distance between `self` and the root node
        :rtype: `float`
        """
        path = self.get_path_to_root()
        out = 0.0
        for i, j in zip(path, path[1:]):
            out += i.euclidean_dist(j)
        return out

    def get_all_segments(self):
        """
        :returns: The start and end point of each not-bifurcated segment
        :note: By definition, all points will be either a bifurcation or leaf points
        :rtype: `set` of :class:`TracingPoint`
        """
        out = set()
        already_done = set()
        q = deque(self.get_tip_nodes())
        while q:
            i = q.pop()
            if i is None:
                continue
            nn = i.next_bif_point()
            if nn == i:
                continue
            if nn not in already_done:
                already_done.add(nn)
                q.extend([nn])
            out.add((nn, i))
        return list(out)

    def select_nodes(self, select_func):
        """
        :parameter select_func: A function to perform the selection from
        :type select_func: `function`
        :returns: All nodes `x` (inclusive) for which :code:`select_func(x)` evaluates to a truthy value
        :rtype: `list` of :class:`TracingPoint`
        """
        # traverse through all nodes and return ones where select_func(x)==True
        q = deque([self])
        out = []
        while q:
            i = q.pop()
            q.extend(i)
            if select_func(i):
                out.append(i)
        return out

    def total_height(self):
        """
        :returns: the height of the smallest box which encloses the branch
        :rtype: `numeric`, inherited from :attr:`self.y`
        """
        min_y, max_y = self.y, self.y
        q = deque([self])
        while q:
            i = q.pop()
            min_y = min(i.y, min_y)
            max_y = max(i.y, max_y)
            q.extend(iter(i))
        return max_y - min_y

    def total_width(self):
        """
        :returns: the width of the smallest box which encloses the branch
        :rtype: `numeric`, inherited from :attr:`self.x`
        """
        min_x, max_x = self.x, self.x
        q = deque([self])
        while q:
            i = q.pop()
            min_x = min(i.x, min_x)
            max_x = max(i.x, max_x)
            q.extend(iter(i))
        return max_x - min_x

    def total_depth(self):
        """
        :returns: the depth of the smallest box which encloses the branch
        :rtype: `numeric`, inherited from :attr:`self.z`
        """
        min_z, max_z = self.z, self.z
        q = deque([self])
        while q:
            i = q.pop()
            min_z = min(i.z, min_z)
            max_z = max(i.z, max_z)
            q.extend(iter(i))
        return max_z - min_z

    def total_volume(self):
        """
        :returns: the volume of the smallest box which encloses the branch
        :rtype: `numeric`, inherited from :func:`total_width`, :func:`total_height`, and :func:`total_depth`
        """
        return self.total_width() * self.total_height() * self.total_depth()

    def euclidean_dist(self, other):
        """
        :param other: A second node
        :type other: :class:`TracingPoint`
        :returns: The euclidean distance to a node :attr:`other`
        :rtype: `float`
        """
        return (
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        ) ** 0.5

    def path_dist_to_child(self, other):
        """
        :returns: The path distance required to traverse the tree from :attr:`self` to :attr:`other`
        :rtype: `float`
        """
        out = 0.0
        while other is not self:
            out += other.euclidean_dist(other.parent)
            other = other.parent
        return out

    def euclidean_to_ends(self):
        """
        :returns: The euclidean distance to each node as defined in :func:`get_tip_nodes`
        :rtype: `list` of `float`
        """
        return [self.euclidean_dist(x) for x in self.get_tip_nodes()]

    def farthest_tip(self):
        """
        TODO
        """
        return max(self.get_tip_nodes(), key=self.euclidean_dist)

    def path_dist_to_ends(self):
        """
        :returns: The path distance to each node defined in :func:`get_tip_nodes`
        :rtype: `list` of `float`
        """
        return [self.path_dist_to_child(x) for x in self.get_tip_nodes()]

    def next_bif_point(self):
        """
        :returns: The next node in the tree, which is a bifurcation
        :note: Returns the root of the tree if there are no bifurcation nodes
        :note: Returns `self` if no parent node is present
        :rtype: :class:`TracingPoint`
        """
        if self.parent is None:
            return self
        out = self.parent
        while not out.is_bif():
            if out.parent is None:
                break
            out = out.parent
        return out

    def branching_order(self):
        """
        :returns: The branching order of this node
        :rtype: `int`
        """
        out = 0
        other = self.parent
        while other is not None:
            if other.total_children() > 1:
                out += 1
            other = other.parent
        return out

    def partition_asymmetry(self):
        """
        :note: Formula: :code:`abs(n1-n2)/(n1+n2-2)` (R. Scorcioni, et al. 2008)
        :note: Returns 0 if n1 == n2
        :returns: The partition asymmetry of this node
        :rtype: `float`
        """
        if not self.is_bif():  # forces children > 1
            raise ValueError(
                "Partition Asymmetry can not be calculated on a point which is not a bifurcation point"
            )
        if self.total_children() > 2:
            raise ValueError(
                "Partition Asymmetry can not be calculated for a node with more than two children"
            )

        l, r = self.children  # unpack array which should be 2-tuple
        l, r = l.total_tip_nodes(), r.total_tip_nodes()
        if l == r:
            return 0.0
        return float(abs(l - r)) / (l + r - 2)

    def neurite_tortuosity(self):
        """Determine the log(tortuosity) of a neurite - tortuosity defined as the ratio between the path
           length (each segment's length combined) and the Euclidean distance from the root and tip node

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
        t = self
        pathLength = 0
        while t.parent:
            pathLength += t.euclidean_dist(t.parent)
            t = t.parent
        euclideanStartToEnd = euclidean_dist(t, self)
        return math.log10(pathLength / euclideanStartToEnd)

    def angle(self, a, b):
        """Determines the angle between two paths in degrees between [0, 180]

        :param self: point connected to both a and b
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
            >>> ba.angle(p1, p2)
            90.0

        Example 2 - TracingPoints in a Neuron object:
            >>> neuron = from_swc("Example1.swc")
            >>> ba = neuron.branches[0].children[0]
            >>> p1 = ba.children[0]
            >>> p2 = ba.children[1]
            >>> ba.angle(p1, p2)
            109.47122063449069
        """
        ax, ay, az = a.x - branchPoint.x, a.y - branchPoint.y, a.z - branchPoint.z
        bx, by, bz = b.x - branchPoint.x, b.y - branchPoint.y, b.z - branchPoint.z
        magA = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5
        magB = (bx ** 2 + by ** 2 + bz ** 2) ** 0.5
        if magA == 0 or magB == 0:
            return 180

        dotProduct = (ax * bx) + (ay * by) + (az * bz)
        rad = dotProduct / (magA * magB)
        rad = min(rad, 1)
        rad = max(rad, -1)
        return m.acos(rad) * (180 / math.pi)

    '''def sackin_index(self):
        """
        Calculates the Sackin index of a given branch

        See: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224197"""
        tips = self.get_tip_nodes()
        out, l = 0.0, len(tips)
        n = ((2.483) ** (l)) * ((l) ** (-3.0 / 2.0))
        for tip in tips:
            out += sum(1 for i in tip.get_path_to_root() if i.is_bif())
        return out / n'''

    '''
    def colless_index(self):
        """
        Calculates the Sackin index of a given branch

        See: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224197"""
        out, l = 0.0, len(self.get_tip_nodes())
        n = ((2.483) ** (l)) * ((l) ** (-3.0 / 2.0))
        for bif in self.get_bifurcation_nodes():
            out += abs(
                bif.children[0].total_tip_nodes() - bif.children[1].total_tip_nodes()
            )
        return (2.0 * out) / ((n - 1) * (n - 2))
    '''

    @staticmethod
    def slice_surface_area(points):
        """
        :param points: A set of points which form an arbitrary polygon
        :type points: `list` of :class:`TracingPoint`
        :note: All points must be in the same z-plane (all :attr:`z` attributes are equal)
        :note: Returns 0.0 if :code:`len(points) < 3`
        :returns: The surface area bound by :attr:`points`
        :rtype: `float`
        """
        if len(points) <= 2:
            return 0.0
        for i in points:
            if i.z != points[0].z:
                raise ValueError("Points are not in same z-plane")

        # implements this algorithm: https://www.mathopenref.com/coordpolygonarea.html
        total = 0.0
        for i, j in zip(points, points[1:] + points[:1]):
            total += (i.x * j.y) - (i.y * j.x)
        total = abs(total)
        return total / 2.0

    @staticmethod
    def slice_perimeter(points):
        """
        :param points: A set of points which form an arbitrary polygon
        :type points: `list` of :class:`TracingPoint`
        :note: All points must be in the same z-plane (all :attr:`z` attributes are equal)
        :note: Returns 0.0 if :code:`len(points) < 2`
        :returns: The perimeter defined by :attr:`points`
        :rtype: `float`
        """
        if len(points) <= 1:
            return 0.0
        for i in points:
            if i.z != points[0].z:
                raise ValueError("Points are not in same z-plane")

        return sum(i.euclidean_dist(j) for i, j in zip(points, points[1:] + points[:1]))

    def to_swc(self, fname=None):
        from collections import deque as queue

        self.fix_parents()

        i = 1  # counter of swc line
        todo = queue([self])
        memory = {None: -1}  # track line numbers of each element
        out = []

        # tail = self.farthest_tip()
        #        while tail is not self:
        # todo.appendleft( tail )
        #           tail = tail.parent

        while todo:
            a = todo.pop()
            if a in memory:  # duplicate
                continue
            todo.extend(a.children)

            parent = memory[a.parent]
            memory[a] = i

            # Columns:  id  t  x  y  z  r  pid
            out.append("%d %d %g %g %g %g %d" % (i, a.t, a.x, a.y, a.z, a.r, parent))
            i += 1

        out = "\n".join(out)
        if fname is None:
            return out
        f = open(fname, "w")
        f.write(out)
        f.close()
