import math
import statistics

from collections import defaultdict, deque, Counter
import numpy as np

from ngauge import __num_types__
from ngauge import TracingPoint

from shapely.geometry import LineString, Point, MultiLineString


class Neuron:
    """A class representing a Neuron, i.e., an object represented by a complete SWC file."""

    def __init__(self):
        self.soma_layers = defaultdict(lambda: [])
        """A `dict` of `lists` of :class:`TracingPoint` which define the layers of a volumetrically-traced soma"""

        self.branches = []
        """A `list` of :class:`TracingPoint` which are the roots of branches in this :class:`Neuron`"""

        self.metadata = ""
        """An empty attribute to store metadata from the SWC file"""

    def plot(self, fig=None, ax=None, axis="z", color="blue"):
        """Draws this Neuron as a figure

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

        # mpl.use("Agg")
        import matplotlib.pyplot as plt

        if not ax and not fig:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        for layer in self.soma_layers.values():
            if axis == "z":
                ax.plot([i.x for i in layer], [i.y for i in layer], color=color)
            elif axis == "x":
                ax.plot([i.y for i in layer], [i.z for i in layer], color=color)
            elif axis == "y":
                ax.plot([i.x for i in layer], [i.z for i in layer], color=color)

        for branch in self.branches:
            branch.plot(ax=ax, fig=fig, axis=axis, color=color)

        return fig

    def fix_parents(self):
        for child in self.branches:
            child.fix_parents()

    def add_branch(self, toadd):
        """
        :param toadd: A point to add as a new branch to this :class:`Neuron`
        :type toadd: :class:`TracingPoint`

        :rtype: `None`
        """
        self.branches.append(toadd)

    def add_soma_points(self, toadd):
        """
        :param toadd: An array of soma points to add to this :class:`Neuron`
        :type toadd: `list` of :class:`TracingPoint`

        :raises AttributeError: Raised if :attr:`x`, :attr:`y`, :attr:`z`, :attr:`r` are not numeric

        :rtype: `None`
        """
        # assumes toadd is [ (x,y,z,r), ... ]
        for x, y, z, r in toadd:
            if type(x) not in __num_types__:
                raise AttributeError(
                    "X must be a numeric type (" + repr(__num_types__) + ") [" + x + "]"
                )
            if type(y) not in __num_types__:
                raise AttributeError(
                    "Y must be a numeric type (" + repr(__num_types__) + ") [" + y + "]"
                )
            if type(z) not in __num_types__:
                raise AttributeError(
                    "Z must be a numeric type (" + repr(__num_types__) + ") [" + z + "]"
                )
            if type(r) not in __num_types__:
                raise AttributeError(
                    "R must be a numeric type (" + repr(__num_types__) + ") [" + r + "]"
                )

            self.soma_layers[z].append(TracingPoint(x=x, y=y, z=z, r=r, t=1))

    def soma_centroid(self):
        """
        :returns: The location of the centroid, found from the average of all soma coordinates
        :rtype: `tuple` of `float` (x, y, z)
        """
        avgs, n = [0, 0, 0], 0  # xyz
        for key, pts in self.soma_layers.items():
            for pt in pts:
                avgs[0] += pt.x
                avgs[1] += pt.y
                avgs[2] += pt.z
                n += 1
        return tuple(i / n for i in avgs)

    def total_root_branches(self):
        """
        :returns: The total number of branch roots contained in this :class:`Neuron`
        :rtype: `int`
        """
        return len(self.branches)

    def total_child_nodes(self):
        """
        :returns: The total number of child nodes contained in this :class:`Neuron`
        :rtype: `int`
        """
        return sum(x.total_child_nodes() for x in self.branches)

    def total_tip_nodes(self):
        """
        :returns: The total number of child nodes contained in this :class:`Neuron`
        :rtype: `int`
        """
        return sum(x.total_tip_nodes() for x in self.branches)

    def total_bif_nodes(self):
        """
        :returns: The total number of bifurcation nodes contained in this :class:`Neuron`
        :rtype: `int`
        """
        return sum(x.total_bif_nodes() for x in self.branches)

    def total_branch_nodes(self):
        """
        Redirects to `total_bif_nodes`
        """
        return self.total_bif_nodes()

    def total_width(self, percentile=None):
        """
        :returns: The width of the smallest bounding box required to encapsulate this :class:`Neuron`
        :rtype: `numeric`, inherited from :attr:`x`
        """
        min_x, max_x = None, None
        for z, layer in self.soma_layers.items():
            for pt in layer:
                if min_x is None:
                    min_x = pt.x
                    max_x = pt.x
                else:
                    min_x = min(pt.x, min_x)
                    max_x = max(pt.x, max_x)
        for branch in self.branches:
            q = deque([branch])
            while q:
                i = q.pop()
                if min_x is None:  # needed in case no soma points
                    min_x = i.x
                    max_x = i.x
                min_x = min(i.x, min_x)
                max_x = max(i.x, max_x)
                q.extend(iter(i))
        return max_x - min_x

    def total_height(self):
        """
        :returns: The height of the smallest bounding box required to encapsulate this :class:`Neuron`
        :rtype: `numeric`, inherited from :attr:`y`
        """
        min_y, max_y = None, None
        for z, layer in self.soma_layers.items():
            for pt in layer:
                if min_y is None:
                    min_y = pt.y
                    max_y = pt.y
                else:
                    min_y = min(pt.y, min_y)
                    max_y = max(pt.y, max_y)
        for branch in self.branches:
            q = deque([branch])
            while q:
                i = q.pop()
                if min_y is None:  # needed in case no soma points
                    min_y = i.x
                    max_y = i.x
                min_y = min(i.y, min_y)
                max_y = max(i.y, max_y)
                q.extend(iter(i))
        return max_y - min_y

    def total_depth(self):
        """
        :returns: The depth of the smallest bounding box required to encapsulate this :class:`Neuron`
        :rtype: `numeric`, inherited from :attr:`z`
        """
        min_z, max_z = None, None
        for z, layer in self.soma_layers.items():
            if min_z is None:
                min_z = z
                max_z = z
            else:
                min_z = min(z, min_z)
                max_z = max(z, max_z)
        for branch in self.branches:
            q = deque([branch])
            while q:
                i = q.pop()
                if min_z is None:  # needed in case no soma points
                    min_z = i.x
                    max_z = i.x
                min_z = min(i.z, min_z)
                max_z = max(i.z, max_z)
                q.extend(iter(i))
        return max_z - min_z

    def slice_surface_areas(self):
        """
        :returns: The surface area of each soma slice in this :class:`Neuron`
        :rtype: `dict` mapping z-values to `float`
        """
        return {
            z: TracingPoint.slice_surface_area(pts)
            for z, pts in self.soma_layers.items()
        }

    def slice_perimeters(self):
        """
        :returns: The perimeter of each soma slice in this :class:`Neuron`
        :rtype: `dict` mapping z-values to `float`
        """
        return {
            z: TracingPoint.slice_perimeter(pts) for z, pts in self.soma_layers.items()
        }

    def soma_volume(self):
        """
        :returns: The total volume of this :class:`Neuron` soma by approximating each slice as a extruded polygon
        :note: This may yield low accuracy if there are few Z-slices traced
        :rtype: `dict` mapping z-values to `float`
        """
        ssa = self.slice_surface_areas()
        total, z_height = 0.0, 0.0

        z_indices = list(sorted(ssa.keys()))
        if len(z_indices) <= 1:
            raise ValueError("Can not calculate volume of 1-slice soma")

        for i, z in enumerate(z_indices):
            z_height = 0
            if i == 0:  # first iteration
                z_above = z_indices[i + 1]
                z_height = abs(z_above - z) / 2
            elif i == len(z_indices) - 1:  # last iteration
                z_below = z_indices[i - 1]
                z_height = abs(z - z_below) / 2
            else:
                z_below, z_above = z_indices[i - 1], z_indices[i + 1]
                z_height = abs(z_above - z_below) / 2
            total += ssa[z] * z_height

        return total

    def soma_surface_area(self):
        """
        :raises ValueError: If the soma has 0 or 1 layers currently registered
        :returns: The total surface area produced by the same model as is used in :func:`soma_volume`
        :note: This may yield low accuracy if there are few Z-slices traced or if the tracing is incomplete
        :rtype: `float`
        """
        sp = self.slice_perimeters()
        ssa = self.slice_surface_areas()
        total, z_height = 0.0, 0.0

        z_indices = list(sorted(sp.keys()))
        if len(z_indices) <= 1:
            raise ValueError("Can not calculate volume of 1-slice soma")

        for i, z in enumerate(z_indices):
            if i == 0 or i == len(z_indices) - 1:  # first or last iteration
                total += ssa[z]  # top/bottom area

            z_height = 0
            if i == 0:  # first iteration
                z_above = z_indices[i + 1]
                z_height = abs(z_above - z) / 2
            elif i == len(z_indices) - 1:  # last iteration
                z_below = z_indices[i - 1]
                z_height = abs(z - z_below) / 2
            else:
                z_below, z_above = z_indices[i - 1], z_indices[i + 1]
                z_height = abs(z_above - z_below) / 2
            total += sp[z] * z_height

        return total

    def total_volume(self):
        """
        :returns: The volume of the smallest bounding box required to encapsulate this :class:`Neuron`
        :rtype: `float`
        """
        return self.total_width() * self.total_height() * self.total_depth()

    def max_partition_asymmetry(self, within=1):
        """
        :param within: The minimum branching order to include
        :type within: `int`
        :returns: The maximum partition asymmetry within a given number of nodes
        :note: Returns `None` if there are no branches registered in this :class:`Neuron`
        :rtype: `float`
        """
        max_pa = None
        for branch in self.branches:
            for bif_node in branch.get_bifurcation_nodes():
                if bif_node.branching_order() >= within:
                    continue
                if max_pa is None:
                    max_pa = bif_node.partition_asymmetry()
                else:
                    max_pa = max(max_pa, bif_node.partition_asymmetry())

        return max_pa

    def min_partition_asymmetry(self, within=1):
        """
        :param within: The minimum branching order to include
        :type within: `int`
        :returns: The minimum partition asymmetry within a given number of nodes
        :note: Returns `None` if there are no branches registered in this :class:`Neuron`
        :rtype: `float`
        """
        min_pa = None
        for branch in self.branches:
            for bif_node in branch.get_bifurcation_nodes():
                if bif_node.branching_order() >= within:
                    continue
                if min_pa is None:
                    min_pa = bif_node.partition_asymmetry()
                else:
                    min_pa = min(min_pa, bif_node.partition_asymmetry())
        return min_pa

    def max_branching_order(self):
        """
        :returns: The maximum branch order achieved by any of the child nodes
        :rtype: `int`
        """
        out = 0
        for branch in self.branches:
            for node in branch.get_tip_nodes():
                out = max(out, node.branching_order())
        return out

    def get_main_branch(self):
        """
        :returns: The longest branch from the :class:`Neuron`
        :note: Returns `None` if there are no branches in this :class:`Neuron`
        :rtype: `int`
        """
        out, m = None, 0
        for branch in self.branches:
            if branch.total_child_nodes() > m:
                m = branch.total_child_nodes()
                out = branch
        return out

    def all_segment_lengths(self):
        """Creates sorted list of all segment lengths in a neuron`

        :returns: all segment lengths sorted by least to greatest distance
        :rtype: `list(float)`

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
        q = deque(self)
        while q:
            i = q.pop()
            branchChildren = deque(i)
            while branchChildren:
                n = branchChildren.pop()
                out.append(n.euclidean_dist(n.parent))
                q.append(n)
        out.sort()
        return out

    def all_path_angles(self):
        """Calculates the path angle for all nodes in a neuron, starting at the node after the root node.

        :returns: all the path angles in a neuron
        :rtype: `list` of `float`
        :note: branch points, as there are then multiple new branches diverging,
                will have multiple path angles angles added to returned list

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.all_path_angles()
               [114.0948425521107,
                 114.0948425521107,
                 82.17876449350037,
                 162.28452765633213,
                 65.9051574478893,
                 132.87598866663572,
                 88.65276496096037,
                 172.50550517788332]
        """
        q = deque(self)
        p = deque()
        out = []
        while q:
            i = q.pop()
            p.extend(i)
        while p:
            i = p.pop()
            if len(i.children) > 1:
                for j in range(len(i.children)):
                    out.append(i.angle(i.parent, i.children[j]))
                    p.append(i.children[j])
            elif i.children:
                out.append(i.angle(i.parent, i.children[0]))
                p.append(i.children[0])
        return out

    def median_path_angle(self):
        """Determines the median path angle across a neuron in degrees between [0, 180]

        :returns: median path angle
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.median_path_angle()
            114.0948425521107
        """
        out = self.all_path_angles()
        out.sort()
        return statistics.median(out)

    def max_path_angle(self):
        """Determines the maximal path angle across a neuron in degrees between [0, 180]

        :returns: maximal path angle
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.max_path_angle()
            172.50550517788332
        """
        out = self.all_path_angles()
        return max(out)

    def all_branch_angles(n):
        """Creates a list with angles of all the branch points in a neuron in degrees between [0, 180]

        :param n: starting point for gathering branch point angles
        :type n: either :class:`Neuron` or :class:`TracingPoint`

        :returns: all branch angles
        :rtype: `list`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.all_branch_angles()
            [90.8386644299175, 83.62062979155719, 83.62062979155719, 109.47122063449069]
        """
        q = get_branch_points(self)
        out = []
        while q:
            i = q.pop()
            a = deque(i)
            s1 = a.pop()
            s2 = a.pop()
            out.append(i.angle(s1, s2))
            while a:
                s1 = a.pop()
                if a:
                    s2 = a.pop()
                out.append(i.angle(s1, s2))
        return out

    def min_branch_angle(self):
        """Determines the minimal branch point angle in degrees between [0, 180]

        :returns: minimal branch angle
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.min_branch_angle()
            83.62062979155719
        """
        out = self.all_branch_angles()
        return min(out)

    def avg_branch_angle(self):
        """Determines the average branch point angle in degrees between [0, 180]

        :returns: average branch angle
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.avg_branch_angle()
            91.88778616188064
        """
        out = self.all_branch_angles()
        return sum(out) / len(out)

    def max_branch_angle(n):
        """Determines the maximal branch point angle in degrees between [0, 180]

        :returns: maximal branch angle
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.max_branch_angle()
            109.47122063449069
        """
        out = self.all_branch_angles()
        return max(out)

    def __repr__(self):
        return "{Neuron of %d branches and %d soma layers}" % (
            len(self.branches),
            len(self.soma_layers),
        )

    def __iter__(self):
        return iter(self.branches)

    # def sackin_index(self):
    #    return self.get_main_branch().sackin_index()

    # def colless_index(self):
    #    return self.get_main_branch().colless_index()

    def arbor_dist(self):
        """
        :returns: The distance from the :func:`soma_centroid` to the positional average of all :attr:`branch` :class:`TracingPoints`
        :rtype: `float`
        """
        x, y, z, n = 0.0, 0.0, 0.0, 0
        for branch in self.branches:
            for node in branch.get_tip_nodes():
                x, y, z = x + node.x, y + node.y, z + node.z
                n += 1
        x, y, z = x / n, y / n, z / n
        sx, sy, sz = self.soma_centroid()
        return ((x - sx) ** 2 + (y - sy) ** 2 + (z + sz) ** 2) ** 0.5

    def persistence_diagram(self, f=lambda x: x.path_dist_to_root()):
        """
        :param f: The distance function to calculate a persistance diagram
        :type f: `callable(` :class:`TracingPoint` `)`
        :returns: A persistance diagram
        :rtype: `tuple` of `list`, representing the X and Y axis
        """
        out = [[], []]  # birth, death
        for branch in self.branches:
            for segment_start, segment_end in branch.get_all_segments():
                out[0].append(f(segment_start))
                out[1].append(f(segment_end))
        return out

    def iter_all_points(self, exclude_soma=False):
        """
        :param exclude_soma: Select if soma :class:`TracingPoint` objects should be included
        :type exclude_soma: `bool`
        :returns: An iteration of all :class:`TracingPoint` in this :class:`Neuron`
        :rtype: `iterable` of `TracingPoint`
        """
        for branch in self.branches:
            for pt in branch.get_all_nodes():
                yield pt
        if not exclude_soma:
            for layer in self.soma_layers.values():
                for pt in layer:
                    yield pt

    def rotate(self, x=0, y=0, z=0):
        """
        Rotates all points within this :class:`Neuron`

        :param x: The rotation angle about the x axis
        :type x: `numeric`
        :param y: The rotation angle about the y axis
        :type y: `numeric`
        :param z: The rotaion angle about the z axis
        :type z: `numeric`

        :returns: Nothing
        :rtype: `None`
        """
        import numpy as np

        rot = rotation_matrix(z, y, x)
        for pt in self.iter_all_points():
            coords = rot.dot(np.array([pt.x, pt.y, pt.z]))
            pt.x = float(coords[0])
            pt.y = float(coords[1])
            pt.z = float(coords[2])

    def translate(self, dx=0, dy=0, dz=0):
        """
        Translates all points within this :class:`Neuron`

        :param dx: The translation along the x axis
        :type dx: `numeric`
        :param dy: The translation along the y axis
        :type dy: `numeric`
        :param dz: The translation along the z axis
        :type dz: `numeric`

        :returns: Nothing
        :rtype: `None`
        """
        for pt in self.iter_all_points():
            pt.x += dx
            pt.y += dy
            pt.z += dz

    def scale(self, dx=1, dy=1, dz=1):
        """
        Scales the coordinates of all points within this :class:`Neuron`

        :param dx: The scale along the x axis
        :type dx: `numeric`
        :param dy: The scale along the y axis
        :type dy: `numeric`
        :param dz: The scale along the z axis
        :type dz: `numeric`

        :returns: Nothing
        :rtype: `None`
        """
        for pt in self.iter_all_points():
            pt.x *= dx
            pt.y *= dy
            pt.z *= dz

    def center_soma(self):
        """
        Performs a :func:`translate` such that the soma is located at the origin :code:`(0,0,0)`

        :returns: Nothing
        :rtype: None
        """
        soma_x, soma_y, soma_z = self.soma_centroid()
        self.translate(dx=-1 * soma_x, dy=-1 * soma_y, dz=-1 * soma_z)

    def average_thickness(self):
        """Determines average radius in microns across all neurites

        :returns: average radius in microns
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> avg_thickness(neuron)
            1.0
        """

        count = 0
        total = 0.0
        q = deque(self.branches)
        while q:
            i = q.pop()
            count += 1
            total += i.r
            q.extend(i)
        return float(total) / float(count)

    def all_neurites_tortuosities(self):
        """Creates a sorted list of the log(tortuosity) values of all the neurites in the input

        :returns: all log(tortuosity) values for neurites sorted least to greatest
        :rtype: `list`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> all_neurites_tortuosities(neuron)
            [0.04134791479135339,
             0.05300604176745361,
             0.14956195330433844,
             0.15049238421642142,
             0.18919849587081047]
        """
        return sorted([i.neurite_tortuosity() for i in self.get_tip_nodes()])

    def max_tortuosity(self):
        """Determines the 99.5 percentile of log(tortuosity) across all neurites in a neuron


        :returns: 99.5 percentile of log(tortuosity)
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> max_tortuosity(neuron)
            0.1884243736377227
        """
        return np.percentile(self.all_neurites_tortuosities(), 99.5)

    def median_tortuosity(self):
        """Determines the medial log(tortuosity) accross all neurites in a neuron

        :returns: median log(tortuosity)
        :rtype: `float`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> median_tortuosity(neuron)
            0.14956195330433844
        """
        return statistics.median(self.all_neurites_tortuosities())

    def branch_angles_histogram(self, bins=20):
        """Creates a histogram (an array of counts and an array of edges) of all branch angles with
           default of 20 bins between [0, 180] degrees

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: histogram of all branch angles
        :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.branch_angles_histogram()
            (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
             array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                    99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
        """
        return np.histogram(self.all_branch_angles(), bins=bins, range=(0, 180))

    def branch_order_counts(self):
        """Creates list from 0, K with K being the max branch order in the neuron and each i in the
                list being the number of bifurcation points with that branch order

        :returns: number of bifurcation points for each branch order value
        :rtype: `list` of `int`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.branch_order_counts()
            [0, 0, 4]
        """
        out = Counter([x.branching_order() for x in self.get_branch_points()])
        return [out[i] for i in range(max(out.keys()) + 1)]

    def path_angles_histogram(self, bins=20):
        """Creates a histogram (an array of counts and an array of edges) of all path angles with
           default of 20 bins between [0, 180] degrees

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: histogram of all path angles
        :rtype: `tuple` two `numpy.array`, one with counts and one with edge values

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.path_angles_histogram()
            (array([0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 1, 1]),
             array([  0.,   9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.,  90.,
                    99., 108., 117., 126., 135., 144., 153., 162., 171., 180.]))
        """
        return np.histogram(self.all_path_angles(), bins=bins, range=(0, 180))

    def segment_length_histogram(self, bins=20):
        """Creates a histogram (an array of counts and an array of edges) of all euclidean segment lengths
           with default of 20 bins between 0 and maximum segment length

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: histogram of all segment lengths
        :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.segment_length_histogram()
            (array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 2]),
             array([0.        , 0.12247449, 0.24494897, 0.36742346, 0.48989795,
                    0.61237244, 0.73484692, 0.85732141, 0.9797959 , 1.10227038,
                    1.22474487, 1.34721936, 1.46969385, 1.59216833, 1.71464282,
                    1.83711731, 1.95959179, 2.08206628, 2.20454077, 2.32701526,
                    2.44948974]))
        """
        out = self.all_segment_lengths()
        return np.histogram(out, bins=bins, range=(0, out[-1]))

    def thickness_histogram(self, bins=30):
        """Creates a histogram (an array of counts and an array of edges) of all nodes' radii, soma
           excluded, with default of 30 bins between 0 and maximum radii

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: histogram of all thicknesses
        :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.thickness()
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
        q = deque(self)
        out = []
        while q:
            i = q.pop()
            out.append(i.r)
            q.extend(i)
        out.sort()
        return np.histogram(out, bins=bins, range=(0, out[-1]))

    def path_distances_to_soma_histogram(self, bins=20):
        """Creates a histogram (an array of counts and an array of edges) of the path length of
           each branch point and tip to the soma with default of 20 bins between 0 and maximum length

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: histogram of all path distances
        :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.path_distance_to_soma_histogram()
            (array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 2]),
             array([0.        , 0.31975865, 0.6395173 , 0.95927595, 1.27903459,
                    1.59879324, 1.91855189, 2.23831054, 2.55806919, 2.87782784,
                    3.19758649, 3.51734513, 3.83710378, 4.15686243, 4.47662108,
                    4.79637973, 5.11613838, 5.43589703, 5.75565568, 6.07541432,
                    6.39517297]))
        """
        out = [i.path_dist_to_root() for i in self.iter_all_points(exclude_soma=True)]
        return np.histogram(out, bins=bins, range=(0, out[-1]))

    def euclidean_distances_to_soma_histogram(self, bins=20):
        """Creates a histogram (an array of counts and an array of edges) of the Euclidean distance
        of each branch point and tip to the soma with default of 20 bins between 0 and maximum length

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: histogram of euclidean distances
        :rtype: `tuple` of two `numpy.array`, one with counts and one with edge values

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.euclidean_distances_to_soma()
            (array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1]),
             array([0.   , 0.225, 0.45 , 0.675, 0.9  , 1.125, 1.35 , 1.575, 1.8  ,
                    2.025, 2.25 , 2.475, 2.7  , 2.925, 3.15 , 3.375, 3.6  , 3.825,
                    4.05 , 4.275, 4.5  ]))
        """
        out = [
            i.euclidean_distances_to_soma()
            for i in self.iter_all_points(exclude_soma=True)
        ]
        return np.histogram(out, bins=bins, range=(0, out[-1]))

    def sholl_intersection(self, steps=36, proj="xy"):
        """Creates a numpy array for the Sholl Intersection, which is all intersections at
           different radii a certain number of steps away from center of input

        :param steps: number of steps desired between center and maximal neurite point
        :type steps: `int`

        :param proj: which plane circle is located on
        :type: `string`

        :returns: `numpy.array` consisting of `tuples` with radii position (`float`) and
                  number of intersections (`int`)
        :rtype: `numpy.ndarray`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.sholl_intersection()
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
        """
        # determine center based on input
        center = self.soma_centroid()
        if proj == "xy":
            center = Point(center[0], center[1])
        elif proj == "yz":
            center = Point(center[1], center[2])
        elif proj == "xz":
            center = Point(center[0], center[2])
        raise AttributeError("proj must be either xy, yz, or xz")

        # fills lines with all the segments in our neuron or tracing point
        out = self.get_tip_nodes(a)
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

    def all_branch_orders(self):
        """Creates a list with all the branch orders of all bifurcation points in neuron

        :returns: `list` with all branch orders
        :rtype: `list` of `int`

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.all_branch_orders()
                [2, 2, 2, 2]
        """
        q = self.get_branch_points()
        out = []
        while q:
            i = q.pop()
            out.append(len(i.children))
        return out

    def branch_angles_x_branch_orders(self, bins=20):
        """Creates a 2D histogram of branch angles as a function of branch orders (across all branch
                points) with default bins of 20 and range 0 to max branch order by 0 to 180 degrees

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: 2D histogram of branch angles as a function of branch orders
        :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.branch_angles_x_branch_orders()
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
        orders = self.all_branch_orders()
        angles = self.all_branch_angles()
        return np.histogram2d(
            orders, angles, bins=bins, range=[[0, ms.max_branch_order(n)], [0, 180]]
        )

    def branch_angles_x_path_distances(self, bins=20):
        """Creates a 2D histogram of branch angles as a function of path distances to the soma in
                microns (across all branch points) with default bins of 20

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: 2D histogram of branch angles as a function of path distances
        :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.branch_angles_x_path_distances()
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
        angles = self.all_branch_angles()
        q = self.get_branch_points()
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

    def path_angle_x_branch_order(self, bins=20):
        """Creates a 2D histogram of path angles as a function of branch orders (across all nodes)
           with default bins of 20

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: 2D histogram of path angles as a function of branch orders
        :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges
        :note: bifurcation nodes will have multiple values in histogram associated with them due
               to multiple path angles

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.path_angle_x_branch_order()
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
        q = deque(self)
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
        return np.histogram2d( angles, orders, bins=bins, range=[[0, 180], [0, max(orders)]] )

    def path_angle_x_path_distance(self, bins=20):
        """Creates a 2D histogram of path angles as a function of path distances to the soma in microns
           (across all nodes) with default bins of 20

        :param bins: number of bins for histogram to have
        :type bins: `int`

        :returns: 2D histogram of path angles as a function of path distances
        :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges
        :note: bifurcation nodes will have multiple values in histogram associated with them due
               to multiple path angles

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.path_angle_x_path_distance()
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
        q = deque(self)
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
        return np.histogram2d(dist, angles, bins=bins, range=[[0, max(sortDist)], [0, 180]])

    def thickness_x_branch_order(self, bins=20):
        """Creates 2D histogram of neurite radii as a function of branch orders (across all nodes)

        :returns: 2D histogram of thickness as a function of branch orders
        :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.thickness_x_branch_order()
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
        q = deque(self)
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

    def thickness_x_path_distance(self, bins=20):
        """Creates 2D histogram of neurite radii as a function of path distances to the soma in microns
                (across all nodes)

        :param n: neuron
        :type n: :class:`Neuron`

        :returns: 2D histogram of thickness as a function of path distances
        :rtype: `tuple` of three `numpy.array`, respectively histogram, x edges, and y edges

        Example:
            >>> neuron = from_swc("Example1.swc")
            >>> neuron.thickness_x_path_distance()
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
        q = deque(self)
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
                length += i.euclidean_dist(i.parent)
                i = i.parent
            if length > maxDist:
                maxDist = length
            dist.append(length)
            q.extend(i2)
        return np.histogram2d(dist, radii, bins=bins, range=[[0, maxDist], [0, maxRadii]])

    @staticmethod
    def from_swc(fname, force_format=True):
        """
        :param fname: A SWC file to import
        :type fname: `str` containing a file path
        :returns: A new neuron created from :attr:`fname`
        :rtype: :class:`Neuron`
        """
        out = Neuron()

        metadata = ""
        soma_pts = []
        soma_ids = set()
        branches = []

        soma_lookup = {}
        branch_lookup = {}

        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) is 0:
                    continue
                if line.startswith("#"):
                    metadata += line + "\n"
                    continue

                line = line.split()
                if len(line) < 7 and force_format:
                    raise AttributeError(
                        "Line "
                        + str(line)
                        + " does not meet formatting requirements. Add force_format=True to ignore this message"
                    )

                pt_num = int(line[0])
                pt_type = int(line[1])
                pt_x = float(line[2])
                pt_y = float(line[3])
                pt_z = float(line[4])
                pt_r = float(line[5])
                pt_parent = int(line[6])

                if pt_type == 1:  # soma
                    toadd = (pt_x, pt_y, pt_z, pt_r)
                    soma_pts.append(toadd)
                    soma_ids.add(pt_num)
                    soma_lookup[pt_num] = toadd
                else:
                    child = TracingPoint(pt_x, pt_y, pt_z, pt_r, pt_type, fid=pt_num)
                    if pt_parent in soma_ids:  # root of branch
                        parent = soma_lookup[pt_parent]
                        parent = TracingPoint(
                            x=parent[0],
                            y=parent[1],
                            z=parent[2],
                            r=parent[3],
                            t=1,
                            fid=pt_parent,
                        )
                        child.parent = parent
                        parent.add_child(child)

                        branch_lookup[pt_parent] = parent
                        branch_lookup[pt_num] = child

                        branches.append(parent)
                    elif pt_parent == -1:
                        child.parent = None
                        branch_lookup[pt_num] = child
                        branches.append(
                            child
                        )  # add branch for child since this is complicated
                    else:
                        if pt_parent not in branch_lookup:
                            raise ValueError("Parent id %d not present" % (pt_parent))
                        branch_lookup[pt_parent].add_child(child)
                        child.parent = branch_lookup[pt_parent]
                        branch_lookup[pt_num] = child

        out.add_soma_points(soma_pts)
        for i in branches:
            out.add_branch(i)
        out.metadata = metadata
        return out

    @staticmethod
    def from_swc_text(data, force_format=True):
        """
        :param fname: Import a SWC object from a text object
        :type data: `str` containing SWC-formatted data
        :returns: A new neuron created from :attr:`data`
        :rtype: :class:`Neuron`
        """
        out = Neuron()

        metadata = ""
        soma_pts = []
        soma_ids = set()
        branches = []

        soma_lookup = {}
        branch_lookup = {}

        if True:
            for line in data:
                line = line.strip()
                if len(line) is 0:
                    continue
                if line.startswith("#"):
                    metadata += line + "\n"
                    continue

                line = line.split()
                if len(line) < 7 and force_format:
                    raise AttributeError(
                        "Line "
                        + str(line)
                        + " does not meet formatting requirements. Add force_format=True to ignore this message"
                    )

                pt_num = int(line[0])
                pt_type = int(line[1])
                pt_x = float(line[2])
                pt_y = float(line[3])
                pt_z = float(line[4])
                pt_r = float(line[5])
                pt_parent = int(line[6])

                if pt_type == 1:  # soma
                    toadd = (pt_x, pt_y, pt_z, pt_r)
                    soma_pts.append(toadd)
                    soma_ids.add(pt_num)
                    soma_lookup[pt_num] = toadd
                else:
                    child = TracingPoint(pt_x, pt_y, pt_z, pt_r, pt_type, fid=pt_num)
                    if pt_parent in soma_ids:  # root of branch
                        parent = soma_lookup[pt_parent]
                        parent = TracingPoint(
                            x=parent[0],
                            y=parent[1],
                            z=parent[2],
                            r=parent[3],
                            t=1,
                            fid=pt_parent,
                        )
                        child.parent = parent
                        parent.add_child(child)

                        branch_lookup[pt_parent] = parent
                        branch_lookup[pt_num] = child

                        branches.append(parent)
                    elif pt_parent == -1:  # root of branch that doesn't connect to soma
                        child.parent = None
                        branch_lookup[pt_num] = child
                        branches.append(
                            child
                        )  # add branch for child since this is complicated
                    else:  # failed lookup
                        if pt_parent not in branch_lookup:
                            raise ValueError("Parent id %d not present" % (pt_parent))
                        branch_lookup[pt_parent].add_child(child)
                        child.parent = branch_lookup[pt_parent]
                        branch_lookup[pt_num] = child

        out.add_soma_points(soma_pts)
        for i in branches:
            out.add_branch(i)
        out.metadata = metadata
        return out

    def to_swc(self, fname=None):
        from collections import deque as queue

        # could be removed for speed, but prevents errors if someone messed with the memory object
        for i in self.branches:
            i.fix_parents()

        swc_counter = 1  # counter of swc line
        todo = queue([self])
        memory = {None: -1}  # track line numbers of each element
        out = []

        for layer in self.soma_layers.values():
            for a in layer:
                if a in memory:
                    continue
                # no children to extend
                # no parent to deal with
                out.append(
                    "%d %d %g %g %g %g %d" % (swc_counter, a.t, a.x, a.y, a.z, a.r, -1)
                )
                memory[a] = swc_counter
                swc_counter += 1

        for branch in self.branches:
            todo = queue([branch])
            is_root = True
            while todo:
                a = todo.pop()
                if is_root:
                    is_root = not is_root
                    todo.extend(a.children)
                else:
                    if a in memory:  # duplicate
                        continue
                    todo.extend(a.children)

                    parent = None
                    try:
                        parent = memory[a.parent]
                    except:
                        for layer in self.soma_layers.values():
                            for b in layer:
                                if (
                                    b.x == a.parent.x
                                    and b.y == a.parent.y
                                    and b.z == a.parent.z
                                ):
                                    parent = b
                        parent = memory[parent]

                    memory[a] = swc_counter

                    # Columns:  id  t  x  y  z  r  pid
                    out.append(
                        "%d %d %g %g %g %g %d"
                        % (swc_counter, a.t, a.x, a.y, a.z, a.r, parent)
                    )
                    swc_counter += 1

        out = "\n".join(out)
        if fname is None:
            return out
        f = open(fname, "w")
        f.write(out)
        f.close()
