from collections import defaultdict, deque
import numpy as np

from ngauge.util import *
import ngauge.TracingPoint

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

    def total_width(self):
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
        :returns: The distance from the :func:`soma_centroid` to the positional average of all :attr:`branch` :class`TracingPoints`
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

    def blast(self, other, spacing=50):  # THIS CODE IS SO BAD NEEDS TO BE DELETED
        total_dist = 0.0
        for i, pt in enumerate(self.iter_all_points(exclude_soma=True)):
            if i % spacing != 0:
                continue
            total_dist += min(
                pt.euclidean_dist(opt)
                for opt in other.iter_all_points(exclude_soma=True)
            )
        return total_dist

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
