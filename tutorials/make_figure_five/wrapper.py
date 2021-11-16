import morphoStats as s
import morphoDistributions as d
from ngauge import Neuron as n
from ngauge import TracingPoint as t
import matplotlib.pyplot as plt

def fillStatsDict(stats_dict):
    """Fills the statistics dictionary with all the statistics functions"""
    stats_dict["num_branch_points"] = s.num_branch_points
    stats_dict["num_tips"] = s.num_tips
    stats_dict["cell_height"] = s.cell_height
    stats_dict["cell_width"] = s.cell_width
    stats_dict["cell_depth"] = s.cell_depth
    stats_dict["num_stems"] = s.num_stems
    stats_dict["avg_thickness"] = s.avg_thickness
    stats_dict["total_length"] = s.total_length
    stats_dict["volume"] = n.total_volume
    stats_dict["max_neurite_length"] = s.max_neurite_length
    stats_dict["max_branch_order"] = s.max_branch_order
    stats_dict["max_segment"] = s.max_segment
    stats_dict["median_intermediate_segment"] = s.median_intermediate_segment
    stats_dict["median_terminal_segment"] = s.median_terminal_segment
    stats_dict["median_path_angle"] = s.median_path_angle
    stats_dict["max_path_angle"] = s.max_path_angle
    stats_dict["min_branch_angle"] = s.min_branch_angle
    stats_dict["avg_branch_angle"] = s.avg_branch_angle
    stats_dict["max_branch_angle"] = s.max_branch_angle
    stats_dict["max_degree"] = s.max_degree
    stats_dict["max_tortuosity"] = s.max_tortuosity
    stats_dict["median_tortuosity"] = s.median_tortuosity
    stats_dict["tree_asymmetry"] = s.tree_asymmetry
    
def fillDistrDict(distr_dict):
    """Fills the distribution dictionary with all the distribution functions"""
    distr_dict["branch_angles"] = d.branch_angles
    distr_dict["branch_orders"] = d.branch_orders
    distr_dict["path_angles"] = d.path_angles
    distr_dict["root_angles"] = d.root_angles
    distr_dict["euler_root_angles"] = d.euler_root_angles
    distr_dict["segment_lengths"] = d.segment_lengths
    distr_dict["thickness"] = d.thickness
    distr_dict["path_distance_to_soma"] = d.path_distance_to_soma
    distr_dict["euclidean_distances_to_soma"] = d.euclidean_distances_to_soma
    distr_dict["sholl_intersection"] = d.sholl_intersection
    distr_dict["branch_angles_x_branch_orders"] = d.branch_angles_x_branch_orders
    distr_dict["branch_angles_x_path_distances"] = d.branch_angles_x_path_distances
    distr_dict["path_angle_x_branch_order"] = d.path_angle_x_branch_order
    distr_dict["path_angle_x_path_distance"] = d.path_angle_x_path_distance
    distr_dict["thickness_x_branch_order"] = d.thickness_x_branch_order
    distr_dict["thickness_x_path_distance"] = d.thickness_x_path_distance
    
def run_all_morpho(n, morphoType = "all"):
    """Runs desired morphology statistics functions, all if not specified
    
    :param n: neuron for the calculations to be performed on
    :type n: :class:`Neuron`
    
    :param morphoType: specifies what functions are to be run, either All, stats, distr,
                   or a list of desired functions
    :type morphoType: `string` or `list` of `string`
    :raises AttributeError: morphoType must be either stats, distr, all, or a list of specific functions
    
    :returns: a dictionary with each function name mapped to its result 
    :rtype: `dictionary` of {`string`: function result type}
    
    Example 1 - stats:
        >>> neuron = from_swc("Example1.swc")
        >>> run_all_morpho(neuron, "stats")
        {'num_branch_points': 4,
         'num_tips': 5,
         'cell_height': 1.5,
         'cell_width': 7.0,
         'cell_depth': 4.0,
         'num_stems': 1,
         'avg_thickness': 1.0,
         'total_length': 14.762407402922234,
         'total_volume': 42.0,
         'max_neurite_length': 6.395172972263274,
         'max_branch_order': 3,
         'max_segment': 2.449489742783178,
         'median_intermediate_segment': 1.5,
         'median_terminal_segment': 1.5,
         'median_path_angle': 114.0948425521107,
         'max_path_angle': 172.50550517788332,
         'min_branch_angle': 83.62062979155719,
         'avg_branch_angle': 91.88778616188064,
         'max_branch_angle': 109.47122063449069,
         'max_degree': 2,
         'max_tortuosity': 0.1884243736377227,
         'median_tortuosity': 0.14956195330433844,
         'tree_asymmetry': 0.3333333333333333}
        
    Example 2 - distr:
        >>> neuron = from_swc("Example1.swc")
        >>> len(run_all_morpho(neuron, "distr"))
        16
        
    Example 3 - all:
        >>> neuron = from_swc("Example1.swc")
        >>> len(run_all_morpho(neuron, "all"))
        39
        
    Example 4 - list:
        >>> neuron = from_swc("Example1.swc")
        >>> func = ["volume", "thickness"]
        >>> run_all_morpho(neuron, func)
        {'volume': 42.0,
         'thickness': (array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10]),
          array([0.        , 0.03333333, 0.06666667, 0.1       , 0.13333333,
                 0.16666667, 0.2       , 0.23333333, 0.26666667, 0.3       ,
                 0.33333333, 0.36666667, 0.4       , 0.43333333, 0.46666667,
                 0.5       , 0.53333333, 0.56666667, 0.6       , 0.63333333,
                 0.66666667, 0.7       , 0.73333333, 0.76666667, 0.8       ,
                 0.83333333, 0.86666667, 0.9       , 0.93333333, 0.96666667,
                 1.        ]))}
    """
    out = dict()
    stats_dict = dict()
    distr_dict = dict()
    if type(morphoType) is list:
        fillStatsDict(stats_dict)
        fillDistrDict(distr_dict)
        for func in morphoType:
            if func in stats_dict:
                val = stats_dict[func](n)
                out[func] = val
            if func in distr_dict:
                val = distr_dict[func](n)
                out[func] = val
    elif morphoType == "stats":
        fillStatsDict(stats_dict)
        for func in stats_dict.values():
            val = func(n)
            out[func.__name__] = val
    elif morphoType == "distr":
        fillDistrDict(distr_dict)
        for func in distr_dict.values():
            val = func(n)
            out[func.__name__] = val
    elif morphoType == "all":
        fillStatsDict(stats_dict)
        fillDistrDict(distr_dict)
        for func in stats_dict.values():
            val = func(n)
            out[func.__name__] = val
        for func in distr_dict.values():
            val = func(n)
            out[func.__name__] = val
    else:
        raise AttributeError("morphoType must be either stats, distr, all, or a list of specific functions")
    return out