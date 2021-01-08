from src.morpho import Neuron, TracingPoint


def read_swc_file(fname):
    """Reads in SWC file and corrects the parent-child relations
    
    :param fname: file to be read
    :type fname: `string`
    
    :returns: corrected values
    :rtype: `dictionary`
    
    Example:
        >>> read_swc_file("neurite_2_v2.0.swc")
        TracingPoint(x=1, y=215, z=88, r=1, t=1, children=[TracingPoint(x=1, y=215, z=89, r=1, t=1, children=[{ 1, truncated }], parent={...})])
    """
    lines = open(fname, "r").readlines()
    i = 0
    if lines[i][0] == "#":
        while lines[i][0] == "#":
            i += 1
    lines = [line.strip().split() for line in lines[i:]]
    lines = [tuple(int(val.split(".")[0]) for val in x) for x in lines[i:]]
    lines = [
        (x[-1], TracingPoint(x[2], x[3], x[4], 1, 1, None, x[0])) for x in lines[i:]
    ]
    stack = {}
    for tp in lines[i:]:
        stack[tp[1].file_id] = tp

    for tp in stack.values():
        if tp[0] == -1:
            continue
        else:
            stack[tp[0]][1].add_child(tp[1])
    stack[1][1].fix_parents()
    return stack[1][1]


if __name__ == "__main__":
    start = read_swc_file("neurite_2_v2.0.swc")
    start.to_swc("output.swc")
