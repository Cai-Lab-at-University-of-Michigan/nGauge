"""
Provides mathematical functions used in the nGauge package
"""

from numpy import cos, sin
import numpy as np

__num_types__ = set([float, int])


def rotation_matrix(a, b, c):
    """Creates a standard rotation matrix of three angles for rotation
    calculations

    :param a: The first rotation angle
    :type a: numeric

    :param b: The second rotation angle
    :type b: numeric

    :param c: The third rotation angle
    :type c: numeric

    :return: The rotation matrix
    :rtype: `numpy.array` (3x3)"""

    return np.array(
        [
            [
                cos(a) * cos(b),
                cos(a) * sin(b) * sin(c) - sin(a) * cos(c),
                cos(a) * sin(b) * cos(c) + sin(a) * sin(c),
            ],
            [
                sin(a) * cos(b),
                sin(a) * sin(b) * sin(c) + cos(a) * cos(c),
                sin(a) * sin(b) * cos(c) - cos(a) * sin(c),
            ],
            [-1.0 * sin(b), cos(b) * sin(c), cos(b) * cos(c)],
        ]
    )


def load_default_smat():
    """
    *Work in progress*
    """

    f = """	"(0,0.1]"	"(0.1,0.2]"	"(0.2,0.3]"	"(0.3,0.4]"	"(0.4,0.5]"	"(0.5,0.6]"	"(0.6,0.7]"	"(0.7,0.8]"	"(0.8,0.9]"	"(0.9,1]"
"(0,0.75]"	9.500096818	9.215083357	9.211150653	8.7784602	9.164807909	9.226703049	9.981771246	9.987695406	10.80477034	11.38922975
"(0.75,1.5]"	8.447755355	9.046068319	8.667958982	8.620980802	8.776274813	8.991696789	9.617999412	9.493972245	9.903896429	10.55586004
"(1.5,2]"	7.814143229	8.275576335	8.186606829	8.237314279	8.155985165	8.449820935	9.003032526	8.779511491	9.077598206	9.727351671
"(2,2.5]"	7.516167196	7.681555246	7.825236429	7.79365903	7.886876327	8.031765025	7.904194473	7.891676679	8.462178601	9.356472385
"(2.5,3]"	6.978314733	6.94307802	7.079217658	7.049650785	7.213062838	6.938749021	7.636968226	7.400216229	8.243724006	8.805585249
"(3,3.5]"	6.337198777	6.510450375	6.357374227	6.730667645	6.641335772	6.684942997	6.845211004	6.965403404	7.584209784	8.309956403
"(3.5,4]"	5.734997422	5.776563856	5.874881169	6.078469213	6.024177456	5.936484828	6.165189213	6.300636628	6.959851818	7.873737324
"(4,5]"	5.115815483	5.021649496	5.156574953	5.104265236	5.140931058	5.108690757	5.313504178	5.32953037	5.908950758	6.513172334
"(5,6]"	4.233994961	4.157947722	4.207281576	4.154590177	4.126860665	4.073368024	4.139708907	4.300275653	4.578050608	5.16486935
"(6,7]"	3.340269069	3.305132487	3.295987474	3.26045244	3.292369387	3.178867136	3.359775859	3.354096549	3.576372369	3.975850334
"(7,8]"	2.495160396	2.52098425	2.52305844	2.469504149	2.482755854	2.495893622	2.532479647	2.478894493	2.57140863	3.033875753
"(8,9]"	1.802393086	1.781094655	1.70675762	1.775359089	1.752898559	1.751466981	1.790826809	1.714786956	1.765916151	2.111905427
"(9,10]"	1.232040898	1.249021758	1.150560463	1.153606462	1.105376439	1.090955764	1.112113408	1.073979959	1.213295343	1.362314481
"(10,12]"	0.401029978	0.405860319	0.364813354	0.445292762	0.340571514	0.3381995	0.280082921	0.257239082	0.309758723	0.460951328
"(12,14]"	-0.232687427	-0.28491254	-0.336660961	-0.341205197	-0.403612584	-0.44962312	-0.41046464	-0.49492806	-0.486278922	-0.343856434
"(14,16]"	-0.720642343	-0.737187894	-0.791598721	-0.913295681	-0.86587451	-0.92991461	-0.938060799	-0.94957494	-0.949001462	-0.892505829
"(16,20]"	-1.207753675	-1.224291438	-1.232821022	-1.3177789	-1.33458514	-1.381696401	-1.399438894	-1.355855899	-1.366778332	-1.314132536
"(20,25]"	-1.645904535	-1.672684781	-1.698075889	-1.756182812	-1.79136287	-1.878540375	-1.874187273	-1.919542562	-1.939410932	-1.937971322
"(25,30]"	-2.517777195	-2.545349187	-2.539723488	-2.5457632	-2.606812735	-2.685946309	-2.663268872	-2.701847019	-2.781737863	-2.912276452
"(30,40]"	-3.960090407	-4.031387597	-4.072118021	-4.147351353	-4.330029905	-4.420053362	-4.507915144	-4.794051466	-4.833212928	-5.085672535
"(40,500]"	-9.921038172	-10.08763	-10.05543472	-10.10268204	-10.08682408	-9.912201864	-10.07995763	-9.951978816	-10.05360783	-10.12875887"""

    f = [x.split("\t") for x in f.replace('"', "").splitlines()]
    products, f = f[0][1:], f[1:]
    dists, f = [x[0] for x in f], [list(map(float, x[1:])) for x in f]

    dists = [tuple(map(float, x[1:-1].split(","))) for x in dists]
    products = [tuple(map(float, x[1:-1].split(","))) for x in products]

    import numpy as np

    f = np.array(f, dtype=float)

    return products, dists, f


def tangent_from_points(list_of_points):
    """
    Returns a tangent vector from a list of points
    """

    ## TODO: check list fomatting

    list_of_points = np.array(list_of_points)
    u, s, vh = scipy.linalg.svd(list_of_points)
    del u, s

    return vh[0]


def f_function_map(di, dp, smat=None):  # map d,a*b to f
    """
    *Work in progress*
    """
    if not smat:
        smat = load_default_smat()
    products, dists, f = smat

    # This needs to be rewritten using proper lookup functions (only for perf)
    for i, v in enumerate(dists):
        if v[0] <= di and v[1] > di:
            for j, vb in enumerate(products):
                if vb[0] <= dp and vb[1] > dp:
                    return f[i][j]
            break  # speed up function misses
    return None


def dot(a, b):
    """
    :param a: A `tuple` representing a vector
    :type a: `tuple` of numeric
    :param b: A `tuple` representing a vector
    :type b: `tuple` of numeric
    :returns: The dot product of `a` and `b`
    :rtype: `numeric`
    """
    return sum(i * j for i, j in zip(a, b))


def abs_dot(a, b):
    """
    :param a: A `tuple` representing a vector
    :type a: `tuple` of numeric
    :param b: A `tuple` representing a vector
    :type b: `tuple` of numeric
    :returns: The absolute value dot product of `a` and `b` (see :func:`dot`)
    :rtype: `numeric`
    """
    return abs(dot(a, b))


def abs_dot_3d(a, b):
    """
    :param a: A 3-dimensional `tuple` representing a vector
    :type a: `tuple` of numeric
    :param b: A 3-dimensional `tuple` representing a vector
    :type b: `tuple` of numeric
    :returns: The absolute value dot product of `a` and `b` (see :func:`dot`)
    :rtype: `numeric`
    """
    return abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
