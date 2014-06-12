'''
Independent Arbitrary Cascade (IAC) is a independent cascade model with arbitrary
 propagation probabilities.
'''
from __future__ import division
from copy import deepcopy
import random, multiprocessing, os
import networkx as nx

def uniformEp(G, p = .01):
    '''
    Every edge has the same probability p.
    '''
    Ep = dict()
    for v1,v2 in G.edges():
        Ep[(v1,v2)] = p
        Ep[(v2,v1)] = p
    return Ep

def randomEp(G, maxp):
    '''
    Every edge has random propagation probability <= maxp <= 1
    '''
    assert maxp <= 1, "Maximum probability cannot exceed 1."
    Ep = dict()
    for v1,v2 in G.edges():
        p = random.uniform(0, maxp)
        Ep[(v1,v2)] = p
        Ep[(v2,v1)] = p
    return Ep

def random_from_range (G, prange):
    '''
    Every edge has propagation probability chosen from prange uniformly at random.
    '''
    for p in prange:
        if p > 1:
            raise ValueError, "Propagation probability inside range should be <= 1"
    Ep = dict()
    for v1,v2 in G.edges():
        p = random.choice(prange)
        Ep[(v1,v2)] = p
        Ep[(v2,v1)] = p
    return Ep

def runIAC (G, S, Ep):
    ''' Runs independent arbitrary cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    Ep -- propagation probabilities
    Output: T -- resulted influenced set of vertices (including S)

    NOTE:
    Ep is a dictionary for each edge it has associated probability
    If graph is undirected for each edge (v1,v2) with probability p,
     we have Ep[(v1,v2)] = p, Ep[(v2,v1)] = p.
    '''
    T = deepcopy(S) # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                p = Ep[(T[i],v)] # propagation probability
                if random.random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
        i += 1
    return T

def avgIAC (G, S, Ep, I):
    '''
    Input:
        G -- undirected graph
        S -- seed set
        Ep -- propagation probabilities
        I -- number of iterations
    Output:
        avg -- average size of coverage
    '''
    avg = 0
    for i in range(I):
        avg += float(len(runIAC(G,S,Ep)))/I
    return avg

def findCC(G, Ep):
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1-Ep[e])**(E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)

    # initialize CC
    CC = dict() # each component is reflection os the number of a component to its members
    explored = dict(zip(E.nodes(), [False]*len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CC[c] = [node]
            component = E[node].keys()
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CC[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    return CC

def getScores(G, k, Ep):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     k -- number of nodes in seed set (int)
     p -- propagation probability among all edges (int)
     Output:
     scores -- scores of nodes according to some weight function (dict)
    '''
    scores = dict(zip(G.nodes(), [0]*len(G))) # initialize scores

    CC = findCC(G, Ep)

    # find ties for components of rank k and add them all as qualified
    sortedCC = sorted([(len(dv), dk) for (dk, dv) in CC.iteritems()], reverse=True)
    topCCnumbers = sortedCC[:k] # CCs we assign scores to
    L = sum([l for (l,_) in topCCnumbers])

    increment = 0
    # add ties of rank k (if len of kth CC == 1 then add all CCs)
    while k+increment < len(sortedCC) and sortedCC[k + increment][0] == sortedCC[k-1][0]:
        topCCnumbers.append(sortedCC[k + increment])
        increment += 1

    # assign scores to nodes in top Connected Components
    # prev_length  = topCCnumbers[0][0]
    # rank = 1
    for length, numberCC in topCCnumbers:
        # if length != prev_length:
        #     prev_length = length
        #     rank += 1
        # weighted_score = length
        weighted_score = 1.0/length
        # weighted_score = 1
        for node in CC[numberCC]:
            scores[node] += weighted_score
    return scores

def CCWP(G, k, Ep, R):
    def map_CCWP(it):
        return getScores(G, k, Ep)
    Scores = map(map_CCWP, range(R))

    scores = {v: sum([s[v] for s in Scores]) for v in G}
    scores_copied = deepcopy(scores)
    S = []
    # penalization phase
    for it in range(k):
        maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv)
        # print maxv,
        S.append(maxk)
        scores_copied.pop(maxk) # remove top element from dict
        for v in G[maxk]:
            if v not in S:
                # weight = scores_copied[v]/maxv
                # print weight,
                penalty = (1-Ep[(maxk, v)])**(G[maxk][v]['weight'])
                scores_copied[v] = penalty*scores_copied[v]
    return S

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('../../graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
            # G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    random.seed(1)

    time2probability = time.time()
    prange = [.01, .02, .04, .08]
    Ep = random_from_range(G, prange)
    print 'Built probabilities Ep'
    print time.time() - time2probability



    with open("Ep_hep_range1.txt", "w+") as f:
        for key, value in Ep.iteritems():
            f.write(str(key[0]) + " " + str(key[1]) + " " + str(value) + os.linesep)
    #
    #
    # time2probability = time.time()
    # Ep = randomEp(G, .1)
    # print 'Built probabilities Ep'
    # print time.time() - time2probability
    #
    # with open("Ep_hep_random1.txt", "w+") as f:
    #     for key, value in Ep.iteritems():
    #         f.write(str(key[0]) + " " + str(key[1]) + " " + str(value) + os.linesep)

    # time2probability = time.time()
    # Ep = uniformEp(G, .01)
    # print 'Built probabilities Ep'
    # print time.time() - time2probability
    #
    # with open("Ep_hep_uniform1.txt", "w+") as f:
    #     for key, value in Ep.iteritems():
    #         f.write(str(key[0]) + " " + str(key[1]) + " " + str(value) + os.linesep)

    # S1 = [66, 76, 111, 3683, 4824, 412, 608, 145, 559, 512, 80, 192, 236, 100, 221, 646, 37, 535, 60, 274, 124, 247, 1890, 1162, 8, 156, 606, 267, 1159, 1292, 41, 239, 741, 154, 426, 422, 140, 88, 805, 525, 1617, 251, 682, 556, 515, 1692, 56, 214, 639, 354, 110, 1257, 1094, 23, 27, 1256, 553, 457, 1827, 634, 562, 2846, 2389, 417, 79, 142, 612, 1954, 1980, 382, 129, 230, 1987, 1449, 885, 265, 122, 15, 99, 736, 519, 5682, 1515, 507, 754, 12, 21, 2497, 866, 92, 950, 3042, 196, 2491, 128, 2183, 210, 624, 649, 1030, 599, 419, 547, 3029, 1156, 5892, 1171, 1153, 193, 1689, 5220, 116, 261, 44, 945, 9994, 329, 1869, 150, 4804, 1060, 396, 981, 613, 862, 1690, 3147, 1775, 2506, 545, 72, 36, 51, 2259, 1115, 1547, 3399, 74, 6561, 1520, 1944, 86, 227, 2017, 255, 632, 571, 310, 107, 2603, 316, 1434, 1759, 151, 474, 563, 31, 1589, 887, 3462, 4280, 3791, 5389, 688, 487, 307, 1103, 1655, 77, 6477, 604, 393, 1765, 514, 1943, 1597, 19, 1764, 610, 328, 135, 2283, 408, 2075, 834, 1327, 586, 592, 850, 315, 777, 2721, 220, 3247, 709, 1599, 2119, 6072, 989, 765, 1405, 1880, 1514, 2927, 1738, 6675, 1730, 1313, 1025, 349, 2626, 883, 742, 195, 2019, 784, 1930, 1793, 234, 5000, 1683, 5572, 157, 700, 2817, 597, 105, 3138, 842, 1049, 8372, 2879, 288, 10084, 511, 14, 2069, 2008, 460, 609, 4273, 797, 4048, 3212, 2409, 152, 607, 3498, 249, 721, 415, 656, 9365, 839, 1044, 3641, 1754, 3792, 1665, 3429, 168, 1, 566, 159, 89, 143, 9576, 20, 524, 791, 4263, 1316, 118, 572, 3122, 174, 7369, 7252, 2462, 85, 6583, 8450, 800, 3736, 133, 1429, 4579, 4266, 1656, 246, 3770, 10813, 2704, 1946, 954, 6271, 312, 2182, 258, 941, 6086, 985, 661, 593, 1587, 2731, 2804, 2573, 240, 2860, 363, 1059, 155, 1528, 224, 1671, 284, 6330, 1031, 1652, 637, 5629, 1035, 8256, 526, 539, 3210, 8293, 5893, 4878, 2002, 2331, 361, 1370, 3164, 7780, 8630, 695, 635, 3677, 9908, 4854, 7070, 5221, 1193, 1074, 2596, 1343, 170, 3778, 1312, 4582, 3779, 5600, 5486, 482, 1197, 2600, 2717, 737, 11431, 667, 1209, 2247, 260, 2293, 437, 2305, 590, 1046, 2038, 1179, 4319, 10, 4585, 1282, 1238, 5370, 389, 136, 2086, 2602, 653, 953, 1955, 61, 26, 1635, 4900, 442, 423, 2011, 861, 1881, 3632, 3696, 2796, 3789, 8991, 4800, 5363, 226, 2133, 205, 336, 4701, 740, 1169, 231, 622, 11406, 2250, 5098, 1351, 4316, 3488, 12186, 4141, 3078, 804, 409, 944, 817, 799, 1036, 690, 242, 3283, 1610, 3908, 833, 651, 3647, 927, 6071, 1290, 1731, 585, 2071, 5004, 1188, 1521, 1492, 295, 8400, 3128, 1102, 283, 2094, 11420, 481, 7077, 2153, 602, 2377, 3530, 3492, 4458, 2117, 6021, 39, 6067, 1078, 1394, 5450, 5664, 43, 1570, 629, 687, 6142, 523, 1706, 103, 536, 7225, 890, 2833, 306, 8899, 8399, 1783, 1466, 10106, 4272, 325, 2374, 4309, 186, 4996, 706, 844, 1657, 4329, 835, 6113, 1960, 7254, 440, 498, 6123, 4252, 11140, 149, 1569, 3028, 11410, 1758, 341, 2807, 297, 7295, 120, 207, 5922, 752, 8255, 308, 32, 564, 1746, 1645, 770, 4320, 11408, 1776, 38, 6265, 4518, 9964, 4156, 2114, 2273, 1221, 6174, 3906, 880, 117, 3553, 1218, 87, 5124, 268, 1967, 1175, 5231, 2787, 1966, 6563, 828, 494, 987, 6699, 245, 5106, 376, 2294, 1331, 69, 8987, 11427, 1841, 2549, 516, 7101, 2739, 401, 5901, 2928, 375, 263, 6122, 2077, 724, 2362, 3780, 42, 386, 2526, 1473, 10062, 259, 5601, 1318, 11428, 827, 6132, 8346, 1988, 7854, 2541, 2841, 4228, 75, 1392, 1530, 6573, 6680, 2244, 7794, 2280, 1969, 2729, 367, 6018, 8210, 6009, 2061, 13256, 4426, 3837, 3041, 936, 11419, 244, 4302, 5367, 1032, 1728, 421, 3670, 121, 9276, 10105, 323, 5313, 2748, 1755, 5003, 4083, 3422, 9379, 331, 48, 1411, 7620, 11423, 3320, 626, 1545, 1390, 1578, 5454, 988, 6684, 2315, 531, 4305, 3682, 11416, 5235, 650, 6638, 7368, 7951, 642, 991, 847, 1784, 4387, 5105, 4882, 209, 7172, 5007, 4920, 11411, 1379, 9280, 7355, 2147, 1464, 2067, 5287, 2295, 1782, 10319, 7438, 46, 357, 2297, 8509, 8652, 436, 8507, 3651, 330, 959, 4586, 1676, 6133, 194, 4158, 4151, 1852, 4519, 3673, 627, 5813, 2941, 4133, 1023, 3015, 1485, 71, 167, 13246, 753, 9265, 11429, 4979, 1098, 2551, 8744, 403, 2314, 1166, 2682, 2333, 5020, 2545, 4041, 416, 413, 3533, 3, 942, 4047, 1941, 3027, 326, 750, 4, 322, 636, 141, 2364, 1377, 809, 9278, 793, 2848, 340, 1338, 2185, 1202, 3978, 3646, 1516, 6002, 11422, 771, 1198, 2801, 173, 4578, 238, 879, 33, 67, 6089, 2615, 1465, 10807, 5651, 8893, 3070, 1037, 5033, 8024, 270, 9263, 4469, 5772, 11426, 47, 595, 4784, 1020, 5879, 1266, 4560, 3254]
    # print avgIAC(G, S1, Ep, 1000)
    #
    # S2 = [131, 639, 287, 267, 608, 100, 559, 124, 359, 66, 1292, 236, 80, 646, 412, 76, 606, 535, 8, 247, 274, 15, 3683, 1162, 4824, 196, 60, 556, 145, 239, 512, 154, 525, 1890, 111, 41, 192, 156, 310, 265, 784, 1159, 382, 515, 426, 1449, 1692, 37, 422, 741, 140, 214, 981, 1094, 457, 1256, 415, 487, 736, 354, 885, 1156, 599, 1434, 128, 56, 1954, 110, 682, 129, 1827, 122, 950, 1980, 2846, 3462, 866, 1690, 2017, 661, 142, 1987, 649, 44, 2389, 3042, 12, 507, 210, 1547, 519, 572, 51, 92, 116, 834, 150, 586, 2183, 3029, 9994, 193, 610, 625, 634, 754, 2491, 315, 562, 1520, 5389, 174, 2259, 36, 419, 5682, 31, 1515, 2506, 1765, 862, 1153, 1759, 5220, 6583, 74, 700, 3792, 2497, 307, 545, 1115, 1327, 1764, 1881, 6132, 6477, 1656, 393, 1730, 1943, 2928, 985, 2019, 349, 79, 2603, 1652, 2721, 107, 2283, 20, 1599, 2069, 220, 842, 8450, 341, 3791, 688, 231, 4272, 5367, 255, 2817, 1036, 2462, 2879, 3770, 6072, 3164, 1030, 1930, 3212, 3147, 1394, 632, 1171, 1074, 1671, 2626, 3122, 1209, 1313, 8256, 1944, 89, 1754, 3138, 1018, 3736, 3779, 555, 312, 1343, 4319, 5003, 6271, 1793, 238, 1530, 3283, 804, 2153, 5922, 7070, 7252, 10812, 8507, 136, 827, 1665, 2038, 2294, 3530, 3647, 5600, 7780, 284, 524, 839, 2573, 3553, 2117, 835, 2133, 4309, 1379, 224, 642, 4266, 4458, 7225, 597, 585, 695, 1852, 2331, 3128, 4469, 941, 295, 724, 376, 1569, 71, 325, 5313, 7172, 9908, 2600, 5629, 401, 1351, 1783, 1969, 2061, 2374, 2615, 4083, 389, 5335, 521, 2215, 4878, 6330, 770, 927, 1610, 1657, 1706, 2796, 3696, 4444, 4979, 1188, 2075, 1045, 2545, 1870, 3492, 3028, 8899, 690, 271, 1331, 2729, 4228, 5287, 1429, 77, 3319, 6699, 7368, 5893, 10105, 69, 436, 531, 1020, 1423, 2541, 3948, 8893, 8987, 10098, 590, 357, 1390, 1724, 2280, 47, 1282, 3670, 3675, 263, 5664, 1179, 8867, 29, 299, 557, 1002, 1677, 3254, 4387, 4996, 8346, 8887, 9379, 21, 799, 742, 403, 2362, 9576, 2833, 171, 367, 936, 1516, 4329, 4523, 6807, 7620, 8267, 9261, 189, 1758, 6482, 169, 6680, 12186, 87, 2739, 5069, 616, 806, 1361, 2185, 2848, 3837, 4561, 5020, 6169, 7435, 9370, 10321, 11079, 225, 850, 421, 442, 7854, 1855, 2785, 6352, 7101, 10616, 11089, 293, 9262, 3073, 5449, 8147, 987, 3777, 5454, 278, 478, 888, 1506, 1743, 3174, 3422, 4318, 6244, 7984, 8574, 8724, 10440, 13245, 4520, 3070, 880, 305, 1035, 1963, 3566, 3847, 4446, 5446, 6002, 8832, 9206, 2400, 167, 1238, 1868, 847, 3215, 622, 375, 8856, 494, 2707, 3533, 6095, 435, 468, 587, 713, 2275, 2633, 3425, 3707, 3743, 4343, 4481, 4754, 5594, 5828, 8207, 10128, 11626, 285, 654, 854, 2679, 3344, 3433, 10071, 13246, 1277, 2977, 11404, 1067, 459, 657, 2805, 1946, 46, 1240, 1410, 1413, 2033, 2127, 2194, 2441, 2583, 2653, 2675, 3684, 4973, 5034, 5624, 5936, 6370, 6512, 7155, 7267, 9886, 11923, 1174, 1195, 5239, 7417, 7588, 10142, 118, 1617, 2237, 9716, 8372, 1370, 737, 725, 9263, 9859, 1050, 6000, 8376, 3906, 2602, 6102, 1098, 1441, 1888, 2639, 2906, 3118, 3390, 3947, 4279, 4860, 5097, 5282, 5607, 7922, 9954, 2501, 3130, 3914, 5060, 2247, 1524, 9081, 260, 571, 582, 3603, 6725, 1544, 6693, 9002, 350, 360, 447, 1039, 1212, 1232, 1320, 1339, 1592, 1912, 2821, 3846, 4036, 4665, 4777, 4898, 4931, 5080, 5176, 5477, 5550, 6165, 7203, 9775, 10473, 779, 1698, 2154, 2385, 4540, 7351, 4141, 1700, 3978, 438, 502, 1300, 7237, 873, 718, 469, 2803, 602, 199, 215, 911, 1033, 1549, 1934, 2176, 2351, 2408, 2873, 3361, 3870, 4353, 5996, 6040, 6873, 7065, 8471, 9602, 10063, 10637, 10644, 12085, 12179, 1637, 2209, 2442, 4693, 6836, 7359, 9963, 10752, 11120, 6100, 13247, 2455, 2177, 844, 290, 858, 1078, 1445, 4285, 1475, 1615, 1999, 2644, 3008, 3266, 3607, 3814, 3827, 3985, 4580, 4612, 4635, 4924, 5066, 6057, 6252, 6287, 6641, 7613, 7941, 8131, 8675, 8778, 10629, 11179, 11199, 11394, 12023, 12249, 52, 628, 1773, 2123, 2646, 4299, 4683, 5077, 5464, 8330, 9860, 11788, 323, 6174, 937, 11360, 8399, 2901, 3614, 1843, 4435, 5033, 1296, 2709, 3305, 1063, 1431, 7956, 439, 301, 915, 1108, 1243, 2040, 3043, 3374, 3915, 4129, 4662, 4699, 4731, 4944, 5855, 6106, 6409, 7299, 7304, 7309, 7352, 7676, 7724, 7766, 7897, 8339, 8428, 8454, 8728, 8950, 9149, 9720, 9825, 11651, 12962, 6067, 243, 660, 696, 920, 969, 1407, 1460, 3280, 5254, 5745, 8872, 9091, 10070, 10137, 12808, 8280]
    # print avgIAC(G, S2, Ep, 1000)

    # import json
    # coverage2length = [[0,0]]
    # with open("plotdata/rawCCWPforDirect1.txt") as f:
    #     for line in f:
    #         [(cov, S)] = json.loads(line).items()
    #         coverage2length.append([len(S), int(cov)])
    #
    # with open("plotdata/plotReverseCCWPforReverse1_v2.txt", "w+") as f:
    #     json.dump(coverage2length, f)


    console = []
