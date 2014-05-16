__author__ = 'ivanovsergey'

import math
import networkx as nx
from DD import DD
from runIAC import avgIAC, randomEp, random_from_range, uniformEp


def binaryDegreeDiscount (G, tsize, Ep, step, I):
    ''' Finds minimal number of nodes necessary to reach tsize number of nodes
    using degreeDiscount algorithms and binary search.
    Input: G -- networkx graph object
    tsize -- number of nodes necessary to reach
    p -- propagation probability
    a -- fraction of tsize to use as initial seed set size
    step -- step between iterations of binary search
    iterations -- number of iterations to average independent cascade
    Output:
    S -- seed set
    Tspread -- spread values for different sizes of seed set
    '''
    Tspread = dict()
    # find initial total spread
    k0 = 1
    S = DD(G, k0, Ep)
    t = avgIAC(G, S, Ep, I)
    Tspread[k0] = t
    # find bound (lower or upper) of total spread
    k = k0
    print k, step, Tspread[k]
    if t >= tsize:
        # find the value of k that doesn't spread influence up to tsize nodes
        step *= -1
        while t >= tsize:
            # reduce step if necessary
            while k + step < 0:
                step = int(math.ceil(float(step)/2))
            k += step
            S = DD(G, k, Ep)
            t = avgIAC(G, S, Ep, I)
            Tspread[k] = t
            print k, step, Tspread[k]
    else:
        # find the value of k that spreads influence up to tsize nodes
        while t < tsize:
            k += step
            S = DD(G, k, Ep)
            t = avgIAC(G, S, Ep, I)
            Tspread[k] = t
            print k, step, Tspread[k]

    if Tspread[k] < Tspread[k-step]:
        k -= step
        step = abs(step)

    # search precise boundary
    stepk = step
    while abs(stepk) != 1:
        if Tspread[k] >= tsize:
            stepk = -int(math.ceil(float(abs(stepk))/2))
        else:
            stepk = int(math.ceil(float(abs(stepk))/2))
        k += stepk

        if k not in Tspread:
            lastS = DD(G, k, Ep)
            Tspread[k] = avgIAC(G, lastS, Ep, I)
            if Tspread[k] > tsize:
                S = lastS
        print k, stepk, Tspread[k]

    return S, Tspread

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
                G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    time2result = time.time()
    tsize = 450
    Ep = uniformEp(G, .01)
    S, Tsize = binaryDegreeDiscount(G, tsize, Ep, step=100, I=1000)
    print S
    print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(S), tsize)
    print 'Time to find result:', time.time() - time2result
    print 'Total time:', time.time() - start
    console = []