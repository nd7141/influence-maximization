__author__ = 'ivanovsergey'

import math

import networkx as nx

from degreeDiscount import degreeDiscountIC
from IC.IC import avgSize


def binaryDegreeDiscount (G, tsize, p=.01, a=0.38, step=5, iterations=200):
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
    k0 = int(a*tsize)
    S = degreeDiscountIC(G, k0, p)
    t = avgSize(G, S, p, iterations)
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
            S = degreeDiscountIC(G, k, p)
            t = avgSize(G, S, p, iterations)
            Tspread[k] = t
            print k, step, Tspread[k]
    else:
        # find the value of k that spreads influence up to tsize nodes
        while t < tsize:
            k += step
            S = degreeDiscountIC(G, k, p)
            t = avgSize(G, S, p, iterations)
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
            S = degreeDiscountIC(G, k, p)
            Tspread[k] = avgSize(G, S, p, iterations)
        print k, stepk, Tspread[k]

    return S, Tspread

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('graphdata/../graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    tsize = 200
    S, Tsize = binaryDegreeDiscount(G, tsize, step=5)
    print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(S), tsize)
    print time.time() - start
    console = []