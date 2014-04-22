'''
Implements degreeDiscount heuristic that stops after necessary amount of nodes is targeted.
Now it calculates spread of influence after a step increase in seed size
and returns if targeted set size is greater than desired input tsize.
'''

import math

import networkx as nx

from degreeDiscount import degreeDiscountIC
from IC.IC import runIC


def binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations):
    # initialization for binary search

    R = iterations
    stepk = -int(math.ceil(float(step)/2))
    k += stepk
    if k not in Tsize:
        S = degreeDiscountIC(G, k, p)
        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R
        Tsize[k] = avg
    # check values of Tsize in between last 2 calculated steps
    while stepk != 1:
        print k, stepk, Tsize[k]
        if Tsize[k] >= targeted_size:
            stepk = -int(math.ceil(float(abs(stepk))/2))
        else:
            stepk = int(math.ceil(float(abs(stepk))/2))
        k += stepk

        if k not in Tsize:
            S = degreeDiscountIC(G, k, p)
            avg = 0
            for i in range(R):
                T = runIC(G, S, p)
                avg += float(len(T))/R
            Tsize[k] = avg
    return S, Tsize

def spreadDegreeDiscount(G, targeted_size, step=1, p=.01, iterations=200):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    targeted_size -- desired size of targeted set
    step -- step after each to calculate spread
    p -- propagation probability
    R -- number of iterations to average influence spread
    Output:
    S -- seed set that achieves targeted_size
    Tsize -- averaged targeted size for different sizes of seed set
    '''

    Tsize = dict()
    k = 0
    Tsize[k] = 0
    R = iterations

    while Tsize[k] <= targeted_size:
        k += step
        S = degreeDiscountIC(G, k, p)
        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R
        Tsize[k] = avg

        print k, Tsize[k]

    # binary search for optimal solution
    return binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations)

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

    targeted_size = 200
    S, Tsize = spreadDegreeDiscount(G, targeted_size, step=100)
    print time.time() - start

    console = []
