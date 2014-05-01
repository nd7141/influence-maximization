from __future__ import division
import networkx as nx
import matplotlib.pylab as plt
from LT import *
import time, os
from greedy import *
from LDAG import *
from copy import deepcopy

if __name__ == "__main__":

    start = time.time()

    # G = nx.DiGraph()
    # G.add_edge(1,2,weight=1)
    # G.add_edge(3,2,weight=3)
    # G.add_edge(2,3,weight=1)
    # G.add_edge(3,3,weight=1)
    # G.add_edge(4,5,weight=5)
    #
    # with open('../graphdata/graph30.txt') as f:
    #     n, m = map(int, f.readline().split())
    #     G = nx.DiGraph()
    #     for line in f:
    #         u, v = map(int, line.split())
    #         try:
    #             G[u][v]['weight'] += 1
    #         except KeyError:
    #             G.add_edge(u,v,weight=1)
    #         try:
    #             G[v][u]['weight'] += 1
    #         except KeyError:
    #             G.add_edge(v,u,weight=1)
    # print 'Built Graph G'
    # print time.time() - start

    # Ewu = uniformWeights(G)
    # Ewr = randomWeights(G)
    # S = [0, 7, 11, 17, 24]

    with open('../graphdata/hep.txt') as f:
        n, m = map(int, f.readline().split())
        G = nx.DiGraph()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except KeyError:
                G.add_edge(u,v,weight=1)
            try:
                G[v][u]['weight'] += 1
            except KeyError:
                G.add_edge(v,u,weight=1)
    print 'Built Graph G'
    print time.time() - start

    Ewu = uniformWeights(G)
    Ewr = randomWeights(G)
    print 'Found edge weights'
    print time.time() - start

    # find seed set
    S = LDAG_heuristic(G, Ewu, 50, 1.0/320)
    print 'Found seed set'
    print time.time() - start

    with open('LDAG.txt', 'w') as f:
        for node in S:
            f.write(str(node) + os.linesep)

    print 'Finding spread for S...'
    print avgLT(G, S, Ewu, 2000)

    console = []