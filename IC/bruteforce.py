import networkx as nx

from IC import runIC, avgSize
from degreeDiscount import degreeDiscountIC
from newGreedyIC import newGreedyIC
from CCHeuristic import CC_heuristic
from singleDiscount import singleDiscount
import multiprocessing
from heapq import nlargest
import matplotlib.pylab as plt
import os
from itertools import combinations

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('small_graph.txt') as f:
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

    seed_size = 5
    p = .01
    nodes = G.nodes()
    C = combinations(nodes, seed_size)

    spread = dict()
    for candidate in C:
        print candidate,
        time2spread = time.time()
        spread[candidate] = avgSize(G, list(candidate), p, 1000)
        print spread[candidate], time.time() - time2spread

    S, val = max(spread.iteritems(), key = lambda (dk, dv): dv)

    print 'S (by brute-force):', S, ' -->', val

    S2 = degreeDiscountIC(G, seed_size, p)
    print 'S (by degree discount):', tuple(S2), ' -->', avgSize(G, S2, p, 1000)
    print 'S (by degree discount) spreads to %s nodes (according to brute-force)' %(spread[tuple(sorted(S2))])
    print 'Total time:', time.time() - start

    console = []