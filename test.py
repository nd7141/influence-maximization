''' File for testing different files
'''
__author__ = 'ivanovsergey'

from IC import runIC
from representativeNodes import representativeNodes
from degreeDiscount import degreeDiscountIC
import networkx as nx
import matplotlib.pylab as plt

if __name__ == '__main__':
    import time
    start = time.time()

    G = nx.Graph()
    with open('graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
    print time.time() - start

    S = degreeDiscountIC(G, int(len(G)/100))
    print 'Initial set of', int(len(G)/100), 'nodes'
    print time.time() - start
    T = runIC(G, S)
    print 'Targeted', len(T), 'nodes out of', len(G)

    console = []
