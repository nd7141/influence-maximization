__author__ = 'sergey'

import networkx as nx
import matplotlib.pylab as plt
from WC import *

if __name__ == "__main__":

    G = nx.DiGraph()
    G.add_edge(1,2,weight=1)
    G.add_edge(3,2,weight=3)
    G.add_edge(2,3,weight=1)
    G.add_edge(3,3,weight=1)
    G.add_edge(4,5,weight=5)

    with open('../graphdata/graph30.txt') as f:
        n, m = map(int, f.readline().split())
        D = nx.DiGraph()
        for line in f:
            u, v = map(int, line.split())
            try:
                D[u][v]['weight'] += 1
            except KeyError:
                D.add_edge(u,v,weight=1)
            try:
                D[v][u]['weight'] += 1
            except KeyError:
                D.add_edge(v,u,weight=1)

    console = []