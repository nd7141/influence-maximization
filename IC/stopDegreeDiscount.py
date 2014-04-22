''' Implementation of degree discount heuristic [1] for Independent Cascade model
of influence propagation in graph G

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
'''
__author__ = 'ivanovsergey'
import math

import networkx as nx

from priorityQueue import PriorityQueue as PQ # priority queue
from IC import avgSize
from IC.spreadDegreeDiscount import binarySearchBoundary


def stopDegreeDiscount(G, tsize, ic_step=1, p=.01, iterations=200):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    tsize -- number of nodes necessary to reach
    ic_step -- step of change in k between 2 iterations of IC
    p -- propagation probability
    Output:
    S -- seed set
    Tspread -- spread values for different sizes of seed set
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    # until necessary number of nodes can be reached
    Tspread = dict() # spread for different k
    k = 0
    Tspread[k] = 0
    stepk = 1
    while Tspread[k] < tsize:
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                dd.add_task(v, -priority)
        # calculate IC spread with ic_step
        if stepk == ic_step:
            k = len(S)
            Tspread[k] = avgSize(G, S, p, iterations)
            print k, Tspread[k]
            stepk = 0
        stepk += 1

    # search precise boundary
    if abs(int(math.ceil(float(ic_step)/2))) == 1:
        return S, Tspread
    else:
        return binarySearchBoundary(G, k, Tspread, tsize, ic_step, p, iterations)

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

    tsize = 50
    S, Tsize = stopDegreeDiscount(G, tsize, ic_step=int(tsize*0.38*2/3))
    print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(S), tsize)
    print time.time() - start

    console = []
