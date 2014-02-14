__author__ = 'ivanovsergey'
import Graph # self-written graph class
import time
# to create and visualize graphs for tests
import networkx as nx
import matplotlib.pylab as plt

def degreeDiscountIC(G, k, p=.1):
    ''' Find seed set using degree discount independent cascade algorithm.
    G -- graph object
    k -- number of initial targeting set
    p -- propagation probability
    Output: S -- set of vertices of size k we want to target first
    '''
    import operator
    S = []
    d = dict() # degree for each vertex
    dd = dict() # degree discount for each vertex
    t = dict() # number of activated neighbors
    # initialize degree discount
    for u in G:
        d[u] = sum(u.connectedTo.values())
        dd[u] = d[u]
        t[u] = 0
    # select vertices to target
    for i in range(k):
        u, val = max(dd.iteritems(), key=operator.itemgetter(1))
        S.append(u)
        dd.pop(u)
        neighbors = u.connectedTo
        for v in neighbors:
            if v not in S:
                t[v] += u.getWeight(v.id)
                dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
    return S

def runIC(G, S, p=.1):
    ''' Runs independent cascade model in a graph.
    G -- graph object
    S -- initial set to target
    p -- propagation probability
    Output: T -- set of targeted nodes after running independent cascade
    '''
    import random
    T = map(lambda u: u.id, S) # targeted set of vertices
    for key in T:
        neighbors = G.getVertex(key).getConnections()
        for v in neighbors:
            if v.id not in T:
                if random.random() < p:
                    T.append(v.id)
    return T

if __name__ == '__main__':
    start = time.time()
    g = Graph.Graph()
    G = nx.Graph()
    # import pandas as pd
    # df = pd.read_csv('graphdata/hep.txt', sep=' ', header=None, skiprows=1)
    # for i in range(len(df)):
    #     g.addEdge(df['X.1'][i], df['X.2'][i])
    with open('./graphdata/graph30.txt') as f:
        n, m = map(int, f.readline().split())
        while True:
            edge = map(int, f.readline().split())
            if edge:
                g.addEdge(edge[0], edge[1])
                G.add_edge(edge[0], edge[1])
            else:
                break
    stop1 = time.time() - start
    print 'Finished reading file in ', stop1, 'sec'
    S = degreeDiscountIC(g, 10)
    stop2 = time.time() - start
    print 'Finished searching seed set in', stop2, 'sec'
    T = runIC(g, S)
    stop3 = time.time() - start
    print 'Finished running IC in', stop3, 'sec'
    print 'Resulting targeted set consists of', len(T), 'nodes out of', n

    console = []