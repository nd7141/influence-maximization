'''
Independent Arbitrary Cascade (IAC) is a independent cascade model with arbitrary
 propagation probabilities.
'''
from __future__ import division
from copy import deepcopy
import random
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


if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('../../graphdata/graph30.txt') as f:
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

    console = []
