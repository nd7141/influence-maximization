'''
File for Linear Threshold model (LT).
For directed graph G = (nodes, edges, weights) and
set of thresholds lambda for each node, LT model works as follows:
Initially set S of nodes is activated. For all outgoing neighbors,
we compare sum of edge weights from activated nodes and node thresholds.
If threshold becomes less than sum of edge weights for any given vertex,
then this vertex becomes active for the following iterations.
LT stops when no activation happens.

More on this: Kempe et al."Maximizing the spread of influence in a social network"
'''
from __future__ import division
import random
from copy import deepcopy
import networkx as nx

def uniformWeights(G):
    '''
    Every incoming edge of v with degree dv has weight 1/dv.
    '''
    Ew = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        dv = sum([edata['weight'] for v1,v2,edata in in_edges])
        for v1,v2,_ in in_edges:
            Ew[(v1,v2)] = 1/dv
    return Ew

def randomWeights(G):
    '''
    Every edge has random weight.
    After weights assigned,
    we normalize weights of all incoming edges so that they sum to 1.
    '''
    Ew = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        ew = [random.random() for e in in_edges] # random edge weights
        total = 0 # total sum of weights of incoming edges (for normalization)
        for num, (v1, v2, edata) in enumerate(in_edges):
            total += edata['weight']*ew[num]
        for num, (v1, v2, _) in enumerate(in_edges):
            Ew[(v1,v2)] = ew[num]/total
    return Ew

def checkLT(G, Ew, eps = 1e-4):
    ''' To verify that sum of all incoming weights <= 1
    '''
    for u in G:
        in_edges = G.in_edges([u], data=True)
        total = 0
        for (v1, v2, edata) in in_edges:
            total += Ew[(v1, v2)]*G[v1][v2]['weight']
        if total >= 1 + eps:
            return 'For node %s LT property is incorrect. Sum equals to %s' %(u, total)
    return True

def runLT(G, S, Ew):
    '''
    Input: G -- networkx directed graph
    S -- initial seed set of nodes
    Ew -- influence weights of edges
    NOTE: multiple k edges between nodes (u,v) are
    considered as one node with weight k. For this reason
    when u is activated the total weight of (u,v) = Ew[(u,v)]*k
    '''

    assert type(G) == nx.DiGraph, 'Graph G should be an instance of networkx.DiGraph'
    assert type(S) == list, 'Seed set S should be an instance of list'
    assert type(Ew) == dict, 'Infleunce edge weights Ew should be an instance of dict'

    T = deepcopy(S) # targeted set
    lv = dict() # threshold for nodes
    for u in G:
        lv[u] = random.random()

    W = dict(zip(G.nodes(), [0]*len(G))) # weighted number of activated in-neighbors
    Sj = deepcopy(S)
    # print 'Initial set', Sj
    while len(Sj): # while we have newly activated nodes
        Snew = []
        for u in Sj:
            for v in G[u]:
                if v not in T:
                    W[v] += Ew[(u,v)]*G[u][v]['weight']
                    if W[v] >= lv[v]:
                        # print 'Node %s is targeted' %v
                        Snew.append(v)
                        T.append(v)
        Sj = deepcopy(Snew)
    return T

def avgLT(G, S, Ew, iterations):
    avgSize = 0
    progress = 1
    for i in range(iterations):
        if i == round(iterations*.1*progress) - 1:
            print 10*progress, '% done'
            progress += 1
        T = runLT(G, S, Ew)
        avgSize += len(T)/iterations

    return avgSize