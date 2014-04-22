from __future__ import division
import random
from copy import deepcopy

def uniformWeights(G):
    E = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        dv = sum([edata['weight'] for v1,v2,edata in in_edges])
        for v1,v2,_ in in_edges:
            E[(v1,v2)] = 1/dv
    return E

def randomWeights(G):
    E = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        ew = [random.random() for e in in_edges] # random edge weights
        total = 0 # total sum of weights of incoming edges (for normalization)
        for num, (v1, v2, edata) in enumerate(in_edges):
            total += edata['weight']*ew[num]
        for num, (v1, v2, _) in enumerate(in_edges):
            E[(v1,v2)] = ew[num]/total
    return E

def checkWC(G, E, eps = 1e-4):
    ''' To verify that sum of all incoming weights <= 1
    '''
    for u in G:
        in_edges = G.in_edges([u], data=True)
        total = 0
        for (v1, v2, edata) in in_edges:
            total += E[(v1, v2)]*G[v1][v2]['weight']
        if total >= 1 + eps:
            raise 'For node %s WC property is incorrect. Sum equals to %s' %(u, total)
    return 'Everything is correct'

def runWC(G, S, E):

    T = deepcopy(S) # targeted set
    lv = dict() # threshold for nodes
    for u in G:
        lv[u] = random.random()

    W = dict(zip(G.nodes(), [0]*len(G))) # weighted number of activated in-neighbors
    Sj = deepcopy(S)
    print 'Initial set', Sj
    while len(Sj):
        Snew = []
        for u in Sj:
            for v in G[u]:
                if v not in T:
                    W[v] += E[(u,v)]*G[u][v]['weight']
                    print 'For node', v, 'W[v] =', W[v], 'lv[v] =', lv[v]
                    if W[v] >= lv[v]:
                        Snew.append(v)
        T.extend(Snew)
        Sj = deepcopy(Snew)

    return T