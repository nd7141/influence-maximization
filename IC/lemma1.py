'''
Lemma 1: For IC, if |Inf(S)| < |T|, S = {max(d_v_i) | i = 1...k, v_i in T},
 then there is no other S' such that |Inf(S')| >= |T|.
'''
__author__ = 'sergey'

import random
import heapq

import networkx as nx

from IC.IC import runIC


def randomSet(T, k):
    return random.sample(T, k)

def highdegreeSet(G, T, k):
    # degree within T (don't consider nodes outside of T)
    d = dict()
    for u in T:
        for v in T:
            if v in G[u]:
                d[u] = d.get(u,0) + G[u][v]['weight']
    top = heapq.nlargest(k, d.iteritems(), key=lambda (key,value): value)
    print top
    Gd = dict()
    for u in T:
        for v in G[u]:
            Gd[u] = Gd.get(u,0) + G[u][v]['weight']
    topG = heapq.nlargest(k, Gd.iteritems(), key=lambda (key,value): value)
    print topG
    return [node for (node, value) in top]

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
                G.add_edge(u,v, weight=1)
    print 'Built graph G'
    print time.time() - start

    # # read in T
    # with open('lemma1.txt') as f:
    #     T = []
    #     k = int(f.readline())
    #     for line in f:
    #         T.append(int(line))
    # print 'Read %s activated nodes' %k
    # print time.time() - start
    S = [131, 639, 287, 267, 608, 100, 559, 124, 359, 66]
    k = len(S)
    T = runIC(G,S)

    highdegreeS = highdegreeSet(G,T,k)

    console = []
