'''
Implementation of greedy algorithm for LT model.
'''

__author__ = 'sergey'

from LT import avgLT
import time

def generalGreedy(G, Ew, k, iterations=20):
    start = time.time()
    S = []
    for i in range(k):
        Inf = dict() # influence for nodes not in S
        for v in G:
            if v not in S:
                Inf[v] = avgLT(G, S + [v], Ew, iterations)
        u, val = max(Inf.iteritems(), key=lambda (k,val): val)
        print i, u
        S.append(u)
    return S