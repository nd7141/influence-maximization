'''
Implementation of greedy algorithm for WC model.
'''

__author__ = 'sergey'



def generalGreedy(G, k):
    S = []
    for iteration in range(k):
        Inf = dict() # influence for nodes not in S
        for v in G:
            if v not in S:
                Inf[v] = avgWC(G, S + [v])
        u, val = max(Inf.iteritems(), key=lambda (k,val): val)
        S.append(u)
    return S