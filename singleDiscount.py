''' Implementation of single discount heuristic[1] for Independent Cascade model
of influence propagation in graph G

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
'''
__author__ = 'ivanovsergey'

from priorityQueue import PriorityQueue as PQ # priority queue

def singleDiscount(G, k, p=.1):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = [] # set of activated nodes
    d = PQ() # degrees
    for u in G:
        degree = sum([G[u][v]['weight'] for v in G[u]])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
        S.append(u)
        for v in G[u]:
            if v not in S:
                [priority, count, task] = d.entry_finder[v]
                d.add_task(v, priority + G[u][v]['weight']) # discount degree by the weight of the edge
    return S
