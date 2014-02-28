''' Implementation of degree heuristic[1] for Independent Cascade model
of influence propagation in graph G.
Takes k nodes with the largest degree.

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
'''
__author__ = 'ivanovsergey'
from priorityQueue import PriorityQueue as PQ # priority queue

def degreeHeuristic(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    d = PQ()
    for u in G:
        degree = sum([G[u][v]['weight'] for v in G[u]])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
        S.append(u)
    return S
