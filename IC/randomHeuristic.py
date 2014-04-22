''' Implementation of random heuristic[1] for Independent Cascade model
of influence propagation in graph G.
Takes k nodes uniformly at random

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
'''
__author__ = 'ivanovsergey'


def randomHeuristic(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    import random
    S = random.sample(G.nodes(), k)
    return S