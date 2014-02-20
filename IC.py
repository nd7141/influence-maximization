''' Independent cascade model for influence propagation
'''
__author__ = 'ivanovsergey'

def runIC (G, S, p = .1):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import copy
    from random import random
    T = copy(S)
    for u in T: # T may increase size during iterations
        for v in G[u].keys(): # check whether new node v is influenced by chosen node u
            if v not in T and random() < p:
                T.append(v)
    return T