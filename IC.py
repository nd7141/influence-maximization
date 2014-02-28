''' Independent cascade model for influence propagation
'''
__author__ = 'ivanovsergey'

def runIC (G, S, p = .01):
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

def runIC2(G, S, p=.01):
    ''' Runs independent cascade model (finds levels of propagation).
    Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
    We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    import random
    T = deepcopy(S)
    Acur = deepcopy(S)
    Anext = []
    i = 0
    while Acur:
        values = dict()
        for u in Acur:
            for v in G[u].keys():
                if v not in T:
                    if random.random() < p:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        print i, Anext
        i += 1
        T.extend(Acur)
        Anext = []
    return T