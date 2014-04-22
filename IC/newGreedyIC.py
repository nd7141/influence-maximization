''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)
'''
__author__ = 'ivanovsergey'

from copy import deepcopy # copy graph object
from random import random

from priorityQueue import PriorityQueue as PQ


def bfs(E, S):
    ''' Finds all vertices reachable from subset S in graph E using Breadth-First Search
    Input: E -- networkx graph object
    S -- list of initial vertices
    Output: Rs -- list of vertices reachable from S
    '''
    Rs = []
    for u in S:
        if u in E:
            if u not in Rs: Rs.append(u)
            for v in E[u].keys():
                if v not in Rs: Rs.append(v)
    return Rs

def newGreedyIC(G, k, p=.01, S0=[]):
    ''' Finds initial set of nodes to propagate in Independent Cascade.
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output: S -- set of k nodes chosen
    '''

    import time
    start = time.time()

    assert type(S0) == list, "S0 must be a list. %s provided instead" % type(S0)
    S = S0 # set of selected nodes
    if len(S) >= k:
        return S[:k]

    iterations = k - len(S)
    for i in range(iterations):
        s = PQ() # number of additional nodes each remained mode will bring to the set S in R iterations
        Rv = dict() # number of reachable nodes for node v
        # initialize values of s
        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0)
        # calculate potential additional spread for each vertex not in S
        prg_idx = 1
        idx = 1
        prcnt = .1 # for progress to print
        R = 200 # number of iterations to run RanCas
        # spread from each node individually in pruned graph E
        # Rv = dict()
        # for v in G:
        #     if v not in S:
        #         Rv[v] = 0
        for j in range(R):
            print j
            # create new pruned graph E
            E = deepcopy(G)
            edge_rem = [] # edges to remove
            for (u,v) in E.edges():
                w = G[u][v]['weight']
                if random() < 1 - (1 - p)**w:
                    edge_rem.append((u,v))
            E.remove_edges_from(edge_rem)
            # find reachable vertices from S
            Rs = bfs(E, S)
            # find additional nodes each vertex would bring to the set S
            for v in G.nodes():
                if v not in S + Rs: # if node has not chosen in S and has chosen by spread from S
                    # Rv[v] += float(len(bfs(E, [v])))/R
                    [priority, c, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(bfs(E, [v])))/R)

            if idx == int(prg_idx*prcnt*R):
                print '%s%%...' %(int(prg_idx*prcnt*100)), time.time() - start
                prg_idx += 1
            idx += 1
        # add spread of nodes in G'
        # for v in Rv:
        #     s.add_task(v, -Rv[v])
        # add vertex with maximum potential spread
        task, priority = s.pop_item()
        S.append(task)
        print i, k, task, -priority, time.time() - start
    return S