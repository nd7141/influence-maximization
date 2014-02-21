''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)
'''
__author__ = 'ivanovsergey'

from IC import runIC
from priorityQueue import PriorityQueue as PQ
from copy import deepcopy # copy graph object
from random import random

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

def newGreedyIC(G, k, p=.1):
    ''' Finds initial set of nodes to propagate in Independent Cascade.
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output: S -- set of k nodes chosen
    '''
    S = []
    R = 100 # number of iterations to run RanCas
    for i in range(k):
        s = PQ() # number of additional nodes each remained mode will bring to the set S in R iterations
        Rv = dict() # number of reachable nodes for node v
        # initialize values of s
        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0)
        # calculate potential additional spread for each vertex not in S
        for j in range(R):
            # create new pruned graph E
            E = deepcopy(G)
            edge_rem = [] # edges to remove
            for (u,v) in E.edges():
                if random() < 1 - p:
                    edge_rem.append((u,v))
            E.remove_edges_from(edge_rem)
            # find reachable vertices from S
            Rs = bfs(E, S)
            # find additional nodes each vertex would bring to the set S
            for v in G.nodes():
                if v not in S + Rs: # if node has not chosen in S and has chosen by spread from S
                    [priority, c, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(bfs(E, [v])))/R)
        # add vertex with maximum potential spread
        task, priority = s.pop_item()
        S.append(task)
    return S