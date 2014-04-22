'''
Implements newDegreeIC heuristic that stops after necessary amount of nodes is targeted.
Now it calculates spread of influence after a step increase in seed size
and returns if targeted set size is greater than desired input tsize.
'''
__author__ = 'ivanovsergey'

from copy import deepcopy # copy graph object
from random import random
import math

from IC.IC import runIC
from priorityQueue import PriorityQueue as PQ
from newGreedyIC import newGreedyIC


def binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations):
    # initialization for binary search

    R = iterations
    stepk = -int(math.ceil(float(step)/2))
    k += stepk
    if k not in Tsize:
        S = newGreedyIC(G, k, p)
        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R
        Tsize[k] = avg
    # check values of Tsize in between last 2 calculated steps
    while stepk != 1:
        print k, stepk, Tsize[k]
        if Tsize[k] >= targeted_size:
            stepk = -int(math.ceil(float(abs(stepk))/2))
        else:
            stepk = int(math.ceil(float(abs(stepk))/2))
        k += stepk

        if k not in Tsize:
            S = (G, k, p)
            avg = 0
            for i in range(R):
                T = runIC(G, S, p)
                avg += float(len(T))/R
            Tsize[k] = avg
    return S, Tsize

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

def spreadNewGreedyIC(G, targeted_size, step=1, p=.01, S0=[], iterations = 200):
    ''' Finds initial set of nodes to propagate in Independent Cascade.
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output: S -- set of k nodes chosen

    TODO: add step functionality
    '''

    import time
    start = time.time()

    assert type(S0) == list, "S0 must be a list. %s provided instead" % type(S0)
    S = S0 # set of selected nodes
    tsize = 0
    R = iterations
    for i in range(R):
        T = runIC(G, S, p)
        tsize += float(len(T))/R

    while tsize <= targeted_size:
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
        R = iterations # number of iterations to run RanCas
        for j in range(R):
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
                    [priority, c, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(bfs(E, [v])))/R)

            if idx == int(prg_idx*prcnt*R):
                print '%s%%...' %(int(prg_idx*prcnt*100))
                prg_idx += 1
            idx += 1
        # add vertex with maximum potential spread
        task, priority = s.pop_item()
        S.append(task)
        print i, len(S), task, -priority, time.time() - start

        tsize = 0
        for j in range(R):
            T = runIC(G, S, p)
            tsize += float(len(T))/R
    return S