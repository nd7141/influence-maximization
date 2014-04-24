from __future__ import division
from priorityQueue import PriorityQueue as PQ
import networkx as nx
from copy import deepcopy

def FIND_LDAG(G, v, t, Ew):
    '''
    Compute local DAG for vertex v.
    Reference: W. Chen "Scalable Influence Maximization in Social Networks under LT model"
    INPUT:
        G -- networkx DiGraph object
        v -- vertex of G
        t -- parameter theta
        Ew -- influence weights of G
        NOTE: Since graph G can have multiple edges between u and v,
        total influence weight between u and v will be
        number of edges times influence weight of one edge.
    OUTPUT:
        D -- networkx DiGraph object that is also LDAG
    '''
    # intialize Influence of nodes
    Inf = PQ()
    Inf.add_task(v, -1)
    x, priority = Inf.pop_item()
    M = -priority
    X = [x]

    D = nx.DiGraph()
    while M >= t:
        # print M, t, x
        out_edges = G.out_edges([x], data=True)
        for (v1,v2,edata) in out_edges:
            if v2 in X:
                D.add_edge(v1, v2, edata)
        # D.add_edges_from(out_edges)
        in_edges = G.in_edges([x])
        for (u,_) in in_edges:
            if u not in X:
                try:
                    [pr, _, _] = Inf.entry_finder[u]
                except KeyError:
                    pr = 0
                Inf.add_task(u, pr - G[u][x]['weight']*Ew[(u,x)]*M)
        try:
            x, priority = Inf.pop_item()
        except KeyError:
            return D
        M = -priority
        X.append(x)

    return D

def tsort(D, u, Ru, reach):
    '''
     Topological sort of DAG D with vertex u first.
     NOTE: vertex u has no outgoing edges.
    '''
    Dc = deepcopy(D)
    L = [u]
    if reach == "in":
        for node in L:
            in_edges = Dc.in_edges([node], data=True)
            for (v1, v2, edata) in in_edges:
                assert v2 == node, 'Second node should be the same'
                Dc.remove_edge(v1, node)
                if not Dc.out_edges([v1]) and v1 in Ru:
                    L.append(v1)
    elif reach == "out":
        for node in L:
            out_edges = Dc.out_edges([node], data=True)
            for (v1, v2, edata) in out_edges:
                assert v1 == node, 'Second node should be the same'
                Dc.remove_edge(node, v2)
                if not Dc.in_edges([v2]) and v2 in Ru:
                    L.append(v2)
    if len(Dc.edges()):
        raise ValueError, 'D has cycles. No topological order.'
    return L

def DFS_reach (D, u, reach):
    ''' Depth-First search of nodes in D that can reach u.
    '''
    # initialize first nodes
    if reach == "in":
        Ru = map(lambda (v1,v2): v1, D.in_edges([u]))
    elif reach == "out":
        Ru = map(lambda (v1,v2): v1, D.out_edges([u]))
    for node in Ru:
        if reach == "in":
            reach_edges = map(lambda (v1,v2): v1, D.in_edges([node]))
        elif reach == "out":
            reach_edges = map(lambda (v1,v2): v1, D.out_edges([node]))
        for v1 in reach_edges:
            if v1 not in Ru:
                Ru.append(v1)
    return Ru

def computeAlpha(D, Ew, S, u):
    A = dict()
    A[u] = 1
    # compute nodes that can reach u in D
    Ru = DFS_reach(D, u, reach="in")
    order = tsort(D, u, Ru, reach="in")
    for node in order[1:]: # miss first node that already has computed Alpha
        A[node] = 0
        if node not in S + [u]:
            out_edges = D.out_edges([node], data=True)
            for (v1,v2, edata) in out_edges:
                assert v1 == node, 'First node should be the same'
                if v2 in order:
                    print v1, v2, edata, Ew[(node, v2)], A[v2]
                    A[node] += edata['weight']*Ew[(node, v2)]*A[v2]
    return A

# TODO implement computeActProb with DFS_reach and tsort routines
def computeActProb(D, Ew, S, u):
    pass