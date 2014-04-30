from __future__ import division
from priorityQueue import PriorityQueue as PQ
import networkx as nx
from copy import deepcopy

# TODO write description for all functions and this script
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

def tsort(Dc, u, reach):
    '''
     Topological sort of DAG D with vertex u first.
     Note: procedure alters graph Dc (in the end no edges will be present)
    '''
    L = [u]
    if reach == "in":
        for node in L:
            in_nodes = map(lambda (v1, v2): v1, Dc.in_edges([node]))
            Dc.remove_edges_from(Dc.in_edges([node]))
            # print len(Dc.edges())
            for v in in_nodes:
                if len(Dc.out_edges([v])) <= 1: # for self loops number of out_edges is 1, for other nodes is 0
                    L.append(v)
    elif reach == "out": # the same just for outcoming edges
        for node in L:
            out_nodes = map(lambda (v1, v2): v2, Dc.out_edges([node]))
            Dc.remove_edges_from(Dc.out_edges([node]))
            for v in out_nodes:
                if len(Dc.in_edges([v])) <= 1:
                    L.append(v)
    if len(Dc.edges()):
        # print L
        # print Dc
        # print Dc.edges()
        raise ValueError, "D has cycles. No topological order."
    return L

def BFS_reach (D, u, reach):
    ''' Breadth-First search of nodes in D that can reach u.
    Input:
    reach == "in" -- nodes that can reach u
    reach == "out" -- nodes that are reachable from u
    '''
    Dc = nx.DiGraph()
    if reach == "in":
        Dc.add_edges_from(D.in_edges([u], data=True))
        in_nodes = map(lambda (v1,v2): v1, D.in_edges([u]))
        for node in in_nodes:
            Dc.add_edges_from(D.in_edges([node], data=True))
            in_nodes.extend(filter(lambda v: v not in in_nodes, map(lambda (v1,v2): v1, D.in_edges([node]))))
    elif reach == "out": # the same just for outcoming edges
        Dc.add_edges_from(D.out_edges([u], data=True))
        out_nodes = map(lambda (v1,v2): v2, D.out_edges([u]))
        for node in out_nodes:
            Dc.add_edges_from(D.out_edges([node], data=True))
            out_nodes.extend(filter(lambda v: v not in out_nodes, map(lambda (v1,v2): v2, D.out_edges([node]))))
    return Dc

def computeAlpha(D, Ew, S, u, val=1):
    A = dict()
    for node in D:
        A[(u,node)] = 0
    A[(u,u)] = val
    # compute nodes that can reach u in D
    Dc = BFS_reach(D, u, reach="in")
    order = tsort(Dc, u, reach="in")
    for node in order[1:]: # miss first node that already has computed Alpha
        if node not in S + [u]:
            out_edges = D.out_edges([node], data=True)
            for (v1,v2, edata) in out_edges:
                assert v1 == node, 'First node should be the same'
                if v2 in order:
                    # print v1, v2, edata, Ew[(node, v2)], A[v2]
                    A[(u,node)] += edata['weight']*Ew[(node, v2)]*A[(u,v2)]
    return A

def computeActProb(D, Ew, S, u, val=1):
    ap = dict()
    for node in D:
        ap[(u,node)] = 0
    ap[(u,u)] = val
    Dc = BFS_reach(D, u, "out")
    order = tsort(Dc, u, "out")
    for node in order:
        if node not in S + [u]:
            in_edges = D.in_edges([node], data=True)
            for (v1, v2, edata) in in_edges:
                assert v2 == node, 'Second node should be the same'
                if v1 in order:
                    ap[(u,node)] += ap[(u,v1)]*Ew[(v1, node)]*edata['weight']
    return ap

def updateInfSet (D, InfSet):
    '''
    Updates InfSet for nodes in D.
    '''
    for v in D:
        Dc = BFS_reach(D, v, "out")
        InfSet.setdefault(v, set([])).update(Dc.nodes())

# TODO check correctnes of LDAG heuristic comparing to results from Chen et al. figure 3
# TODO check for random edge weights Ewr
def LDAG_heuristic(G, Ew, k, t):
    S = []
    IncInf = PQ()
    for node in G:
        IncInf.add_task(node, 0)
    # IncInf = dict(zip(G.nodes(), [0]*len(G)))
    LDAGs = dict()
    InfSet = dict()
    ap = dict()
    A = dict()
    for v in G:
        LDAGs[v] = FIND_LDAG(G, v, t, Ew)
        for u in LDAGs[v]:
            InfSet.setdefault(u, []).append(v)
        # updateInfSet(LDAGs[v], InfSet)
        print v
        alpha = computeAlpha(LDAGs[v], Ew, S, v)
        A.update(alpha)
        for u in LDAGs[v]:
            ap[(v, u)] = 0
            priority, _, _ = IncInf.entry_finder[u]
            IncInf.add_task(u, priority - A[(v, u)])
            # IncInf[u] += A[(v, u)]

    for it in range(k):
        s, priority = IncInf.pop_item()
        print it, s, -priority
        for v in InfSet[s]:
            if v not in S:
                D = LDAGs[v]
                # update alpha_v_u for all u that can reach s in D
                alpha_v_s = A[(v,s)]
                dA = computeAlpha(D, Ew, S, s, val=-alpha_v_s)
                for (s,u) in dA:
                    if u not in S + [s]: # don't update IncInf if it's already in S
                        A[(v,u)] += dA[(s,u)]
                        # print u, IncInf.entry_finder[u]
                        priority, _, _ = IncInf.entry_finder[u]
                        IncInf.add_task(u, priority - dA[(s,u)]*(1 - ap[(v,u)]))
                # update ap_v_u for all u reachable from s in D
                dap = computeActProb(D, Ew, S + [s], s, val=1-ap[(v,s)])
                for (s,u) in dap:
                    if u not in S + [s]:
                        ap[(v,u)] += dap[(s,u)]
                        priority, _, _ = IncInf.entry_finder[u]
                        IncInf.add_task(u, priority + A[(v,u)]*dap[(s,u)])
        S.append(s)
    return S
