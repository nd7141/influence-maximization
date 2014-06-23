''' Implementation of PMIA algorithm [1].

[1] -- Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks.
'''

from __future__ import division
import networkx as nx
import math

def updateAP(ap, v, S, PMIIAv, PMIIA_MIPv, Ep):
    ''' Assumption: PMIIAv is a directed tree, which is a subgraph of general G.
    PMIIAv is rooted at v.
    '''
    sorted_MIPs = sorted(PMIIA_MIPv.iteritems(), key = lambda (_, MIP): len(MIP), reverse = True)
    for u, _ in sorted_MIPs:
        if u in S:
            ap[(u, PMIIAv)] = 1
        elif not PMIIAv.in_edges([u]):
            ap[(u, PMIIAv)] = 0
        else:
            in_edges = PMIIAv.in_edges([u], data=True)
            prod = 1
            for w, _, edata in in_edges:
                p = (1 - (1 - Ep[(w, u)])**edata["weight"])
                prod *= 1 - ap[(w, PMIIAv)]*p
            ap[(u, PMIIAv)] = 1 - prod

def updateAlpha(alpha, v, S, PMIIAv, Ep, ap):
    ws = [v]
    alpha[(v, v)] = 1
    # moving from the root to leaves
    for w in ws:
        # add in-neighbors to ws
        in_neighbors = PMIIAv.in_edges([w])
        ws.extend([u for u, _ in in_neighbors if u not in ws])

        # calculate alpha for in-neighbors
        total_prod = 1
        for u, _ in in_neighbors:
            pp_up = 1 - (1 - Ep[(u, w)])**PMIIAv[u][w]["weight"]
            total_prod *= (1 - ap[(u, PMIIAv)])*pp_up

        for u, _ in in_neighbors:
            if w in S:
                alpha[(v, u)] = 0
            else:
                pp_u = 1 - (1 - Ep[(u,w)])**PMIIAv[u][w]["weight"]
                prod = 1
                for up, _ in in_neighbors:
                    if up != u:
                        pp_up = 1 - (1 - Ep[(up, w)])**PMIIAv[up][w]["weight"]
                        prod *= (1 - ap[(up, PMIIAv)])*pp_up
                alpha[(v, u)] = alpha[(v, w)]*pp_u*prod

def computePMIOA(G, u, theta, S, Ep):
    '''
     Compute PMIOA -- subgraph of G that's rooted at u.
     Uses Dijkstra's algorithm until length of path doesn't exceed -log(theta)
     or no more nodes can be reached.
    '''
    # initialize PMIOA
    PMIOA = nx.DiGraph()
    PMIOA.add_node(u)
    PMIOA_MIP = {u: [u]} # MIP(u,v) for v in PMIOA

    crossing_edges = set([out_edge for out_edge in G.out_edges([u]) if out_edge[0] not in S])
    edge_weights = dict()
    dist = {u: 0} # shortest paths from the root u

    # grow PMIOA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
            edge_weight = edge_weights[edge]
            if dist[edge[0]] + edge_weight < min_dist:
                min_dist = dist[edge[0]] + edge_weight
                min_edge = edge
        # check stopping criteria
        # print min_edge, ':', min_dist, '-->', -math.log(theta)
        if min_dist < -math.log(theta):
            dist[min_edge[1]] = min_dist
            PMIOA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIOA_MIP[min_edge[1]] = PMIOA_MIP[min_edge[0]] + [min_edge[1]]
            # update crossing edges
            crossing_edges.difference_update(G.in_edges(min_edge[1]))
            crossing_edges.update([out_edge for out_edge in G.out_edges(min_edge[1])
                                   if (out_edge[1] not in PMIOA) and (out_edge[1] not in S)])
        else:
            break
    return PMIOA, PMIOA_MIP

# TODO closely look at how it changes IS
def updateIS(IS, S, u, PMIOA, PMIIA):
    for v in PMIOA[u]:
        for si in S:
            # if seed node is effective and it's blocked by u
            # then it becomes ineffective
            if (si in PMIIA[v]) and (si not in IS[v]) and (u in PMIIA[v][si]):
                    IS[v].append(si)

def computePMIIA(G, ISv, v, theta, S, Ep):

    # initialize PMIIA
    PMIIA = nx.DiGraph()
    PMIIA.add_node(v)
    PMIIA_MIP = {v: [v]} # MIP(u,v) for u in PMIIA

    crossing_edges = set([in_edge for in_edge in G.in_edges([v]) if in_edge[0] not in ISv])
    edge_weights = dict()
    dist = {v: 0} # shortest paths from the root u

    # grow PMIIA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
            edge_weight = edge_weights[edge]
            if dist[edge[1]] + edge_weight < min_dist:
                min_dist = dist[edge[1]] + edge_weight
                min_edge = edge
        # check stopping criteria
        # print min_edge, ':', min_dist, '-->', -math.log(theta)
        if min_dist < -math.log(theta):
            dist[min_edge[0]] = min_dist
            PMIIA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIIA_MIP[min_edge[0]] = PMIIA_MIP[min_edge[1]] + [min_edge[0]]
            # update crossing edges
            crossing_edges.difference_update(G.out_edges(min_edge[0]))
            if min_edge[0] not in S:
                crossing_edges.update([in_edge for in_edge in G.in_edges(min_edge[0])
                                       if (in_edge[0] not in PMIIA) and (in_edge[0] not in ISv)])
        else:
            break
    return PMIIA, PMIIA_MIP

def PMIA(G, k, theta, Ep):
    import time
    start = time.time()
    # initialization
    S = []
    IncInf = dict(zip(G.nodes(), [0]*len(G)))
    PMIIA = dict() # node to tree
    PMIOA = dict()
    PMIIA_MIP = dict() # node to MIPs (dict)
    PMIOA_MIP = dict()
    ap = dict()
    alpha = dict()
    IS = dict()
    for v in G:
        IS[v] = []
        PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
        for u in PMIIA[v]:
            ap[(u, PMIIA[v])] = 0 # ap of u node in PMIIA[v]
        updateAlpha(alpha, v, S, PMIIA[v], Ep, ap)
        for u in PMIIA[v]:
            IncInf[u] += alpha[(v,u)]*(1 - ap[(u, PMIIA[v])])
            # print u, "-->", IncInf[u]
    print 'Finished initialization'
    print time.time() - start

    # main loop
    for i in range(k):
        u, _ = max(IncInf.iteritems(), key = lambda (dk, dv): dv)
        IncInf.pop(u) # exclude node u for next iterations
        PMIOA[u], PMIOA_MIP[u] = computePMIOA(G, u, theta, S, Ep)
        # print PMIOA_MIP[u]
        for v in PMIOA[u]:
            for w in PMIIA[v]:
                if w not in S + [u]:
                    IncInf[w] -= alpha[(v,w)]*(1 - ap[(w, PMIIA[v])])
        updateIS(IS, S, u, PMIOA_MIP, PMIIA_MIP)

        S.append(u)

        for v in PMIOA[u]:
            if v != u:
                PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
                updateAP(ap, v, S, PMIIA[v], PMIIA_MIP[v], Ep)
                updateAlpha(alpha, v, S, PMIIA[v], Ep, ap)
                # add new incremental influence
                for w in PMIIA[v]:
                    if w not in S:
                        IncInf[w] += alpha[(v, w)]*(1 - ap[(w, PMIIA[v])])

    return S