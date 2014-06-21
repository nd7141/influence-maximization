''' Implementation of PMIA algorithm [1].

[1] -- Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks.
'''

from __future__ import division
import networkx as nx
import math

def updateAP(ap, v, S, PMIIA, Ep):
    ''' Assumption: PMIIA is a directed tree, which is a subgraph of general G.
    PMIIA is rooted at v.
    '''
    # Find leaves
    us = []
    for node in PMIIA:
        if not PMIIA.in_edges([node]):
            us.append(node)

    # Find ap moving from leaves to the root
    for u in us:
        if u in S:
            ap[(u, v)] = 1
        elif not PMIIA.in_edges([u]):
            ap[(u, v)] = 0
        else:
            in_edges = PMIIA.in_edges([u], data=True)
            prod = 1
            for w, _, edata in in_edges:
                prod *= 1 - ap[(w, v)]*(1 - (1 - Ep[(w, u)])**edata["weight"])
            ap[(u, v)] = 1 - prod

        out_edges = PMIIA.out_edges([u])
        us.extend([out_node for _, out_node in out_edges])

def updateAlpha(alpha, v, S, PMIIA, Ep, ap):
    ws = [v]
    alpha[(v, v)] = 1
    # moving from the root to leaves
    for w in ws:
        # add in-neighbors to ws
        in_neighbors = PMIIA.in_egdes([w])
        ws.extend([u for u, _ in in_neighbors])

        # calculate alpha for in-neighbors
        total_prod = 1
        for u, _ in in_neighbors:
            pp_up = (1 - Ep[(u, w)])**PMIIA[u][w]["weight"]
            total_prod *= (1 - ap[(u, v)])*pp_up

        for u, _ in in_neighbors:
            if w in S:
                alpha[(v, u)] = 0
            else:
                pp_u = (1 - Ep[(u,w)])**PMIIA[u][w]["weight"]
                # exclude u node from in_neighbors
                prod  = total_prod/((1 - ap[(u, v)])*pp_u)
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

    crossing_edges = set([in_edge for in_edge in G.in_edges([u]) if in_edge[0] not in S])
    edge_weights = dict()
    dist = {u: 0} # shortest paths from the root u

    # grow PMIOA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        for edge in crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
            edge_weight = edge_weights[edge]
            if dist[edge[1]] + edge_weight < min_dist:
                min_dist = dist[edge[1]] + edge_weight
                min_edge = edge
        # check stopping criteria
        print min_edge, ':', min_dist, '-->', -math.log(theta)
        if min_dist < -math.log(theta):
            dist[min_edge[0]] = min_dist
            PMIOA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            # update crossing edges
            crossing_edges.difference_update(G.out_edges(min_edge[0]))
            crossing_edges.update([in_edge for in_edge in G.in_edges(min_edge[0])
                                   if (in_edge[0] not in PMIOA) and (in_edge[0] not in S)])
        else:
            break
    return PMIOA


def PMIA(G, k, theta, Ep):
    # initialization
    S = []
    IncInf = dict(zip(G.nodes(), [0]*len(G)))
    PMIIA = dict()
    PMIOA = dict()
    ap = dict()
    alpha = dict()
    for v in G:
        PMIIA[v] = computePMIIA(v, theta, S)
        updateAlpha(alpha, v, S, PMIIA[v])
        for u in PMIIA[v]:
            ap[(u, v)] = 0 # ap of u node in PMIIA[v]
            IncInf[u] += alpha[(v,u)]*(1 - ap[(u, v)])

    # main loop
    for i in range(k):
        u, _ = max(IncInf.iteritems(), key = lambda (dk, dv): dv)
        IncInf.pop(u) # exclude node u for next iterations
        PMIOA[u] = computePMIOA(u, theta, S)
        for v in PMIOA[u]:
            for w in PMIIA[v]:
                if w not in S:
                    IncInf[w] -= alpha[(v,w)]*(1 - ap[(w, PMIIA[v])])

        S.append(u)

        for v in PMIOA[u]:
            if v != u:
                PMIIA[v] = computePMIIA(v, theta, S)
                updateAP(ap, v, S, PMIIA[v], Ep)
                updateAlpha(alpha, v, S, PMIIA[v])
                # add new incremental influence
                for w in PMIIA[v]:
                    if w not in S:
                        IncInf[w] += alpha[(v, w)]*(1 - ap[(w, v)])

    return S