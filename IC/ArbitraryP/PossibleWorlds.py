'''
Representative possible worlds that aims to preserve degree distribution.

Parchas et al. 2014 "The Pursuit of a Good Possible World"
http://www.francescobonchi.com/SIGMOD14-UG.pdf
'''
from __future__ import division
import networkx as nx

def GP (G, Ep):
    '''
    Greedy Probability algorithm
    '''
    if isinstance(G, nx.Graph):
        expected_degree = dict(zip(G.nodes(), [0]*len(G)))
        for e in Ep:
            expected_degree[e[0]] += Ep[e]

        # initialize discrepancy
        discrepancy = dict()
        for node, value in expected_degree.iteritems():
            discrepancy[node] = -value

        live_edges = []
        sorted_edges = sorted(Ep.iteritems(), key = lambda (e, p): p, reverse= True)
        for e, _ in sorted_edges:
            u, v = e
            dis_u = discrepancy[u]
            dis_v = discrepancy[v]
            if abs(dis_u + 1) + abs(dis_v + 1) < abs(dis_u) + abs(dis_v):
                live_edges.append(e)

        E = nx.Graph()
        E.add_nodes_from(G)
        E.add_edges_from(live_edges)
        return E
    elif isinstance(G, nx.DiGraph):
        raise NotImplementedError
    else:
        raise NotImplementedError





