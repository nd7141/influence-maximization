'''
Representative possible worlds that aims to preserve degree distribution.

Parchas et al. 2014 "The Pursuit of a Good Possible World"
http://www.francescobonchi.com/SIGMOD14-UG.pdf
'''
from __future__ import division
import networkx as nx
from itertools import cycle
import random

def GP (G, Ep):
    '''
    Greedy Probability algorithm
    Note: do not consider multigraphs.
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
                discrepancy[u] += 1
                discrepancy[v] += 1

        E = nx.Graph()
        E.add_nodes_from(G)
        E.add_edges_from(live_edges, weight=1)
        return E
    elif isinstance(G, nx.DiGraph):
        raise NotImplementedError
    else:
        raise NotImplementedError

def ADR (G, Ep, steps):
    '''
    Average degree rewiring algorithm
    Note: do not consider multigraphs.
    '''
    if isinstance(G, nx.Graph):
        E_edges = dict()
        for node in G:
            E_edges[node] = []
        P = round(sum(Ep.values()))
        selected = dict(zip(Ep.keys(), [False]*len(Ep)))
        number_of_edges = 0

        expected_degree = dict(zip(G.nodes(), [0]*len(G)))
        for e in Ep:
            expected_degree[e[0]] += Ep[e]

        # initialize discrepancy
        discrepancy = dict()
        for node, value in expected_degree.iteritems():
            discrepancy[node] = -value

        sorted_edges = cycle(sorted(Ep.iteritems(), key = lambda (e, p): p, reverse = True))

        # Phase 1: select first live edges
        while number_of_edges < P:
            e, _ = sorted_edges.next()
            if not selected[e] and random.random() < Ep[e]:
                E_edges[e[0]].append((e[0], e[1]))
                E_edges[e[1]].append((e[1], e[0]))
                discrepancy[e[0]] += 1
                discrepancy[e[1]] += 1
                selected[e] = True
                number_of_edges += 1
        blocked_edges = [e for e, value in selected.iteritems() if not value]

        # Phase 2: rewire live edges with blocked edges
        for it1 in range(steps):
            for node in G:
                try:
                    e1 = random.choice(E_edges[node])
                except IndexError:
                    continue
                e2 = random.choice(blocked_edges)
                u, v = e1
                dis_u = discrepancy[u]
                dis_v = discrepancy[v]
                x, y = e2
                dis_x = discrepancy[x]
                dis_y = discrepancy[y]
                d1 = abs(dis_u - 1) + abs(dis_v - 1) - (abs(dis_u) + abs(dis_v))
                d2 = abs(dis_x + 1) + abs(dis_y + 1) - (abs(dis_x) + abs(dis_y))
                if d1 + d2 < 0:
                    # update live edges
                    E_edges[u].remove((e1[0], e1[1]))
                    E_edges[v].remove((e1[1], e1[0]))
                    E_edges[x].append((e2[0], e2[1]))
                    E_edges[y].append((e2[1], e2[0]))
                    # update blocked edges
                    blocked_edges.append(e1)
                    blocked_edges.remove(e2)

        selected_edges = []
        for edges_list in E_edges.values():
            selected_edges.extend(edges_list)
        E = nx.Graph()
        E.add_nodes_from(G)
        E.add_edges_from(selected_edges, weight=1)
        return E
    elif isinstance(G, nx.DiGraph):
        raise NotImplementedError
    else:
        raise NotImplementedError



