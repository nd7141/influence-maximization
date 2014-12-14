'''
Representative possible worlds that aims to preserve degree distribution.

Parchas et al. 2014 "The Pursuit of a Good Possible World"
http://www.francescobonchi.com/SIGMOD14-UG.pdf
'''
from __future__ import division
import networkx as nx
from itertools import cycle
import random, copy, time
from priorityQueue import PriorityQueue

def avg(lst):
    if len(lst):
        return float(sum(lst))/len(lst)
    return 0

def GP (G, Ep):
    '''
    Greedy Probability algorithm
    Note: do not consider multigraphs.
    '''
    if type(G) == type(nx.Graph()):
        expected_degree = dict(zip(G.nodes(), [0]*len(G)))
        for e in Ep:
            # expected_degree[e[0]] += 1 - (1 - Ep[e])**G[e[0]][e[1]]["weight"]
            expected_degree[e[0]] += Ep[e]

        # initialize discrepancy
        discrepancy = dict()
        for node, value in expected_degree.iteritems():
            discrepancy[node] = -value

        live_edges = []
        # sorted_edges = sorted(G.edges(), key = lambda e: 1 - (1 - Ep[e])**G[e[0]][e[1]]["weight"], reverse = True)
        sorted_edges = sorted(G.edges(), key = lambda e: Ep[e], reverse = True)
        for e in sorted_edges:
            u, v = e
            dis_u = discrepancy[u]
            dis_v = discrepancy[v]
            if abs(dis_u + 1) + abs(dis_v + 1) < abs(dis_u) + abs(dis_v):
                live_edges.append(e)
                discrepancy[u] += 1
                discrepancy[v] += 1
        print 'Discrepancy after phase 1: %s' %sum(map(abs, discrepancy.values()))
        print 'MAE: %s' %avg(map(abs, discrepancy.values()))

        E = nx.Graph()
        E.add_nodes_from(G)
        E.add_edges_from(live_edges)
        return E
    elif type(G) == type(nx.DiGraph()):
        raise NotImplementedError
    else:
        raise NotImplementedError

def ADR(G, Ep, steps):
    '''
    Average degree rewiring algorithm
    Note: do not consider multigraphs.
    '''
    if type(G) == type(nx.Graph()):
        E_edges = dict()
        for node in G:
            E_edges[node] = []
        # P = round(sum([1 - (1-Ep[e])**G[e[0]][e[1]]["weight"] for e in G.edges_iter()]))
        P = round(sum([Ep[e] for e in G.edges_iter()]))
        selected = dict(zip(G.edges(), [False]*len(G.edges())))
        number_of_edges = 0

        expected_degree = dict(zip(G.nodes(), [0]*len(G)))
        for e in Ep:
            # expected_degree[e[0]] += 1 - (1 - Ep[e])**G[e[0]][e[1]]["weight"]
            expected_degree[e[0]] += Ep[e]

        # initialize discrepancy
        discrepancy = dict()
        for node, value in expected_degree.iteritems():
            discrepancy[node] = -value

        # sorted_edges = cycle(sorted(G.edges(), key = lambda e: 1 - (1 - Ep[e])**G[e[0]][e[1]]["weight"], reverse = True))
        sorted_edges = cycle(sorted(G.edges(), key = lambda e: Ep[e], reverse = True))

        # Phase 1: select first live edges
        while number_of_edges < P:
            # e, _ = sorted_edges.next()
            e = sorted_edges.next()
            # if not selected[e] and random.random() < 1 - (1 - Ep[e])**G[e[0]][e[1]]["weight"]:
            if not selected[e] and random.random() < Ep[e]:
                E_edges[e[0]].append((e[0], e[1]))
                E_edges[e[1]].append((e[1], e[0]))
                discrepancy[e[0]] += 1
                discrepancy[e[1]] += 1
                selected[e] = True
                number_of_edges += 1
        blocked_edges = [e for e, value in selected.iteritems() if not value]
        # print 'Discrepancy after phase 1: %s' %sum(map(abs, discrepancy.values()))

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
                    # update discrepancy
                    discrepancy[e1[0]] -= 1
                    discrepancy[e1[1]] -= 1
                    discrepancy[e2[0]] += 1
                    discrepancy[e2[1]] += 1
        # print 'Discrepancy after phase 2: %s' %sum(map(abs, discrepancy.values()))
        # print 'MAE: %s' %avg(map(abs, discrepancy.values()))

        selected_edges = []
        for edges_list in E_edges.values():
            selected_edges.extend(edges_list)
        E = nx.Graph()
        E.add_nodes_from(G)
        E.add_edges_from(selected_edges)
        return E
    elif type(G) == type(nx.DiGraph()):
        raise NotImplementedError
    else:
        raise NotImplementedError

def bipartite(w, discrepancy):
    Ebp_edges = []
    Q = PriorityQueue()
    incident_edges = dict()
    for e in w:
        Q.add_task(e, -w[e])
        incident_edges.setdefault(e[0], []).append((e[1], w[e]))
        incident_edges.setdefault(e[1], []).append((e[0], w[e]))
    processed_edges = []
    while len(processed_edges) < len(w):
        (e, weight) = Q.pop_item()
        processed_edges.append(e)
        try:
            incident_edges[e[0]].remove((e[1], -weight))
            incident_edges[e[1]].remove((e[0], -weight))
        except ValueError:
            pass
        Ebp_edges.append(e)

        # discard all edges in Q incident to b (i.e. e[1])
        for (a, weight) in incident_edges[e[1]]:
            try:
                Q.remove_task((a,e[1]))
                processed_edges.append((a,e[1]))
            except KeyError:
                pass
            try:
                incident_edges[a].remove((e[1], weight))
                incident_edges[e[1]].remove((a, weight))
            except ValueError:
                pass
        discrepancy[e[0]] += 1
        discrepancy[e[1]] += 1

        if -1 < discrepancy[e[0]] < .5:
            for (x, _) in incident_edges[e[0]]:
                try:
                    Q.remove_task((e[0], x))
                except KeyError:
                    pass
                new_weight = abs(discrepancy[e[0]]) + 2*abs(discrepancy[x]) - abs(discrepancy[e[0]]) - 1
                if new_weight > 0:
                    Q.add_task((e[0], x), -new_weight)
                else:
                    processed_edges.append((e[0], x))
        elif discrepancy[e[0]] > .5:
            for (x, _) in incident_edges[e[0]]:
                try:
                    Q.remove_task((e[0], x))
                    processed_edges.append((e[0], x))
                except KeyError:
                    pass
    return Ebp_edges

def ABM(G, Ep):
    if type(G) == type(nx.Graph()):
        # calculate expected degree
        expected_degree = dict(zip(G.nodes(), [0]*len(G)))
        for e in Ep:
            # expected_degree[e[0]] += 1 - (1 - Ep[e])**G[e[0]][e[1]]["weight"]
            expected_degree[e[0]] += Ep[e]

        current_degree = dict(zip(G.nodes(), [0]*len(G)))
        Em_edges = []
        Eprime_edges = []

        # Phase 1
        b = dict()
        for node in expected_degree:
            b[node] = round(expected_degree[node])

        unsorted_edges = G.edges()
        shuffled_idxs = range(len(G.edges()))
        random.shuffle(shuffled_idxs)
        shuffled_edges = [unsorted_edges[idx] for idx in shuffled_idxs]
        for e in shuffled_edges:
            if current_degree[e[0]] < b[e[0]] and current_degree[e[1]] < b[e[1]]:
                Em_edges.append(e)
                current_degree[e[0]] += 1
                current_degree[e[1]] += 1
            else:
                Eprime_edges.append(e)

        # Phase 2
        A = []
        B = []
        C = []
        discrepancy = dict()
        for u in G:
            discrepancy[u] = current_degree[u] - expected_degree[u]
            if discrepancy[u] < -.5:
                A.append(u)
            elif -.5 < discrepancy[u] < 0:
                B.append(u)
            else:
                C.append(u)
        # print 'Discrepancy after phase 1: %s' %sum(map(abs, discrepancy.values()))

        w = dict()
        for e in Eprime_edges:
            weight = abs(discrepancy[e[0]]) + 2*abs(discrepancy[e[1]]) - abs(1 + discrepancy[e[0]]) - 1
            if e[0] in A and e[1] in B and weight > 0:
                w[e] = weight

        Gprime = nx.Graph()
        Gprime.add_weighted_edges_from(map(lambda ((u, v), weight): (u,v,weight), w.items()))
        Ebp_edges = bipartite(w, discrepancy)
        # print 'Discrepancy after phase 2: %s' %sum(map(abs, discrepancy.values()))
        # print 'MAE: %s' %avg(map(abs, discrepancy.values()))

        E = nx.Graph()
        E.add_nodes_from(G)
        E.add_edges_from(Em_edges + Ebp_edges)
        return E
    elif type(G) == type(nx.DiGraph()):
        raise NotImplementedError
    else:
        raise NotImplementedError