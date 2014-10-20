from __future__ import division
import networkx as nx
import random

def make_possible_world(G, Ep):
    if isinstance(G, nx.Graph):
        live_edges = [e for e in G.edges_iter() if random.random() <= 1 - (1-Ep[e])**G[e[0]][e[1]]["weight"]]
        E = nx.Graph()
        E.add_nodes_from(G.nodes())
        E.add_edges_from(live_edges)
        return E
    elif isinstance(G, nx.DiGraph):
        raise NotImplementedError

def update_scores(E, k, scores):
    if isinstance(E, nx.Graph):
        # return connected components from largest to smallest
        connected_components = nx.connected_components(E)
        connected_components = sorted(connected_components, key=lambda cc: len(cc), reverse=True)
        # assign scores to first k+ties connected components
        for cc_idx, cc in enumerate(connected_components):
            if cc_idx < k:
                last_cc_size = len(cc)
                score = 1./last_cc_size
                for node in cc:
                    scores[node] += score
            else:
                new_cc_size = len(cc)
                if new_cc_size == last_cc_size:
                    score = 1./new_cc_size
                    for node in cc:
                        scores[node] += score
                else:
                    break
    elif isinstance(E, nx.DiGraph):
        raise NotImplementedError

def select_seeds(G, k ,Ep, scores):
    selected = dict(zip(G.nodes(), [False]*len(G)))
    S = []
    for i in range(k):
        node, score = max(scores.iteritems(), key=lambda(node, score): score)
        S.append(node)
        selected[node] = True
        scores.pop(node)
        # penalize out_nodes
        # note: out_node will be both in the case of directed and undirected graph (syntax of nx)
        for out_node in G[node]:
            if not selected[out_node]:
                scores[out_node] *= (1-Ep[(node, out_node)])**G[node][out_node]["weight"]
    return S

def Harvester(G, k, Ep, MC):
    scores = dict(zip(G.nodes(), [0]*len(G)))
    for it1 in range(MC):
        E = make_possible_world(G, Ep)
        update_scores(E, k, scores)
    seeds = select_seeds(G, k ,Ep, scores)
    return seeds
