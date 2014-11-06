from __future__ import division
import networkx as nx
import random

def make_possible_world(G, Ep):
    live_edges = [e for e in G.edges_iter() if random.random() <= 1 - (1-Ep[e])**G[e[0]][e[1]]["weight"]]
    if type(G) == type(nx.Graph()):
        E = nx.Graph()
    elif type(G) == type(nx.DiGraph()):
        E = nx.DiGraph()
    else:
        raise NotImplementedError
    E.add_nodes_from(G.nodes())
    E.add_edges_from(live_edges)
    return E

def find_multihop_neighbors(E):
    node_outhop_neighbors = {u: [] for u in E}
    node_inhop_neighbors = {u: [] for u in E}

    for node in E:
        out_edges = E.out_edges(node)
        i = 0
        while i < len(out_edges):
            e = out_edges[i]
            if e[1] not in node_outhop_neighbors[node] and e[1] != node:
                node_outhop_neighbors[node].append(e[1])
                node_inhop_neighbors[e[1]].append(node)
                out_edges.extend(E.out_edges(e[1]))
            i += 1
    return node_inhop_neighbors, node_outhop_neighbors

def avg(lst):
    if len(lst):
        return float(sum(lst))/len(lst)
    return 0

def avg2(lst):
    if len(lst):
        return float(sum(lst))/len(lst)
    return float('inf')

def min2(lst):
    if len(lst):
        return min(lst)
    return float('inf')

def max2(lst):
    if len(lst):
        return max(lst)
    return float('inf')

def update_scores(E, k, scores):
    if type(E) == type(nx.Graph()):
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
    elif type(E) == type(nx.DiGraph()):
        # find reach of incoming and outgoing nodes
        inhop_reach, outhop_reach = find_multihop_neighbors(E)
        # assign credit for each node (reverse to # of nodes that can reach this node)
        M  = dict()
        for node in E:
            Mi = len(inhop_reach[node])
            if Mi:
                M[node] = 1./Mi
            else:
                M[node] = 0
        # sort nodes by their reach
        sorted_outhop_reach = sorted(outhop_reach.iteritems(),
                                     key = lambda (_,outreach): len(outreach),
                                     reverse = True)
        # print sorted_outhop_reach[:2]
        marked_nodes = set()
        nonoveralpping_nodes = 0
        last_idx = -1
        # assign scores to nodes in sorted order until k mutually unreachable nodes get their scores
        for node, node_outhop_reach in sorted_outhop_reach:
            last_idx += 1
            score = sum([M[out_node] for out_node in node_outhop_reach])
            scores[node] += score
            if node not in marked_nodes:
                nonoveralpping_nodes += 1
                marked_nodes.update([node] + node_outhop_reach)
            if nonoveralpping_nodes == k:
                break
        else:
            print 'Number of nonoverlapping nodes is less than k.'
            print 'Assigned scores to all nodes.'
        # handle ties
        for node, reach in sorted_outhop_reach[last_idx+1:]:
            if len(reach) != len(node_outhop_reach):
                break
            else:
                score = sum([M[out_node] for out_node in reach])
                scores[node] += score
    else:
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
    '''
    Harvester function for influence maximization problem.
    Implemented for both undirected and directed case.
    '''

    # initialization
    scores = dict(zip(G.nodes(), [0]*len(G)))
    # run Monte-Carlo simulations to find scores
    for it1 in range(MC):
        E = make_possible_world(G, Ep)
        update_scores(E, k, scores)
    # select seeds
    seeds = select_seeds(G, k ,Ep, scores)
    return seeds
