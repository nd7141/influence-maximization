'''
Implementation of Most Probable Spanning Tree (MPST).
Extract MPST and delete edges from original graph that belong to that MPST.
Continue until all edges are deleted from G.
Use those MPST to select a possible world (PW) and approximate reliability.

We are solving sparsification problem of uncertain graphs for preserving reliability.
'''
from __future__ import division
import networkx as nx
from itertools import cycle
from math import exp, log
import random, time, sys, json, os
from collections import Counter
from itertools import product, izip
import matplotlib.pyplot as plt
import matplotlib

def _comprehension_flatten(iter_lst):
    return [item for lst in iter_lst for item in lst]

def minimum_spanning_edges(G,weight='weight',data=True):
    from networkx.utils import UnionFind
    if G.is_directed():
        raise nx.NetworkXError(
            "Mimimum spanning tree not defined for directed graphs.")

    subtrees = UnionFind()
    edges = sorted(G.edges(data=True),key=lambda t: t[2].get(weight,1))
    for u,v,d in edges:
        if subtrees[u] != subtrees[v]:
            if data:
                yield (u,v,d)
            else:
                yield (u,v)
            subtrees.union(u,v)

# adopted MST algo without isolated nodes
# solution found here http://networkx.lanl.gov/_modules/networkx/algorithms/mst.html
def minimum_spanning_tree(G,weight='weight'):
    T=nx.Graph(nx.minimum_spanning_edges(G,weight=weight,data=True))
    return T

def get_PW(G):
    '''
    Expected to have -log(p_e) as a weight for every edge e in G.
    '''
    assert type(G) == type(nx.Graph()), "Graph should be undirected"

    P = sum([exp(1)**(-data["weight"]) for (u,v,data) in G.edges(data=True)]) # expected number of edges
    print 'P:', P
    E = G.copy()

    # extract multiple MPST until all edges in G are removed
    MPSTs = []
    while len(E.edges()):
        mpst = minimum_spanning_tree(E)
        print '|mpst|:', len(mpst), '|mpst.edges|:', len(mpst.edges())
        MPSTs.append(mpst)
        E.remove_edges_from(mpst.edges())
    print 'Found %s MPST' %(len(MPSTs))

    # sort edges
    sorted_edges = []
    for mpst in MPSTs:
        sorted_edges.extend(sorted(mpst.edges(data=True), key = lambda (u,v,data): exp(1)**(-data["weight"]), reverse=True))
    print "Sorted edges..."

    # create a PW
    selected = dict()
    for (u,v) in G.edges_iter():
        selected[(u,v)] = False
        selected[(v,u)] = False
    PW = nx.Graph()
    for (u,v,data) in cycle(sorted_edges):
        # add edge with probability p
        if not selected[(u,v)] and exp(1)**(-data["weight"]) > random.random():
            PW.add_edge(u,v,weight=0)
            selected[(u,v)] = True
            selected[(v,u)] = True
            # print "Added edge (%s,%s): P %s |PW.edges| %s" %(u,v,P,len(PW.edges()))
        # stop when expected number of edges reached
        if len(PW.edges()) > P:
            break
    print 'PW:', len(PW), 'PW.edges:', len(PW.edges())
    print 'G:', len(G), 'G.edges', len(G.edges())
    return PW, MPSTs

def find_reliability_in_mpst(mpst, u, v):
    '''
    Expected to have -log(p_e) as weights in each mpst.
    '''
    # find reliability along the path between u and v
    try:
        # paths = list(nx.all_simple_paths(mpst, u, v))
        path = nx.shortest_path(mpst, u, v, "weight")
    except (nx.NetworkXError, KeyError, nx.NetworkXNoPath):
        r = 0
    else:
        r = 1
        for ix in range(len(path)-1):
            weight = mpst[path[ix]][path[ix+1]]["weight"]
            r *= exp(1)**(-weight)
    return r

def get_rel_with_mpst(MPSTs, pairs=None):
    '''
    Compute estimates of reliability of every pair of nodes

    Expected to have -log(p_e) as weights in each mpst.

    if pairs = None finds reliability among all pairs,
    else only for those pairs specified.
    '''
    rel = dict()
    if pairs:
        print_dict = dict()
        for idx, pair in enumerate(pairs):
            u = pair[0]
            v = pair[1]
            for idx, mpst in enumerate(MPSTs):
                r = find_reliability_in_mpst(mpst, u, v)
                rel[(u,v)] = rel.get((u,v), 0) + r
                rel[(v,u)] = rel.get((v,u), 0) + r
                print_dict[pair] = rel[pair]
    else:
        for en, mpst in enumerate(MPSTs):
            print 'mpst #', en, "len(mpst):", len(mpst)
            nodes = mpst.nodes()
            for i in range(len(nodes)-1):
                for j in range(i+1, len(nodes)):
                    u = nodes[i]
                    v = nodes[j]
                    r = find_reliability_in_mpst(mpst, u, v)
                    rel[(u,v)] = rel.get((u,v), 0) + r
                    rel[(v,u)] = rel.get((v,u), 0) + r
    if pairs:
        print sorted(print_dict.items(), key = lambda (_,v): v, reverse = True)[:10]
    else:
        print sorted(rel.items(), key = lambda (_,v): v, reverse=True)[:10]
    return rel

def get_rel_with_mc(G, mc=100, pairs=None, cutoff_multiplier=2):
    '''
    Compute reliability between a pair of nodes using Monte-Carlo simulations

    Expected to have -log(p_e) as weights in G.

    if pairs = None finds reliability among all pairs,
    else only for those pairs specified (pairs (u,v) and (v,u) should not appear together).

    cutoff_multiplier of the number of MC simulations a pair should participate
    not to be pruned.
    '''
    print 'MC:', mc
    # initialize reliability
    rel = dict()
    for _ in range(mc):
        # create a random PW
        live_edges = [e for e in G.edges(data=True) if e[2]["weight"] < -log(random.random())]
        E = nx.Graph()
        E.add_edges_from(live_edges)
        if pairs != None:
            print_dict = dict()
            # for every provided pair of nodes check if there is a path
            for pair in pairs:
                u = pair[0]
                v = pair[1]
                r = 0
                if u in E and v in E and nx.has_path(E, u, v):
                    r = 1
                rel[(u,v)] = rel.get((u,v), 0) + r
                rel[(v,u)] = rel.get((v,u), 0) + r
                print_dict[pair] = rel[pair]
        else:
            # for every pair of nodes in E check if there is a path
            print_dict = dict()
            CCs = nx.connected_components(E)
            for cc in CCs:
                if len(cc) > 1:
                    for u in cc:
                        for v in cc:
                            rel[(u,v)] = rel.get((u,v), 0) + 1
                            if (u,v) not in print_dict or (v,u) not in print_dict:
                                print_dict[(u,v)] = rel[(u,v)]
    for key in rel:
        rel[key] = float(rel[key])/mc
        if key in print_dict:
            print_dict[key] /= mc
    if pairs:
        print sorted(print_dict.items(), key = lambda (k,v): v, reverse = True)[:10]
        cutoff_rel = dict(filter(lambda (k,v): v >= cutoff_multiplier/mc, rel.iteritems()))
        print 'Found %s non-zero rel out of %s pairs' %(len(filter(lambda v: v >= cutoff_multiplier/mc, print_dict.values())), len(print_dict))
    else:
        print sorted(rel.items(), key = lambda (k,v): v, reverse=True)[:10]
        cutoff_rel = dict(filter(lambda (k,v): v >= cutoff_multiplier/mc, rel.iteritems()))
        print 'Found %s non-zero rel out of %s pairs' %(len(filter(lambda v: v >= cutoff_multiplier/mc, print_dict.values())), len(print_dict))
    return cutoff_rel

def get_rel_for_pw(PW, pairs=None):
    '''
    if pairs = None finds reliability among all pairs,
    else only for those pairs specified.
    '''
    # print len(PW), len(PW.edges())
    rel = dict()
    if pairs:
        print_dict = dict()
        for pair in pairs:
            u = pair[0]
            v = pair[1]
            r = 0
            if u in PW and v in PW and nx.has_path(PW, u, v):
                r = 1
            rel[(u, v)] = r
            rel[(v, u)] = r
            print_dict[pair] = rel[pair]
    else:
        CCs = nx.connected_components(PW)
        for cc in CCs:
            cc_pairs = list(product(cc, repeat=2))
            rel.update(dict(zip(cc_pairs, [1]*len(cc_pairs))))
    if pairs:
        print sorted(print_dict.items(), key = lambda (k,v): v, reverse = True)[:10]
    else:
        print sorted(rel.items(), key = lambda (k,v): v, reverse=True)[:10]
    return rel

def get_objective(G_rel, PW_rel):
    '''
    Computes the objective, which is the sum of reliability discrepancies over all pairs of nodes.
    '''
    Obj = 0
    pairs = set(G_rel.keys() + PW_rel.keys())
    print 'Found %s pairs' %len(pairs)
    for p in pairs:
        if p[0] != p[1]:
            Obj += abs(G_rel.get(p, 0) - PW_rel.get(p, 0))
    return Obj, Obj/len(pairs)

def _make_pairs(G, l):
    '''
    Create l different pairs of nodes (u,v), where u != v
    '''
    pairs = []
    nodes = G.nodes()
    while len(pairs) < l:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and (u,v) not in pairs:
            pairs.append((u,v))
    return pairs

def get_sparsified_mpst(MPSTs, K):
    '''
    Get sparsified (uncertain graph with K edges) graph using MPST.

    MPSTs ordered from largest to smallest.

    Expected to have -log(p_e) as weights in G.
    '''

    # sort edges
    sorted_edges = []
    for mpst in MPSTs:
        if len(sorted_edges) + len(mpst.edges()) < K:
            sorted_edges.extend(mpst.edges(data = True))
        else:
            sorted_edges.extend(sorted(mpst.edges(data=True),
                                       key = lambda (u,v,data): exp(1)**(-data["weight"]),
                                       reverse=True))
            break
    return sorted_edges[:K]

def get_sparsified_greedy(G, K):
    '''
    Get sparsified (uncertain graph with K edges) graph choosing the most probable edges first.

    Expected to have -log(p_e) as weights in G.
    '''
    all_edges = G.edges(data=True)
    sorted_edges = sorted(all_edges, key = lambda (u,v,data): exp(1)**(-data["weight"]), reverse=True)
    print 'sorted edges...'
    # create a SP
    selected = dict()
    for (u,v,d) in all_edges:
        selected[(u,v)] = False
        selected[(v,u)] = False
    SP = nx.Graph()
    progress = 1
    for (u,v,data) in cycle(sorted_edges):
        # add edge with probability p
        if not selected[(u,v)] and exp(1)**(-data["weight"]) > random.random():
            SP.add_edge(u,v,data)
            selected[(u,v)] = True
            selected[(v,u)] = True
        # stop when expected number of edges reached
        if len(SP.edges()) == int(progress*.1*K):
            progress += 1
            print '%s%% processed...' %(progress*10)
        if len(SP.edges()) == K:
            break
    return SP

def get_sparsified_top(G, K):
    '''
    Get sparsified (uncertain graph with K edges) graph using top most probable edges.

    Expected to have -log(p_e) as weights in G.
    '''
    all_edges = G.edges(data=True)
    sorted_edges = sorted(all_edges,
                          key = lambda (u,v,data): exp(1)**(-data["weight"]),
                          reverse=True)
    return sorted_edges[:K]

def get_sparsified_random(G, K):
    '''
    Get sparsified (uncertain graph with K edges) graph using random edges.

    Expected to have -log(p_e) as weights in G.
    '''
    all_edges = G.edges(data=True)
    random.shuffle(all_edges)
    random_edges = random.sample(all_edges, K)
    return random_edges[:K]

def get_sparsified_top2(G, K, edge_rel=None):
    '''
    Get sparsified (uncertain graph with K edges) graph choosing edges
    based on ascending order of (1 - rel)*p/(1-p)

    Expected to have -log(p_e) as weights in G.
    '''
    all_edges = G.edges(data = True)
    if edge_rel == None:
        G_rel = get_rel_with_mc(G, MC, pairs=G.edges(), cutoff_multiplier= 0)
    else:
        G_rel = edge_rel
    sorted_edges = sorted(all_edges,
                          key = lambda (u,v,d): exp(1)**(-d["weight"])*(1 - G_rel[(u,v)])/(1 - exp(1)**(-d["weight"])),
                          reverse = True)
    return sorted_edges[:K]

def get_sparsified_top3(G, K, edge_rel=None):
    '''
    Get sparsified (uncertain graph with K edges) graph choosing edges
    based on descending order of (1 - rel)*p/(1-p)

    Expected to have -log(p_e) as weights in G.
    '''
    all_edges = G.edges(data = True)
    if edge_rel == None:
        G_rel = get_rel_with_mc(G, MC, pairs=G.edges(), cutoff_multiplier= 0)
    else:
        G_rel = edge_rel
    sorted_edges = sorted(all_edges,
                          key = lambda (u,v,d): exp(1)**(-d["weight"])*(1 - G_rel[(u,v)])/(1 - exp(1)**(-d["weight"])),
                          reverse = False)
    return sorted_edges[:K]

def get_sparsified_APSP(G, K, cutoff=None):
    '''
    Get sparsified (uncertain graph with K edges) graph choosing edges
    based on ascending order of sp*p, where sp is the the number of shortest path of an edge

    Expected to have -log(p_e) as weights in G.
    '''
    print 'Start APSP...'
    time2APSP = time.time()
    paths = {}
    for i, n in enumerate(G):
        print i, len(G)
        paths[n] = nx.single_source_dijkstra_path(G, n, cutoff=cutoff,
                                               weight="weight")
    print 'Finish APSP in %s sec' %(time.time() - time2APSP)
    print paths

    checked = dict()
    edges_score = dict()
    for u in paths:
        u_paths = paths[u]
        for v in u_paths:
            # print 'Pair %s %s' %(u,v)
            if u != v and not checked.get((v,u), False):
                path = u_paths[v]
                for i in range(len(path) - 1):
                    e1, e2 = path[i], path[i+1]
                    # print 'Add to edge %s %s' %(e1, e2)
                    edges_score[(e1, e2)] = edges_score.get((e1, e2), 0) + 1
                    edges_score[(e2, e1)] = edges_score.get((e2, e1), 0) + 1
                checked[(u,v)] = True
                checked[(v,u)] = True
    print edges_score
    sorted_edges = sorted(G.edges(data=True),
                          key = lambda (u,v,d): exp(1)**(-d["weight"])*edges_score[(u,v)],
                          reverse = True)

    return sorted_edges[:K]

def get_sparsified_MP_MPST(G, K, directed=False):
    '''
    Sparsify graph by first finding minimum spanning tree,
    then adding most probable edges.
    '''
    G_edges = G.edges(data=True)
    if directed:
        # MPST_edges = branchings.minimum_spanning_arborescence(G, attr='weight').edges(data=True)
        pass
    else:
        MPST_edges = list(nx.minimum_spanning_edges(G,weight='weight',data=True))
    edges = [e for e in G_edges if e not in MPST_edges]
    mp_edges = sorted(edges,
                    key = lambda (u,v,d): exp(1)**(-d["weight"]),
                    reverse = True)
    if len(MPST_edges) <= K:
        MPST_edges.extend(mp_edges[:(K - len(MPST_edges))])
    else:
        # remove edges that are adjacent to leaves (keeping connectivity)
        # if ties remove with lowest probability (keeping probability)
        #TODO check why in case of directed MPST it doesn't work
        MPST = nx.Graph(MPST_edges)
        degrees = dict()
        leaves = set()
        for u in MPST:
            degrees[u] = len(MPST[u])
            if degrees[u] == 1:
                v, d = MPST[u].items()[0]
                leaves.add((u,v,d["weight"]))
        for _ in range(len(MPST_edges) - K):
            u,v,d = min(leaves, key = lambda (u,v,d): exp(1)**(-d))
            MPST.remove_edge(u,v)
            leaves.remove((u,v,d))
            v_edges = MPST[v].items()
            if len(v_edges) == 1:
                w, t = v_edges[0]
                leaves.add((v,w,t["weight"]))
            elif len(v_edges) == 0:
                leaves.remove((v,u,d))
        print len(MPST.edges()), K
        MPST_edges = MPST.edges(data=True)

        # MPST_edges = sorted(MPST_edges,
        #                     key = lambda (u,v,d): exp(1)**(-d["weight"]),
        #                     reverse = True)
        # MPST_edges = MPST_edges[:K]
    return MPST_edges

def get_sparsified_MPSTplus(G, K):
    '''
    Sparsify graph using most probable spanning tree.
    If K < |MPST|, then add most probable edges that are not included.
    If K > |MPST|, then remove edges based on the number of neighbors
    for endpoints of the edges.

    :param G: undirected graph with -log(p_e) on edges
    :param K: number of edges to preserve
    :return: edges with probabilities -log(p_e) of size K
    '''
    G_edges = G.edges(data=True)
    print 'Finding MPST'
    MPST_edges = list(nx.minimum_spanning_edges(G,weight='weight',data=True))
    print 'Found spanning tree'
    if len(MPST_edges) <= K:
        print 'Start sorting remaining edges'
        edges = [e for e in G_edges if e not in MPST_edges]
        mp_edges = sorted(edges,
                    key = lambda (u,v,d): exp(1)**(-d["weight"]),
                    reverse = True)
        print 'Finished sorting edges'
        MPST_edges.extend(mp_edges[:(K - len(MPST_edges))])
    else:
        #remove edges that will not make isolated vertices

        MPST = nx.Graph()
        MPST.add_edges_from(MPST_edges)
        # edges where both endpoints have at least 2 edges
        power1 = {tuple(sorted((e[0],e[1]))): len(MPST[e[0]]) + len(MPST[e[1]]) - 2 for e in MPST.edges() if len(MPST[e[0]]) != 1 and len(MPST[e[1]]) != 1}
        # edges where at least one vertex has only one edge
        power2 = {tuple(sorted((e[0],e[1]))): len(MPST[e[0]]) + len(MPST[e[1]]) - 2 for e in MPST.edges() if len(MPST[e[0]]) == 1 or len(MPST[e[1]]) == 1}

        def _decrease_power(MPST, endpoint, power1, power2):
            for node in MPST[endpoint]:
                edge = tuple(sorted((endpoint, node)))
                if edge in power1:
                    power1[edge] -= 1
                elif edge in power2:
                    power2[edge] -= 1
                else:
                    raise ValueError, "Edge %s should be in MPST" %edge

        while len(MPST.edges()) > K:
            if len(power1) > 0:
                e, val = max(power1.iteritems(), key = lambda(dk, dv): dv)
                # print e, val
                power1.pop(e)
                MPST.remove_edge(e[0],e[1])
                _decrease_power(MPST, e[0], power1, power2)
                _decrease_power(MPST, e[1], power1, power2)
            elif len(power2) > 0:
                e, val = max(power2.iteritems(), key = lambda(dk, dv): dv)
                # print e, val
                power2.pop(e)
                MPST.remove_edge(e[0],e[1])
                _decrease_power(MPST, e[0], power1, power2)
                _decrease_power(MPST, e[1], power1, power2)
            else:
                raise ValueError, "No more edges"

        MPST_edges = MPST.edges(data=True)
    return MPST_edges

def get_edges_greedy_min(G,K):
    '''
    Selects an edge based on max loss of objective. Assigns mean of degrees of endpoints.
    :param G: graph
    :param K: number of edges
    :return:
    '''
    # calculate expected degrees
    d = dict()
    for v in G:
        for u in G[v]:
            d[v] = d.get(v, 0) + exp(1)**(-G[v][u]['weight'])
    # initialize current degree
    cur_d = {v: 0 for v in G}

    edges = dict()

    while len(edges) < K:
        max_prob = 0
        max_edge = None
        for e in G.edges():
            edge = tuple(sorted(e))
            if edge not in edges:
                e_prob = min((abs(d[edge[0]] - cur_d[edge[0]]) + abs(d[edge[1]] - cur_d[edge[1]]))/2, 1)
                if e_prob >= max_prob:
                    max_prob = e_prob
                    max_edge = edge
        if max_edge == None:
            raise ValueError, "Cannot find an edge"
        print max_edge, max_prob
        edges[max_edge] = max_prob
        cur_d[max_edge[0]] += max_prob
        cur_d[max_edge[1]] += max_prob

    return edges

def ChungLu(G):
    '''
    Compute Chung-Lu probabilities.
    Uses expected degrees of G as weights.

    Expected to have -log(p_e) as weights in G.
    '''
    w = dict() # expected degrees of nodes
    for u in G:
        for v in G[u]:
            p = exp(1)**(-G[u][v]["weight"])
            w[u] = w.get(u, 0) + p
    W = sum(w.values())
    CL = dict() # Chung-Lu probabilities
    for u in G:
        for v in G:
            CL[(u,v)] = w[u]*w[v]/W
            if u == v:
                CL[(u,v)] /= 2
    return CL

def exp_degrees(G):
    '''
    Expected to have -log(p_e) as weights in G.
    :param G:
    :return:
    '''
    in_d = dict(zip(G, [0]*len(G)))
    out_d = dict(zip(G, [0]*len(G)))
    for v in G:
        for u in G[v]:
            out_d[v] += exp(1)**(-G[v][u]['weight'])
            in_d[u] += exp(1)**(-G[v][u]['weight'])
    return in_d, out_d

def pres_dir_degrees_ChungLu(edges, degrees, dir="in"):
    '''
    Preserve directed degrees
    :param edges: directed edges
    :param in_degrees: dictionary of expected in_degrees
    :return: Q -- graph from edges with preserved in_degrees

    Note the problem that sometimes probabilities = 0 or > 1.
    '''
    Q = nx.DiGraph()
    Q.add_edges_from(edges)
    if dir == "in":
        denom = {v: sum([degrees[u] for (u,_) in Q.in_edges(v)]) for v in Q}
    elif dir == "out":
        denom = {v: sum([degrees[u] for (_,u) in Q.out_edges(v)]) for v in Q}

    assert denom > 0
    for e in edges:
        if dir == "in":
            sum_d = denom[e[1]]
        elif dir == "out":
            sum_d = denom[e[0]]
        if degrees[e[0]]*degrees[e[1]] > sum_d:
            print '(',e[0],e[1],') -->', degrees[e[0]]*degrees[e[1]], sum_d
        print '(',e[0],e[1],') -->', degrees[e[0]], degrees[e[1]], sum_d
        p = degrees[e[0]]*degrees[e[1]]/sum_d
        print p
        Q.add_edge(*e, **{"weight": -log(p)})
    return Q

def pres_dir_degrees_equally(edges, degrees, dir="in"):
    '''
    Preserve directed degrees
    :param edges: directed edges
    :param degrees: dictionary of expected in_degrees
    :return: Q -- graph from edges with preserved in_degrees
    '''
    Q = nx.DiGraph()
    Q.add_edges_from(edges)
    for e in edges:
        if dir == "in":
            p = min(degrees[e[1]]/len(Q.in_edges(e[1])), 1)
        elif dir == "out":
            p = min(degrees[e[0]]/len(Q.out_edges(e[0])), 1)
        Q.add_edge(e[0],e[1],weight=-log(p))
    return Q


def get_undirected_prob(Q):
    '''

    :param Q: directed graph
    :return: undirected graph where probabiltiies is some function of previous probabilities
    '''
    G = nx.Graph()
    for e in Q.edges(data=True):
        u,v,log_p1 = e[0],e[1],e[2]['weight']
        log_p2 = Q[v][u]['weight']
        p = exp(1)**(-log_p1) + exp(1)**(-log_p2)
        G.add_edge(u,v,{'weight': -log(p/2)})
    return G

def get_graph_from_file(filename, directed=False):
    SP = nx.Graph()
    if directed:
        SP = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            u, v, p = map(float, line.split())
            if u != v:
                SP.add_edge(int(u), int(v), weight = -log(p))
    return SP

def test_MC(G, pair, gt, MC_limit, chunks=10):
    '''
    Outputs the error of reliability of a pair in the graph computed by MC
    comparing to ground-truth
    '''
    errors = []
    mcs = [1 + i*int(MC_limit/chunks) for i in range(chunks+1)]
    print mcs
    rel = 0
    u, v = pair
    for mc in range(1, MC_limit+2):
        # create a random PW
        live_edges = [(e[0], e[1], e[2]) for e in G.edges(data=True) if e[2]["weight"] < -log(random.random())]
        E = nx.Graph()
        E.add_edges_from(live_edges)

        if u in E and v in E and nx.has_path(E, u, v):
            rel += 1
        if mc in mcs:
            errors.append(abs(gt - rel/mc))
    return errors

def get_redistributed(G, selected_edges):
    extra_prob = dict() # extra probability for each node
    new_degree = dict() # degree of each node in sparsified graph
    unweighted_selected_edges = [(u,v) for (u,v,d) in selected_edges]
    for (u,v,d) in G.edges(data=True):
        # if we remove an edge, then give a half portion to each node
        if (u,v) not in unweighted_selected_edges and (v,u) not in unweighted_selected_edges:
            p = exp(1)**(-d["weight"])
            # print p
            extra_prob[u] = extra_prob.get(u, 0) + p/2
            extra_prob[v] = extra_prob.get(v, 0) + p/2
        # else increase degree of each node by one
        else:
            new_degree[u] = new_degree.get(u, 0) + 1
            new_degree[v] = new_degree.get(v, 0) + 1

    new_edges = []
    for (u,v,d) in selected_edges:
        p = exp(1)**(-d["weight"])
        assert new_degree[u] > 0
        assert new_degree[v] > 0
        # print extra_prob.get(u, 0)/new_degree[u], extra_prob.get(v, 0)/new_degree[v]
        new_p = p + extra_prob.get(u, 0)/new_degree[u] + extra_prob.get(v, 0)/new_degree[v]
        new_edges.append((u, v, -log(min(new_p, 1))))

    RD = nx.Graph()
    RD.add_weighted_edges_from(new_edges)
    return RD

def save_for_LP(f1, f2, G, G_orig, f3 = "edge_order.txt", f4 = "D.txt"):
    '''
    Saves matrix A and vector b to files f1 and f2,
    for linear programming min||Ax - b||, s.t. 0 <= x <= 1 in MATLAB.

    Save edge order to f3,
    sum of probabilities of sparsified edges to f4.

    Expected to have -log(p_e) as weights in G.
    '''

    edge_order = dict()
    for i, e in enumerate(G.edges()):
        edge_order[(e[0],e[1])] = i + 1
    node_order = dict()
    for i, u in enumerate(G):
        node_order[u] = i + 1
    wi = dict()
    for u in G:
        wi[u] = sum([exp(1)**(-G[u][v]["weight"]) for v in G[u]])

    with open(f1, "w+") as f:
        for e in G.edges():
            e1, e2 = e[0], e[1]
            f.write("%s %s %s\n" %(node_order[e1], edge_order[(e1, e2)], 1))
            f.write("%s %s %s\n" %(node_order[e2], edge_order[(e1, e2)], 1))
    with open(f2, "w+") as f:
        for u in G:
            f.write("%s %s %s\n" %(node_order[u], 1, wi[u]))
    with open(f3, "w+") as f:
        for e in G.edges():
            f.write("%s %s %s\n" %(e[0], e[1], edge_order[(e[0],e[1])]))

    nodes = set(G_orig).difference(G)
    print "# nodes: ", len(nodes),
    d = dict() # degree of remaining nodes
    for u in nodes:
        d[u] = 0
        for v in G_orig[u]:
            p = exp(1)**(-G_orig[u][v]["weight"])
            d[u] += p
    surplus = sum(d.values())
    print "surplus", surplus
    with open(f4, "w+") as f:
        f.write('%s\n' %len(G_orig))
        f.write('%s\n' %surplus)

def save_for_LP_dir (Ainf, Aoutf, dinf, doutf, Q, G, e_ordf, D):
    '''
    Saves matrices for linear program max|x| s.t. din > Ain*x, dout > Aout*x
    :param Ainf: filenames
    :param Aoutf:
    :param dinf:
    :param doutf:
    :param Q: sparsified directed graph
    e_ord: edges for which probabilities should be found
    D: number of nodes and surplus for sparsified nodes
    G: original graph
    :return:
    '''

    assert type(Q) == type(nx.DiGraph())
    edges = Q.edges()

    # calculate in_degree, out_degree of u
    win = dict()
    wout = dict()
    for u in G:
        win[u] = 0
        wout[u] = 0
        for (v,_) in G.in_edges(u):
            p = exp(1)**(-G[v][u]['weight'])
            win[u] += p
        for (_,v) in G.out_edges(u):
            p = exp(1)**(-G[u][v]['weight'])
            wout[u] += p

    # save order of edges
    e_ord = dict() # order of edges
    v_ord = dict() # order of nodes
    count = 1
    for e in edges:
        e_ord[(e[0],e[1])] = count
        e_ord[(e[1],e[0])] = count
        count += 1
    count = 1
    for u in Q:
        v_ord[u] = count
        count += 1
    with open(e_ordf, "w+") as f:
        for e in edges:
            f.write("%s %s\n" %(e[0], e[1]))

    # save Ain and Aout
    nodes = set()
    with open(Ainf, "w+") as f1, open(Aoutf, "w+") as f2:
        for e in edges:
            f1.write("%s %s %s\n" %(v_ord[e[1]], e_ord[e], 1))
            f2.write("%s %s %s\n" %(v_ord[e[0]], e_ord[e], 1))
            nodes.add(e[0])
            nodes.add(e[1])

    sp_n = len(nodes)
    sp_m = len(edges)
    print len(nodes), len(edges)

    # save din and dout
    with open(dinf, "w+") as f1, open(doutf, "w+") as f2:
        for i, u in enumerate(nodes):
            if i == 54307:
                print u, v_ord[u], win[u], wout[u]
            f1.write("%s %s %s\n" %(v_ord[u], 1, win[u]))
            f2.write("%s %s %s\n" %(v_ord[u], 1, wout[u]))

    # save number of nodes and surplus
    n = len(G)
    nodes = set(G).difference(Q)
    print "# nodes: ", len(nodes),
    in_surplus = sum([win[u] for u in nodes])
    out_surplus = sum([wout[u] for u in nodes])
    print "in-degree surplus", in_surplus
    print "out-degree surplus", out_surplus
    with open(D, "w+") as f:
        f.write("%s\n" %(n))
        f.write("%s\n" %(sp_n))
        f.write("%s\n" %(sp_m))
        f.write("%s\n" %(in_surplus + out_surplus))

def construct_lp_graph(lp_file, e_file, output_file):
    with open(lp_file) as f, open(e_file) as g, open(output_file, "w+") as h:
        for x, y in izip(f,g):
            d1 = x.split()
            d2 = y.split()
            h.write("%s %s %s\n" %(d2[0], d2[1], d1[0]))

def get_possible_worlds(graph_file, PW_folder, I):
    '''
    Creates possible worlds from the graph file
    :param graph_file: file of edges with probabilities
    :param PW_folder: folder where to write PWs
    :param I: number of possible worlds
    :return:
    '''
    with open(graph_file) as f:
        f_lines = f.readlines()
    for i in range(I):
        with open(PW_folder + "PW%s.txt" %(i+1), "w+") as f:
            for line in f_lines:
                d = map(float, line.split())
                if random.random() < d[2]:
                    f.write(line)



if __name__ == "__main__":
    time2execute = time.time()

    # time2read = time.time()
    # datasets = "datasets/"
    # flickr = datasets + "Flickr.txt_reduced-FF_5000.txt"
    # # flixster = "Flixster/" + "flixster3.txt"
    # G = get_graph_from_file(flickr, False)
    # print "Read graph in % sec" %(time.time() - time2read)
    # print "G: n = %s m = %s" %(len(G), len(G.edges()))
    # print

    # for i in range(1,11):
    #     start = time.time()
    #     edges = get_sparsified_MP_MPST(G, int(i*len(G.edges())/10))
    #     Q = nx.DiGraph()
    #     Q.add_edges_from(edges)
    #     save_for_LP("Flickr/LP/A%s.dat" %i, "Flickr/LP/d%s.dat" %i, Q, G, "Flickr/LP/e%s.dat" %i, "Flickr/LP/D%s.dat" %i,)
    #     print 'Spent %s sec for %i iteration' %(time.time() - start, i)

    # rewrite file
    with open("Flickr/runtime/runtime.txt", "w") as f:
        pass
    for i in range(1,11):
        print i
        graph_file = "Flickr/LP/G%i.txt" %(i*10)
        PW_folder = "Flickr/PW/G%i/" %(i*10)
        if not os.path.exists(PW_folder):
            os.makedirs(PW_folder)
        time2PW_start = time.time()
        get_possible_worlds(graph_file, PW_folder, 100)
        time2PW_finish = time.time() - time2PW_start
        print 'Created worlds in %s sec' %(time2PW_finish)
        with open("Flickr/runtime/runtime.txt", 'a+') as f:
            f.write("%s\n" %(time2PW_finish))

    time2pairs = time.time()
    # G_rel_mc = dict()
    # l = 10000
    # pairs = _make_pairs(G, l)
    # print 'Made %s pairs in %s sec' %(l, time.time() - time2pairs)
    # pairs = [(11052, 11371), (9097, 10655)]
    # pairs = [(11052, 11371)]
    # print pairs
    # edges = G.edges()
    # pairs = set(edges + pairs)
    # print len(pairs)
    # pairs = None

    # # triangle graph
    # G = nx.Graph()
    # G.add_weighted_edges_from([(0,1,-log(.5)), (1,2,-log(.1)), (2,0,-log(1))])
    # pairs = [(0,1)]
    #
    # # one-edge graph
    # G = nx.Graph()
    # G.add_weighted_edges_from([(0,1,-log(.5))])
    # pairs = [(0,1)]

    # # protein graph
    # G = nx.Graph()
    # G.add_weighted_edges_from([(0,2,-log(.3)), (1,2,-log(.3)), (3,4,-log(.3)), (3,5,-log(.3)), (2,3,-log(.2))])
    #
    # G.add_edge(0, 4, weight=-log(.1))
    # G.add_edge(4, 5, weight=-log(.15))
    # G.add_edge(5, 6, weight=-log(0.5))
    #
    # G = nx.DiGraph()
    # G.add_edges_from([(1,2),(1,3),(2,3)])
    #
    # G = nx.DiGraph()
    # G.add_weighted_edges_from([(1,2,1.3),(1,3,1.5),(2,3,1.7)])
    # Q = G.copy()
    # Q.remove_edge(1,2)
    #
    # save_for_LP_dir("LP/Ain.txt", "LP/Aout.txt", "LP/din.txt", "LP/dout.txt", Q, G, "LP/e.txt", "LP/D.txt")

    console = []

    # in_d, out_d = exp_degrees(G)
    # print in_d
    # print out_d
    # time2prob = time.time()
    # Q = pres_dir_degrees_equally(nx.DiGraph(G).edges(), in_d, 'in')
    # print exp_degrees(Q)
    # print 'Finished', time.time() - time2prob

    # names2numb = dict()
    # count = 0
    # G = nx.DiGraph()
    # with open('memeS.probs') as f:
    #     next(f)
    #     for line in f:
    #         data = line.split()
    #         if data[0] not in names2numb:
    #             names2numb[data[0]] = count
    #             count += 1
    #         if data[1] not in names2numb:
    #             names2numb[data[1]] = count
    #             count += 1
    #         G.add_edge(names2numb[data[0]], names2numb[data[1]], weight=-log(float(data[2])))
    #
    # in_d, out_d = exp_degrees(G)
    # print in_d
    # print sorted(in_d.iteritems(), key=lambda(k,v):v)
    # print sorted(out_d.iteritems(), key=lambda(k,v):v)

    # Q = pres_dir_degrees_equally(G.edges(), out_d, 'out')
    # in_d, out_d = exp_degrees(Q)
    # print in_d
    # for e in Q.edges(data=True):
    #     print e[0],e[1],exp(1)**(-Q[e[0]][e[1]]['weight'])
    # print sorted(in_d.iteritems(), key=lambda(k,v):v)
    # print sorted(out_d.iteritems(), key=lambda(k,v):v)


    # for i in range(1,11):
    #     edges = get_sparsified_top(G, int(i*len(G.edges())/10))
    #     Q = pres_dir_degrees_equally(edges, out_d, "out")
    #     Q.add_nodes_from(G)
    #     print len(Q), len(Q.edges())
        # with open("memeM/memeM_%s.txt" %(i*10), "w+") as f:
        #     for e in Q.edges(data=True):
        #         f.write("%s %s %s\n" %(e[0], e[1], exp(1)**(-e[2]['weight'])))

    console = []

    # save_for_LP("A.dat", "b.dat", G, G)

    # edge_order = []
    # with open("edge_order.txt") as f:
    #     for line in f:
    #         d = line.split()
    #         e = tuple(map(int, (d[0], d[1])))
    #         edge_order.append(e)
    # x = []
    # with open("x.txt") as f:
    #     for line in f:
    #         x.append(float(line))
    # print calculate_obj(edge_order, x, G)

    # G = nx.Graph()
    # G.add_weighted_edges_from([(1,2,-log(.5)),(1,4,-log(.5)),(2,4,-log(.5)),(4,3,-log(.5)),(4,5,-log(.5)),(5,7,-log(.5)),(5,6,-log(.5))])
    # # get_edges_greedy_min(G, 6)
    # get_sparsified_MPSTplus(G, 2)

    # time2sp = time.time()
    # get_sparsified_MP_MPST(G, int(len(G.edges())/2 + 1000))
    # print time.time() - time2sp

    # Get PW, MPST, SP
    # time2PW = time.time()
    # PW, MPSTs = get_PW(G)
    # print "Extracted MPST PW in %s sec" %(time.time() - time2PW)
    # print

    # G = nx.Graph()
    # G.add_edge(0, 1, weight=-log(.1))
    # G.add_edge(1,2, weight=-log(.2))
    # CL = ChungLu(G)
    # print CL

    MC = 100

    # G_rel1 = dict()
    # pairs = []
    # selected = dict()
    # with open("hep15233_2_rel.txt") as f:
    #     for line in f:
    #         u,v,r = line.split()
    #         u = int(u)
    #         v = int(v)
    #         G_rel1[(u,v)] = float(r)
    #         if not selected.get((u,v), False):
    #             pairs.append((u,v))
    #             selected[(v,u)] = True
    # print 'Total number of pairs: %s' %(len(pairs))

    # print 'Finding edge reliabilities...'
    # edge_rel = get_rel_with_mc(G, MC, pairs=G.edges(), cutoff_multiplier= 0)

    percentage = .1
    obj_random_results = dict()
    avg_random_error = dict()
    obj_top_results = dict()
    avg_top_error = dict()
    obj_top2_results = dict()
    avg_top2_error = dict()
    obj_mpst_results = dict()
    avg_mpst_error = dict()
    obj_top3_results = dict()
    avg_top3_error = dict()
    obj_ABM1_results = dict()
    avg_ABM1_error = dict()
    obj_ABM1uncertain_results = dict()
    avg_ABM1uncertain_error = dict()
    obj_top_rd_results = dict()
    avg_top_rd_error = dict()
    obj_top2_rd_results = dict()
    avg_top2_rd_error = dict()
    obj_top3_rd_results = dict()
    avg_top3_rd_error = dict()
    obj_random_rd_results = dict()
    avg_random_rd_error = dict()
    obj_random_rd_results = dict()
    avg_mpst_rd_error = dict()
    obj_mpst_rd_results = dict()

    # get sparsified edges
    # for i in range(1, 11):
    #     time2top = time.time()
    #     top_edges_full = get_sparsified_top(G, int(i*len(G.edges())/1000))
    #     print 'Sparsified Top in %s sec' %(time.time() - time2top)
    #     SP_top = nx.Graph()
    #     SP_top.add_edges_from(top_edges_full)
    #     print len(SP_top), len(SP_top.edges())
    #
    #     track = i*10
    #
    #     save_for_LP("LP/A%s.dat" %track, "LP/b%s.dat" %track, SP_top, G,
    #                 "LP/edge_order%s.dat" %track, "LP/D%s.dat" %track)


    # save_for_LP("A.dat", "b.dat", SP_top)

    # # with distribution
    # time2RD_top = time.time()
    # RD_top = get_redistributed(G, top_edges_full)
    # print "Extracted Top RD in %s sec" %(time.time() - time2RD_top)
    # print

    # time2top2 = time.time()
    # top2_edges_full = get_sparsified_top2(G, len(G.edges()), edge_rel=edge_rel)
    # print 'Sparsified Top2 in %s sec' %(time.time() - time2top2)
    # time2top3 = time.time()
    # top3_edges_full = get_sparsified_top3(G, len(G.edges()), edge_rel=edge_rel)
    # print 'Sparsified Top3 in %s sec' %(time.time() - time2top3)
    # time2random = time.time()
    # random_edges_full = get_sparsified_random(G, len(G.edges()))
    # print 'Sparsified Random in %s sec' %(time.time() - time2random)
    # time2mpst = time.time()
    # mpst_edges_full = get_sparsified_mpst(MPSTs, len(G.edges()))
    # print 'Sparsified MPST in %s sec' %(time.time() - time2mpst)

    # for track in range(1,11):
    #     time2track = time.time()
        # print '------------------------------------'
        # print 'Step %s' %track
        # K = int(track*percentage*len(G.edges()))
        # track += 1
        # # K = 5*int(P)
        # print 'K:', K, "|G|:", len(G.edges())

        # edges = get_sparsified_MP_MPST(G, K, True) # Note directed graph
        # with open("memeS/MPST/K%s.txt" %((track-1)*10), "w+") as f:
        #     for (u,v,d) in edges:
        #         f.write("%d %d %s\n" %(u,v,exp(-d["weight"])))
        # with open("memeS/MPST/attrK%s.txt" %((track-1)*10), "w+") as f:
        #     f.write("n=%s\n" %len(G))
        #     f.write("m=%s\n" %(K))

        # get sparsified edges
        # top_edges = top_edges_full[:K]
        # top2_edges = top2_edges_full[:K]
        # top3_edges = top3_edges_full[:K]
        # random_edges = random_edges_full[:K]
        # mpst_edges = mpst_edges_full[:K]

        # time2SP_top = time.time()
        # SP_top = nx.Graph()
        # SP_top.add_edges_from(top_edges)
        # print "Extracted Top SP in %s sec" %(time.time() - time2SP_top)
        # print
        #
        # time2SP_top2 = time.time()
        # SP_top2 = nx.Graph()
        # SP_top2.add_edges_from(top2_edges)
        # print "Extracted Top2 SP in %s sec" %(time.time() - time2SP_top2)
        # print
        #
        # time2SP_top3 = time.time()
        # SP_top3 = nx.Graph()
        # SP_top3.add_edges_from(top3_edges)
        # print "Extracted Top3 SP in %s sec" %(time.time() - time2SP_top3)
        # print
        #
        # time2SP_random = time.time()
        # SP_random = nx.Graph()
        # SP_random.add_edges_from(random_edges)
        # print "Extracted Random SP in %s sec" %(time.time() - time2SP_random)
        # print
        #
        # time2SP_mpst = time.time()
        # SP_mpst = nx.Graph()
        # SP_mpst.add_edges_from(mpst_edges)
        # print "Extracted MPST SP in %s sec" %(time.time() - time2SP_mpst)
        # print
        #
        # time2SP_ABM1 = time.time()
        # SP_ADR = get_graph_from_file("Sparsified_results/ADR_noredistribution/ADR%s_hep15233_2.txt" %((track-1)*10))
        # print "Extracted ADR SP in %s sec" %(time.time() - time2SP_ABM1)
        # print
        #
        # # with distribution
        # time2RD_top = time.time()
        # RD_top = get_redistributed(G, top_edges)
        # print "Extracted Top RD in %s sec" %(time.time() - time2RD_top)
        # print
        #
        # with open("Sparsified_results/Redistributed_graph/hep15233_2_K%s.txt" %((track-1)*10), "w+") as f:
        #     for e in sorted(RD_top.edges(data = True), key = lambda (u, v, d): exp(1)**(-d["weight"]), reverse=True):
        #         f.write("%s %s %s\n" %(e[0], e[1], exp(1)**(-e[2]["weight"])))
        #
        # time2RD_top2 = time.time()
        # RD_top2 = get_redistributed(G, top2_edges)
        # print "Extracted Top2 RD in %s sec" %(time.time() - time2RD_top2)
        # print
        #
        # time2RD_top3 = time.time()
        # RD_top3 = get_redistributed(G, top3_edges)
        # print "Extracted Top3 RD in %s sec" %(time.time() - time2RD_top3)
        # print
        #
        # time2RD_random = time.time()
        # RD_random = get_redistributed(G, random_edges)
        # print "Extracted Random RD in %s sec" %(time.time() - time2RD_random)
        # print
        #
        # time2RD_mpst = time.time()
        # RD_mpst = get_redistributed(G, mpst_edges)
        # print "Extracted MPST RD in %s sec" %(time.time() - time2RD_mpst)
        # print

        # time2RD_ABM1uncertain = time.time()
        # RD_ADR = get_graph_from_file("Sparsified_results/ADR_redistribution/ADR%suncertain_hep15233_2.txt" %((track-1)*10))
        # print "Extracted ADR RD in %s sec" %(time.time() - time2RD_ABM1uncertain)
        # print

        # ----------------- Get Reliability ------------------ #

        # time2SP_toprel = time.time()
        # SP_top_rel = get_rel_with_mc(SP_top, MC, pairs, cutoff_multiplier=0)
        # # SP_top_rel = get_rel_for_pw(SP_top, pairs)
        # print "Calculated SP_top MC reliability  in %s sec" %(time.time() - time2SP_toprel)
        # print
        #
        # time2SP_top2rel = time.time()
        # SP_top2_rel = get_rel_with_mc(SP_top2, MC, pairs, cutoff_multiplier=0)
        # # SP_top2_rel = get_rel_for_pw(SP_top2, pairs)
        # print "Calculated SP_top2 MC reliability  in %s sec" %(time.time() - time2SP_top2rel)
        # print
        #
        # time2SP_top3rel = time.time()
        # SP_top3_rel = get_rel_with_mc(SP_top3, MC, pairs, cutoff_multiplier=0)
        # # SP_top3_rel = get_rel_for_pw(SP_top3, pairs)
        # print "Calculated SP_top3 MC reliability  in %s sec" %(time.time() - time2SP_top3rel)
        # print
        #
        # time2SP_Randomrel = time.time()
        # SP_random_rel = get_rel_with_mc(SP_random, MC, pairs, cutoff_multiplier=0)
        # # SP_random_rel = get_rel_for_pw(SP_random, pairs)
        # print "Calculated SP_random MC reliability  in %s sec" %(time.time() - time2SP_Randomrel)
        # print
        #
        # time2SP_mpstrel = time.time()
        # SP_mpst_rel = get_rel_with_mc(SP_mpst, MC, pairs, cutoff_multiplier=0)
        # # SP_mpst_rel = get_rel_for_pw(SP_mpst, pairs)
        # print "Calculated SP_mpst MC reliability  in %s sec" %(time.time() - time2SP_mpstrel)
        # print
        #
        # time2SP_abm1rel = time.time()
        # SP_ADR_rel = get_rel_with_mc(SP_ADR, MC, pairs, cutoff_multiplier=0)
        # # SP_ABM1_rel = get_rel_for_pw(SP_ABM1, pairs)
        # print "Calculated SP_ADR MC reliability  in %s sec" %(time.time() - time2SP_abm1rel)
        # print

        # with distribution
        #
        # time2SP_toprel = time.time()
        # RD_top_rel = get_rel_with_mc(RD_top, MC, pairs, cutoff_multiplier=0)
        # # SP_top_rel = get_rel_for_pw(SP_top, pairs)
        # print "Calculated RD_top MC reliability  in %s sec" %(time.time() - time2SP_toprel)
        # print
        #
        # time2SP_top2rel = time.time()
        # RD_top2_rel = get_rel_with_mc(RD_top2, MC, pairs, cutoff_multiplier=0)
        # # SP_top2_rel = get_rel_for_pw(SP_top2, pairs)
        # print "Calculated SP_top2 MC reliability  in %s sec" %(time.time() - time2SP_top2rel)
        # print
        #
        # time2SP_top3rel = time.time()
        # RD_top3_rel = get_rel_with_mc(RD_top3, MC, pairs, cutoff_multiplier=0)
        # # SP_top3_rel = get_rel_for_pw(SP_top3, pairs)
        # print "Calculated SP_top3 MC reliability  in %s sec" %(time.time() - time2SP_top3rel)
        # print
        #
        # time2SP_Randomrel = time.time()
        # RD_random_rel = get_rel_with_mc(RD_random, MC, pairs, cutoff_multiplier=0)
        # # SP_random_rel = get_rel_for_pw(SP_random, pairs)
        # print "Calculated SP_random MC reliability  in %s sec" %(time.time() - time2SP_Randomrel)
        # print
        #
        # time2SP_mpstrel = time.time()
        # RD_mpst_rel = get_rel_with_mc(RD_mpst, MC, pairs, cutoff_multiplier=0)
        # # SP_mpst_rel = get_rel_for_pw(SP_mpst, pairs)
        # print "Calculated SP_mpst MC reliability  in %s sec" %(time.time() - time2SP_mpstrel)
        # print

        # time2SP_abm1uncertainrel = time.time()
        # RD_ADR_rel = get_rel_with_mc(RD_ADR, MC, pairs, cutoff_multiplier=0)
        # # SP_ABM1uncertain_rel = get_rel_for_pw(SP_ABM1uncertain, pairs)
        # print "Calculated RD_ADR MC reliability  in %s sec" %(time.time() - time2SP_abm1uncertainrel)
        # print


        # time2Grel2 = time.time()
        # G_rel2 = get_rel_with_mpst(MPSTs, pairs)
        # print "Calculated G MPST reliability in %s sec" %(time.time() - time2Grel2)
        # print
        #
        # time2PWrel = time.time()
        # PW_rel = get_rel_for_pw(PW, pairs)
        # print "Calculated PW reliability in %s sec" %(time.time() - time2PWrel)
        # print

        # ---------------------- Calculate Objective -------------------------- #
        # time2Obj1 = time.time()
        # Obj1, avg_Obj1 = get_objective(G_rel1, SP_random_rel)
        # print "Calculated MC-MC Random objective in %s sec" %(time.time() - time2Obj1)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj1, avg_Obj1)
        # print
        #
        # obj_random_results.update({(track-1)*10: Obj1})
        # avg_random_error.update({(track-1)*10: avg_Obj1})
        # with open("Sparsified_results/Sparsified_Random_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_random_results, fp)
        # with open("Sparsified_results/Sparsified_Random_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_random_error, fp)
        #
        # time2Obj2 = time.time()
        # Obj2, avg_Obj2 = get_objective(G_rel1, SP_top_rel)
        # print "Calculated MC-MC Top objective in %s sec" %(time.time() - time2Obj2)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj2, avg_Obj2)
        # print
        #
        # obj_top_results.update({(track-1)*10: Obj2})
        # avg_top_error.update({(track-1)*10: avg_Obj2})
        # with open("Sparsified_results/Sparsified_Top_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_top_results, fp)
        # with open("Sparsified_results/Sparsified_Top_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_top_error, fp)
        #
        # time2Obj3 = time.time()
        # Obj3, avg_Obj3 = get_objective(G_rel1, SP_top2_rel)
        # print "Calculated MC-MC Top2 objective in %s sec" %(time.time() - time2Obj3)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj3, avg_Obj3)
        # print
        #
        # obj_top2_results.update({(track-1)*10: Obj3})
        # avg_top2_error.update({(track-1)*10: avg_Obj3})
        # with open("Sparsified_results/Sparsified_Top2_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_top2_results, fp)
        # with open("Sparsified_results/Sparsified_Top2_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_top2_error, fp)
        #
        # time2Obj4 = time.time()
        # Obj4, avg_Obj4 = get_objective(G_rel1, SP_mpst_rel)
        # print "Calculated MC-MC MPST objective in %s sec" %(time.time() - time2Obj4)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj4, avg_Obj4)
        # print
        #
        # obj_mpst_results.update({(track-1)*10: Obj4})
        # avg_mpst_error.update({(track-1)*10: avg_Obj4})
        # with open("Sparsified_results/Sparsified_MPST_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_mpst_results, fp)
        # with open("Sparsified_results/Sparsified_MPST_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_mpst_error, fp)
        #
        # time2Obj5 = time.time()
        # Obj5, avg_Obj5 = get_objective(G_rel1, SP_top3_rel)
        # print "Calculated MC-MC Top3 objective in %s sec" %(time.time() - time2Obj5)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj5, avg_Obj5)
        # print
        #
        # obj_top3_results.update({(track-1)*10: Obj5})
        # avg_top3_error.update({(track-1)*10: avg_Obj5})
        # with open("Sparsified_results/Sparsified_Top3_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_top3_results, fp)
        # with open("Sparsified_results/Sparsified_Top3_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_top3_error, fp)
        #
        # time2Obj6 = time.time()
        # Obj6, avg_Obj6 = get_objective(G_rel1, SP_ADR_rel)
        # print "Calculated MC-MC ABM1 objective in %s sec" %(time.time() - time2Obj6)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj6, avg_Obj6)
        # print
        #
        # obj_ABM1_results.update({(track-1)*10: Obj6})
        # avg_ABM1_error.update({(track-1)*10: avg_Obj6})
        # with open("Sparsified_results/Sparsified_ADR_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_ABM1_results, fp)
        # with open("Sparsified_results/Sparsified_ADR_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_ABM1_error, fp)

        # with redistribution
        # print "WITH DISTRIBUTION"

        # time2Obj7 = time.time()
        # Obj7, avg_Obj7 = get_objective(G_rel1, RD_ADR_rel)
        # print "Calculated MC-MC ABM1uncertain objective in %s sec" %(time.time() - time2Obj7)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj7, avg_Obj7)
        # print
        #
        # obj_ABM1uncertain_results.update({(track-1)*10: Obj7})
        # avg_ABM1uncertain_error.update({(track-1)*10: avg_Obj7})
        # with open("Sparsified_results/Redistributed_ADR_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_ABM1uncertain_results, fp)
        # with open("Sparsified_results/Redistributed_ADR_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_ABM1uncertain_error, fp)
        #
        # time2Obj8 = time.time()
        # Obj8, avg_Obj8 = get_objective(G_rel1, RD_random_rel)
        # print "Calculated MC-MC Random objective in %s sec" %(time.time() - time2Obj8)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj8, avg_Obj8)
        # print
        #
        # obj_random_rd_results.update({(track-1)*10: Obj8})
        # avg_random_rd_error.update({(track-1)*10: avg_Obj8})
        # with open("Sparsified_results/Redistributed_Random_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_random_rd_results, fp)
        # with open("Sparsified_results/Redistributed_Random_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_random_rd_error, fp)
        #
        # time2Obj9 = time.time()
        # Obj9, avg_Obj9 = get_objective(G_rel1, RD_top_rel)
        # print "Calculated MC-MC Top objective in %s sec" %(time.time() - time2Obj9)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj9, avg_Obj9)
        # print
        #
        # obj_top_rd_results.update({(track-1)*10: Obj9})
        # avg_top_rd_error.update({(track-1)*10: avg_Obj9})
        # with open("Sparsified_results/Redistributed_Top_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_top_rd_results, fp)
        # with open("Sparsified_results/Redistributed_Top_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_top_rd_error, fp)
        #
        # time2Obj10 = time.time()
        # Obj10, avg_Obj10 = get_objective(G_rel1, RD_top2_rel)
        # print "Calculated MC-MC Top2 objective in %s sec" %(time.time() - time2Obj10)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj10, avg_Obj10)
        # print
        #
        # obj_top2_rd_results.update({(track-1)*10: Obj10})
        # avg_top2_rd_error.update({(track-1)*10: avg_Obj10})
        # with open("Sparsified_results/Redistributed_Top2_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_top2_rd_results, fp)
        # with open("Sparsified_results/Redistributed_Top2_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_top2_rd_error, fp)
        #
        # time2Obj11 = time.time()
        # Obj11, avg_Obj11 = get_objective(G_rel1, RD_mpst_rel)
        # print "Calculated MC-MC MPST objective in %s sec" %(time.time() - time2Obj11)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj11, avg_Obj11)
        # print
        #
        # obj_mpst_rd_results.update({(track-1)*10: Obj11})
        # avg_mpst_rd_error.update({(track-1)*10: avg_Obj11})
        # with open("Sparsified_results/Redistributed_MPST_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_mpst_rd_results, fp)
        # with open("Sparsified_results/Redistributed_MPST_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_mpst_rd_error, fp)
        #
        # time2Obj12 = time.time()
        # Obj12, avg_Obj12 = get_objective(G_rel1, RD_top3_rel)
        # print "Calculated MC-MC Top3 objective in %s sec" %(time.time() - time2Obj12)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj12, avg_Obj12)
        # print
        #
        # obj_top3_rd_results.update({(track-1)*10: Obj12})
        # avg_top3_rd_error.update({(track-1)*10: avg_Obj12})
        # with open("Sparsified_results/Redistributed_Top3_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_top3_rd_results, fp)
        # with open("Sparsified_results/Redistributed_Top3_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_top3_rd_error, fp)

        # print
        # print 'Spent %s for iteration' %(time.time() - time2track)

    print "Finished execution in %s sec" %(time.time() - time2execute)

    console = []