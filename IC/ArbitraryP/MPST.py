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
import random, time, sys, json
from collections import Counter
from itertools import product

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

def get_rel_with_mc(G, mc=100, pairs=None):
    '''
    Compute reliability between a pair of nodes using Monte-Carlo simulations

    Expected to have -log(p_e) as weights in G.

    if pairs = None finds reliability among all pairs,
    else only for those pairs specified.
    '''
    print 'MC:', mc
    rel = dict()
    for _ in range(mc):
        # create a random PW
        live_edges = [(e[0], e[1], e[2]) for e in G.edges(data=True) if e[2]["weight"] < -log(random.random())]
        E = nx.Graph()
        E.add_edges_from(live_edges)
        if pairs:
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
            CCs = nx.connected_components(E)
            for cc in CCs:
                if len(cc) > 1:
                    for u in cc:
                        for v in cc:
                            rel[(u,v)] = rel.get((u,v), 0) + 1./mc
    for key in rel:
        rel[key] = float(rel[key])/mc
        if key in print_dict:
            print_dict[key] /= mc
    if pairs:
        print sorted(print_dict.items(), key = lambda (k,v): v, reverse = True)[:10]
    else:
        print sorted(rel.items(), key = lambda (k,v): v, reverse=True)[:10]
    return rel

def get_rel_for_pw(PW, pairs=None):
    '''
    if pairs = None finds reliability among all pairs,
    else only for those pairs specified.
    '''
    # print len(PW), len(PW.edges())
    all_edges = PW.edges()
    rel = dict()
    if pairs:
        print_dict = dict()
        for pair in pairs:
            u = pair[0]
            v = pair[1]
            r = 0
            if pair in all_edges:
                r = 1
            rel[(u,v)] = r
            rel[(v,u)] = r
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
    return Obj/2, Obj/(2*len(pairs))

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
        sorted_edges.extend(sorted(mpst.edges(data=True), key = lambda (u,v,data): exp(1)**(-data["weight"]), reverse=True))
    print "Sorted edges..."

    SP = nx.Graph() # sparcified graph
    SP.add_edges_from(sorted_edges[:K])

    return SP

def get_sparsified_greedy(G, K):
    '''
    Get sparsified (uncertain graph with K edges) graph using greedy.

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
    sorted_edges = sorted(all_edges, key = lambda (u,v,data): exp(1)**(-data["weight"]), reverse=True)
    SP = nx.Graph() # sparcified graph
    SP.add_edges_from(sorted_edges[:K])
    return SP

def get_sparsified_random(G, K):
    '''
    Get sparsified (uncertain graph with K edges) graph using random edges.

    Expected to have -log(p_e) as weights in G.
    '''
    all_edges = G.edges(data=True)
    random_edges = random.sample(all_edges, K)
    SP = nx.Graph() # sparsified graph
    SP.add_edges_from(random_edges[:K])
    return SP

if __name__ == "__main__":
    time2execute = time.time()

    G = nx.Graph()
    time2read = time.time()
    with open("hep15233_2.txt") as f:
        for line in f:
            u, v, p = map(float, line.split())
            if u != v:
                G.add_edge(int(u), int(v), weight = -log(p))
    print "Read graph in % sec" %(time.time() - time2read)
    print "G: n = %s m = %s" %(len(G), len(G.edges()))
    print

    time2pairs = time.time()
    l = 4000
    # pairs = _make_pairs(G, l)
    # print 'Made %s pairs in %s sec' %(l, time.time() - time2pairs)
    pairs = [(11052, 11371), (9097, 10655)]
    pairs = [(11052, 11371)]
    print pairs
    # pairs = G.edges()
    # pairs = None

    # triangle graph
    # G = nx.Graph()
    # G.add_weighted_edges_from([(0,1,-log(.5)), (1,2,-log(.1)), (2,0,-log(1))])
    # pairs = [(0,1)]

    # Get PW, MPST, SP
    # time2PW = time.time()
    # PW, MPSTs = get_PW(G)
    # print "Extracted MPST PW in %s sec" %(time.time() - time2PW)
    # print

    MC = 1000

    time2Grel1 = time.time()
    G_rel1 = get_rel_with_mc(G, MC, pairs)
    print "Calculated G MC reliability  in %s sec" %(time.time() - time2Grel1)
    print

    # time2SP1 = time.time()
    # SP1 = get_sparsified_mpst(MPSTs, K)
    # print "Extracted MPST SP in %s sec" %(time.time() - time2SP1)
    # print

    # time2SP2 = time.time()
    # SP2 = get_sparsified_greedy(G, K)
    # print "Extracted Greedy SP in %s sec" %(time.time() - time2SP2)
    # print

    # time2SP3 = time.time()
    # SP3 = get_sparsified_top(G, K)
    # print "Extracted Top SP in %s sec" %(time.time() - time2SP3)
    # print

    # P = sum([exp(1)**(-data["weight"]) for (u,v,data) in G.edges(data=True)]) # expected number of edges
    # percentage = .1
    # obj_results = dict()
    # avg_error = dict()
    # for track in range(10, 11):
    #     print '------------------------------------'
    #     print 'Step %s' %track
    #     K = int(track*percentage*len(G.edges()))
    #     track += 1
    #     # K = 5*int(P)
    #     print 'K:', K, "|G|:", len(G.edges())

        # time2SP_Random = time.time()
        # SP_random = get_sparsified_random(G, K)
        # print "Extracted Random SP in %s sec" %(time.time() - time2SP_Random)
        # print

        # ----------------- Get Reliability ------------------ #

        # time2SP_Randomrel = time.time()
        # SP_random_rel = get_rel_with_mc(G, MC, pairs)
        # print "Calculated SP MC reliability  in %s sec" %(time.time() - time2SP_Randomrel)
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

        # time2SP1rel = time.time()
        # SP1_rel = get_rel_with_mc(SP1, MC, pairs)
        # print "Calculated MPST SP MC reliability  in %s sec" %(time.time() - time2SP1rel)
        # print

        # time2SP2rel = time.time()
        # SP2_rel = get_rel_with_mc(SP2, MC, pairs)
        # print "Calculated Greedy SP MC reliability  in %s sec" %(time.time() - time2SP2rel)
        # print

        # time2SP3rel = time.time()
        # SP3_rel = get_rel_with_mc(SP3, MC, pairs)
        # print "Calculated Top SP MC reliability  in %s sec" %(time.time() - time2SP3rel)
        # print

        # ---------------------- Calculate Objective -------------------------- #
        # time2Obj1 = time.time()
        # Obj1, avg_Obj1 = get_objective(G_rel1, SP_random_rel)
        # print "Calculated MC-MC objective in %s sec" %(time.time() - time2Obj1)
        # print "MC-MC Objective: %s; avg Objective: %s" %(Obj1, avg_Obj1)
        # print
        # obj_results.update({(track-1)*10: Obj1})
        # avg_error.update({(track-1)*10: avg_Obj1})
        # with open("Sparsified_results/Sparsified_Random_MC%s.txt" %MC, "w+") as fp:
        #     json.dump(obj_results, fp)
        # with open("Sparsified_results/Sparsified_Random_MC%s_avg.txt" %MC, "w+") as fp:
        #     json.dump(avg_error, fp)
        #
        # time2Obj2 = time.time()
        # Obj2 = get_objective(G_rel2, PW_rel)
        # print "Calculated MPST-MC objective in %s sec" %(time.time() - time2Obj2)
        # print "MPST-MC Objective: %s" %Obj2
        # print

    print "Finished execution in %s sec" %(time.time() - time2execute)

    console = []