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
            if u in PW and v in PW and nx.has_path(G, u, v):
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
    print "Sorted edges..."

    SP = nx.Graph() # sparsified graph
    SP.add_edges_from(sorted_edges[:K])

    return SP

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
    SP = nx.Graph()
    SP.add_edges_from(sorted_edges[:K])
    return SP

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
    SP = nx.Graph()
    SP.add_edges_from(sorted_edges[:K])
    return SP

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

    SP = nx.Graph()
    SP.add_edges_from(sorted_edges[:K])
    return SP

def get_graph_from_file(filename):
    SP = nx.Graph()
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

if __name__ == "__main__":
    time2execute = time.time()

    time2read = time.time()
    G = get_graph_from_file("hep15233_2.txt")
    print "Read graph in % sec" %(time.time() - time2read)
    print "G: n = %s m = %s" %(len(G), len(G.edges()))
    print

    time2pairs = time.time()
    G_rel_mc = dict()
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

    # protein graph
    # G = nx.Graph()
    # G.add_weighted_edges_from([(0,2,-log(.3)), (1,2,-log(.3)), (3,4,-log(.3)), (3,5,-log(.3)), (2,3,-log(.2))])

    # Get PW, MPST, SP
    time2PW = time.time()
    PW, MPSTs = get_PW(G)
    print "Extracted MPST PW in %s sec" %(time.time() - time2PW)
    print

    MC = 100

    # time2Grel1 = time.time()
    # G_rel1 = get_rel_with_mc(G, MC, pairs=pairs, cutoff_multiplier= 2.5)
    # print 'Number of rel: %s' %(len(G_rel1))
    # G_rel_mc.update(G_rel1)
    # print 'Number of total rel: %s' %(len(G_rel_mc))
    # print "Calculated G MC reliability  in %s sec" %(time.time() - time2Grel1)
    # print

    G_rel1 = dict()
    pairs = []
    selected = dict()
    with open("hep15233_2_rel.txt") as f:
        for line in f:
            u,v,r = line.split()
            u = int(u)
            v = int(v)
            G_rel1[(u,v)] = float(r)
            if not selected.get((u,v), False):
                pairs.append((u,v))
                selected[(v,u)] = True

    edge_rel = get_rel_with_mc(G, MC, pairs=G.edges(), cutoff_multiplier= 0)
    # G_rel1 = get_rel_for_pw(G, pairs)

    # time2SP2 = time.time()
    # SP2 = get_sparsified_greedy(G, K)
    # print "Extracted Greedy SP in %s sec" %(time.time() - time2SP2)
    # print


    # P = sum([exp(1)**(-data["weight"]) for (u,v,data) in G.edges(data=True)]) # expected number of edges
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
    for track in range(1,10):
        time2track = time.time()
        print '------------------------------------'
        print 'Step %s' %track
        K = int(track*percentage*len(G.edges()))
        track += 1
        # K = 5*int(P)
        print 'K:', K, "|G|:", len(G.edges())

        time2SP_top = time.time()
        SP_top = get_sparsified_top(G, K)
        print "Extracted Top SP in %s sec" %(time.time() - time2SP_top)
        print

        time2SP_top2 = time.time()
        SP_top2 = get_sparsified_top2(G, K, edge_rel)
        print "Extracted Top2 SP in %s sec" %(time.time() - time2SP_top2)
        print

        time2SP_top3 = time.time()
        SP_top3 = get_sparsified_top3(G, K, edge_rel)
        print "Extracted Top3 SP in %s sec" %(time.time() - time2SP_top3)
        print

        time2SP_random = time.time()
        SP_random = get_sparsified_random(G, K)
        print "Extracted Random SP in %s sec" %(time.time() - time2SP_random)
        print

        time2SP_ABM1 = time.time()
        SP_ABM1 = get_graph_from_file("Sparsified_results/ADR_noredistribution/ADR%s_hep15233_2.txt" %((track-1)*10))
        print "Extracted ABM1 SP in %s sec" %(time.time() - time2SP_ABM1)
        print

        time2SP_ABM1uncertain = time.time()
        SP_ABM1uncertain = get_graph_from_file("Sparsified_results/ADR_redistribution/ADR%suncertain_hep15233_2.txt" %((track-1)*10))
        print "Extracted ABM1uncertain SP in %s sec" %(time.time() - time2SP_ABM1uncertain)
        print

        time2SP_mpst = time.time()
        SP_mpst = get_sparsified_mpst(MPSTs, K)
        print "Extracted MPST SP in %s sec" %(time.time() - time2SP_mpst)
        print

        # ----------------- Get Reliability ------------------ #

        time2SP_toprel = time.time()
        SP_top_rel = get_rel_with_mc(SP_top, MC, pairs, cutoff_multiplier=0)
        # SP_top_rel = get_rel_for_pw(SP_top, pairs)
        print "Calculated SP_top MC reliability  in %s sec" %(time.time() - time2SP_toprel)
        print

        time2SP_top2rel = time.time()
        SP_top2_rel = get_rel_with_mc(SP_top2, MC, pairs, cutoff_multiplier=0)
        # SP_top2_rel = get_rel_for_pw(SP_top2, pairs)
        print "Calculated SP_top2 MC reliability  in %s sec" %(time.time() - time2SP_top2rel)
        print

        time2SP_top3rel = time.time()
        SP_top3_rel = get_rel_with_mc(SP_top3, MC, pairs, cutoff_multiplier=0)
        # SP_top3_rel = get_rel_for_pw(SP_top3, pairs)
        print "Calculated SP_top3 MC reliability  in %s sec" %(time.time() - time2SP_top3rel)
        print

        time2SP_Randomrel = time.time()
        SP_random_rel = get_rel_with_mc(SP_random, MC, pairs, cutoff_multiplier=0)
        # SP_random_rel = get_rel_for_pw(SP_random, pairs)
        print "Calculated SP_random MC reliability  in %s sec" %(time.time() - time2SP_Randomrel)
        print

        time2SP_mpstrel = time.time()
        SP_mpst_rel = get_rel_with_mc(SP_mpst, MC, pairs, cutoff_multiplier=0)
        # SP_mpst_rel = get_rel_for_pw(SP_mpst, pairs)
        print "Calculated SP_mpst MC reliability  in %s sec" %(time.time() - time2SP_mpstrel)
        print

        time2SP_abm1rel = time.time()
        SP_ABM1_rel = get_rel_with_mc(SP_ABM1, MC, pairs, cutoff_multiplier=0)
        # SP_ABM1_rel = get_rel_for_pw(SP_ABM1, pairs)
        print "Calculated SP_ABM1 MC reliability  in %s sec" %(time.time() - time2SP_abm1rel)
        print

        time2SP_abm1uncertainrel = time.time()
        SP_ABM1uncertain_rel = get_rel_with_mc(SP_ABM1uncertain, MC, pairs, cutoff_multiplier=0)
        # SP_ABM1uncertain_rel = get_rel_for_pw(SP_ABM1uncertain, pairs)
        print "Calculated SP_ABM1uncertain MC reliability  in %s sec" %(time.time() - time2SP_abm1uncertainrel)
        print


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
        time2Obj1 = time.time()
        Obj1, avg_Obj1 = get_objective(G_rel1, SP_random_rel)
        print "Calculated MC-MC Random objective in %s sec" %(time.time() - time2Obj1)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj1, avg_Obj1)
        print

        obj_random_results.update({(track-1)*10: Obj1})
        avg_random_error.update({(track-1)*10: avg_Obj1})
        with open("Sparsified_results/Sparsified_Random_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_random_results, fp)
        with open("Sparsified_results/Sparsified_Random_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_random_error, fp)

        time2Obj2 = time.time()
        Obj2, avg_Obj2 = get_objective(G_rel1, SP_top_rel)
        print "Calculated MC-MC Top objective in %s sec" %(time.time() - time2Obj2)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj2, avg_Obj2)
        print

        obj_top_results.update({(track-1)*10: Obj2})
        avg_top_error.update({(track-1)*10: avg_Obj2})
        with open("Sparsified_results/Sparsified_Top_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_top_results, fp)
        with open("Sparsified_results/Sparsified_Top_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_top_error, fp)

        time2Obj3 = time.time()
        Obj3, avg_Obj3 = get_objective(G_rel1, SP_top2_rel)
        print "Calculated MC-MC Top2 objective in %s sec" %(time.time() - time2Obj3)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj3, avg_Obj3)
        print

        obj_top2_results.update({(track-1)*10: Obj3})
        avg_top2_error.update({(track-1)*10: avg_Obj3})
        with open("Sparsified_results/Sparsified_Top2_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_top2_results, fp)
        with open("Sparsified_results/Sparsified_Top2_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_top2_error, fp)

        time2Obj4 = time.time()
        Obj4, avg_Obj4 = get_objective(G_rel1, SP_mpst_rel)
        print "Calculated MC-MC MPST objective in %s sec" %(time.time() - time2Obj4)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj4, avg_Obj4)
        print

        obj_mpst_results.update({(track-1)*10: Obj4})
        avg_mpst_error.update({(track-1)*10: avg_Obj4})
        with open("Sparsified_results/Sparsified_MPST_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_mpst_results, fp)
        with open("Sparsified_results/Sparsified_MPST_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_mpst_error, fp)

        time2Obj5 = time.time()
        Obj5, avg_Obj5 = get_objective(G_rel1, SP_top3_rel)
        print "Calculated MC-MC Top3 objective in %s sec" %(time.time() - time2Obj5)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj5, avg_Obj5)
        print

        obj_top3_results.update({(track-1)*10: Obj5})
        avg_top3_error.update({(track-1)*10: avg_Obj5})
        with open("Sparsified_results/Sparsified_Top3_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_top3_results, fp)
        with open("Sparsified_results/Sparsified_Top3_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_top3_error, fp)

        time2Obj6 = time.time()
        Obj6, avg_Obj6 = get_objective(G_rel1, SP_ABM1_rel)
        print "Calculated MC-MC ABM1 objective in %s sec" %(time.time() - time2Obj6)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj6, avg_Obj6)
        print

        obj_ABM1_results.update({(track-1)*10: Obj6})
        avg_ABM1_error.update({(track-1)*10: avg_Obj6})
        with open("Sparsified_results/Sparsified_ABM1_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_ABM1_results, fp)
        with open("Sparsified_results/Sparsified_ABM1_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_ABM1_error, fp)

        time2Obj7 = time.time()
        Obj7, avg_Obj7 = get_objective(G_rel1, SP_ABM1uncertain_rel)
        print "Calculated MC-MC ABM1uncertain objective in %s sec" %(time.time() - time2Obj7)
        print "MC-MC Objective: %s; avg Objective: %s" %(Obj7, avg_Obj7)
        print

        obj_ABM1uncertain_results.update({(track-1)*10: Obj7})
        avg_ABM1uncertain_error.update({(track-1)*10: avg_Obj7})
        with open("Sparsified_results/Sparsified_ABM1uncertain_MC%s.txt" %MC, "w+") as fp:
            json.dump(obj_ABM1uncertain_results, fp)
        with open("Sparsified_results/Sparsified_ABM1uncertain_MC%s_avg.txt" %MC, "w+") as fp:
            json.dump(avg_ABM1uncertain_error, fp)

        print
        print 'Spent %s for iteration' %(time.time() - time2track)

    print "Finished execution in %s sec" %(time.time() - time2execute)

    console = []