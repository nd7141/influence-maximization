'''
Implementation of Most Probable Spanning Tree (MPST).
Extract MPST and delete edges from original graph that belong to that MPST.
Continue until all edges are deleted from G.
Use those MPST to select a possible world (PW) and approximate reliability.

How do we compute ground-truth?
How do we compute reliability for PW?

We are solving 1st problem: minimizing sum over all pairs of reliability with a constraint on the number of edges
'''
from __future__ import division
import networkx as nx
from itertools import cycle
from math import exp, log
import random, time

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
        mpst = nx.minimum_spanning_tree(E)
        print '|mpst.edges|:', len(mpst.edges())
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
            print "Added edge (%s,%s): P %s |PW.edges| %s" %(u,v,P,len(PW.edges()))
        # stop when expected number of edges reached
        if len(PW.edges()) > P:
            break
    print 'PW:', len(PW), 'PW.edges:', len(PW.edges())
    print 'G:', len(G), 'G.edges', len(G.edges())
    return PW, MPSTs

def get_rel_with_mpst(MPSTs):
    '''
    Compute estimates of reliability of every pair of nodes

    Expected to have -log(p_e) as weights in each mpst
    '''
    rel = dict()
    nodes = MPSTs[0].nodes()
    for en, mpst in enumerate(MPSTs):
        print 'mpst #', en
        # TODO consider pairs only in the tree, not in G
        for i in range(len(nodes)-1):
            for j in range(i+1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                # find reliability along the path between u and v
                paths = list(nx.all_simple_paths(mpst, u,v))
                assert len(paths) <= 1, "Should be only one path from u=%s to v=%s" %(u,v)
                if len(paths) == 1:
                    path = paths[0]
                    r = 1 # reliability between u and v along the path in this mpst
                    for ix in range(len(path)-1):
                        weight = mpst[path[ix]][path[ix+1]]["weight"]
                        r *= exp(1)**(-weight)
                    # add reliability in this mpst to total reliability
                    rel[(u,v)] = rel.get((u,v), 0) + r
                    rel[(v,u)] = rel.get((v,u), 0) + r
    return rel

def get_rel_with_mc(G, mc=100):
    '''
    Compute reliability between a pair of nodes using Monte-Carlo simulations

    Expected to have -log(p_e) as weights in G
    '''
    rel = dict()
    for _ in range(mc):
        # create a random PW
        live_edges = [(e[0], e[1]) for e in G.edges(data=True) if exp(1)**(-e[2]["weight"]) > random.random()]
        E = nx.Graph()
        E.add_edges_from(live_edges)
        print "Created a random PW"
        print 'E:', len(E), 'E.edges:', len(E.edges())
        # TODO assign 1./mc to all pairs in CC instead of in G
        # for every pair of nodes check if there is a path
        nodes = E.nodes()
        for i in range(len(nodes)-1):
            print len(nodes), '-->', i
            for j in range(i+1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                if nx.has_path(E, u, v):
                    rel[(u,v)] = rel.get((u,v), 0) + 1./mc
                    rel[(v,u)] = rel.get((v,u), 0) + 1./mc
    return rel

def get_objective(G_rel, PW_rel, edges):
    '''
    Computes the objective, which is the sum of reliability discrepancies over all pairs of nodes.
    '''
    Obj = 0
    for e in edges:
        Obj += abs(G_rel[e] - PW_rel[e])
    return Obj

if __name__ == "__main__":
    time2execute = time.time()

    G = nx.Graph()
    time2read = time.time()
    with open("hep15233.txt") as f:
        for line in f:
            u,v,p = map(float, line.split())
            if u != v:
                G.add_edge(int(u), int(v), weight = -log(p))
    print "Read graph in % sec" %(time.time() - time2read)
    print "G: n %s m %s" %(len(G), len(G.edges()))
    print

    time2PW = time.time()
    PW, MPSTs = get_PW(G)
    print "Extracted MPST PW in %s sec" %(time.time() - time2PW)
    print

    # TODO check how mc affects results
    time2Grel2 = time.time()
    G_rel2 = get_rel_with_mpst(MPSTs)
    print "Calculated G MPST reliability in % sec" %(time.time() - time2Grel2)
    print

    time2PWrel = time.time()
    PW_rel = get_rel_with_mc(PW, mc = 1)
    print "Calculated PW reliability in % sec" %(time.time() - time2PWrel)
    print

    time2Grel1 = time.time()
    G_rel1 = get_rel_with_mc(G, mc = 1)
    print "Calculated G MC reliability  in % sec" %(time.time() - time2Grel1)
    print

    edges = G.edges()

    time2Obj1 = time.time()
    Obj1 = get_objective(G_rel1, PW_rel, edges)
    print "Calculated MC-MC objective in %s sec" %(time.time() - time2Obj1)
    print "MC-MC Objective: %s" %Obj1
    print

    time2Obj2 = time.time()
    Obj2 = get_objective(G_rel2, PW_rel, edges)
    print "Calculated MPST-MC objective in %s sec" %(time.time() - time2Obj2)
    print "MPST-MC Objective: %s" %Obj2
    print

    print "Finished execution in %s sec" %(time.time() - time2execute)

    console = []