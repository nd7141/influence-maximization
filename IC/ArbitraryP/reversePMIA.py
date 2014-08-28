''' Implementation of PMIA algorithm [1].

[1] -- Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks.
'''

from __future__ import division
import networkx as nx
import math, time
from copy import deepcopy
from runIAC import *
import multiprocessing, json

def updateAP(ap, S, PMIIAv, PMIIA_MIPv, Ep):
    ''' Assumption: PMIIAv is a directed tree, which is a subgraph of general G.
    PMIIA_MIPv -- dictionary of MIP from nodes in PMIIA
    PMIIAv is rooted at v.
    '''
    # going from leaves to root
    sorted_MIPs = sorted(PMIIA_MIPv.iteritems(), key = lambda (_, MIP): len(MIP), reverse = True)
    for u, _ in sorted_MIPs:
        if u in S:
            ap[(u, PMIIAv)] = 1
        elif not PMIIAv.in_edges([u]):
            ap[(u, PMIIAv)] = 0
        else:
            in_edges = PMIIAv.in_edges([u], data=True)
            prod = 1
            for w, _, edata in in_edges:
                p = (1 - (1 - Ep[(w, u)])**edata["weight"])
                prod *= 1 - ap[(w, PMIIAv)]*p
            ap[(u, PMIIAv)] = 1 - prod

def updateAlpha(alpha, v, S, PMIIAv, PMIIA_MIPv, Ep, ap):
    # going from root to leaves
    sorted_MIPs =  sorted(PMIIA_MIPv.iteritems(), key = lambda (_, MIP): len(MIP))
    for u, mip in sorted_MIPs:
        if u == v:
            alpha[(PMIIAv, u)] = 1
        else:
            out_edges = PMIIAv.out_edges([u])
            assert len(out_edges) == 1, "node u=%s must have exactly one neighbor, got %s instead" %(u, len(out_edges))
            w = out_edges[0][1]
            if w in S:
                alpha[(PMIIAv, u)] = 0
            else:
                in_edges = PMIIAv.in_edges([w], data=True)
                prod = 1
                for up, _, edata in in_edges:
                    if up != u:
                        pp_upw = 1 - (1 - Ep[(up, w)])**edata["weight"]
                        prod *= (1 - ap[(up, PMIIAv)]*pp_upw)
                alpha[(PMIIAv, u)] = alpha[(PMIIAv, w)]*(1 - (1 - Ep[(u,w)])**PMIIAv[u][w]["weight"])*prod

def computePMIOA(G, u, theta, S, Ep):
    '''
     Compute PMIOA -- subgraph of G that's rooted at u.
     Uses Dijkstra's algorithm until length of path doesn't exceed -log(theta)
     or no more nodes can be reached.
    '''
    # initialize PMIOA
    PMIOA = nx.DiGraph()
    PMIOA.add_node(u)
    PMIOA_MIP = {u: [u]} # MIP(u,v) for v in PMIOA

    crossing_edges = set([out_edge for out_edge in G.out_edges([u]) if out_edge[1] not in S + [u]])
    edge_weights = dict()
    dist = {u: 0} # shortest paths from the root u

    # grow PMIOA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
            edge_weight = edge_weights[edge]
            if dist[edge[0]] + edge_weight < min_dist:
                min_dist = dist[edge[0]] + edge_weight
                min_edge = edge
        # check stopping criteria
        if min_dist < -math.log(theta):
            dist[min_edge[1]] = min_dist
            PMIOA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIOA_MIP[min_edge[1]] = PMIOA_MIP[min_edge[0]] + [min_edge[1]]
            # update crossing edges
            crossing_edges.difference_update(G.in_edges(min_edge[1]))
            crossing_edges.update([out_edge for out_edge in G.out_edges(min_edge[1])
                                   if (out_edge[1] not in PMIOA) and (out_edge[1] not in S)])
        else:
            break
    return PMIOA, PMIOA_MIP

def updateIS(IS, S, u, PMIOA, PMIIA):
    for v in PMIOA[u]:
        for si in S:
            # if seed node is effective and it's blocked by u
            # then it becomes ineffective
            if (si in PMIIA[v]) and (si not in IS[v]) and (u in PMIIA[v][si]):
                    IS[v].append(si)

def computePMIIA(G, ISv, v, theta, S, Ep):

    # initialize PMIIA
    PMIIA = nx.DiGraph()
    PMIIA.add_node(v)
    PMIIA_MIP = {v: [v]} # MIP(u,v) for u in PMIIA

    crossing_edges = set([in_edge for in_edge in G.in_edges([v]) if in_edge[0] not in ISv + [v]])
    edge_weights = dict()
    dist = {v: 0} # shortest paths from the root u

    # grow PMIIA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = float("Inf")
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = -math.log(1 - (1 - Ep[edge])**G[edge[0]][edge[1]]["weight"])
            edge_weight = edge_weights[edge]
            if dist[edge[1]] + edge_weight < min_dist:
                min_dist = dist[edge[1]] + edge_weight
                min_edge = edge
        # check stopping criteria
        # print min_edge, ':', min_dist, '-->', -math.log(theta)
        if min_dist < -math.log(theta):
            dist[min_edge[0]] = min_dist
            PMIIA.add_edge(min_edge[0], min_edge[1], {"weight": G[min_edge[0]][min_edge[1]]["weight"]})
            PMIIA_MIP[min_edge[0]] = PMIIA_MIP[min_edge[1]] + [min_edge[0]]
            # update crossing edges
            crossing_edges.difference_update(G.out_edges(min_edge[0]))
            if min_edge[0] not in S:
                crossing_edges.update([in_edge for in_edge in G.in_edges(min_edge[0])
                                       if (in_edge[0] not in PMIIA) and (in_edge[0] not in ISv)])
        else:
            break
    return PMIIA, PMIIA_MIP

def updateS (S, IncInf, alpha, ap, IS, PMIOA_MIP, PMIIA_MIP):
    u, _ = max(IncInf.iteritems(), key = lambda (dk, dv): dv)
    IncInf.pop(u) # exclude node u for next iterations
    PMIOA[u], PMIOA_MIP[u] = computePMIOA(G, u, theta, S, Ep)
    for v in PMIOA[u]:
        for w in PMIIA[v]:
            if w not in S + [u]:
                IncInf[w] -= alpha[(PMIIA[v],w)]*(1 - ap[(w, PMIIA[v])])

    updateIS(IS, S, u, PMIOA_MIP, PMIIA_MIP)

    S.append(u)

    for v in PMIOA[u]:
        if v != u:
            PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
            updateAP(ap, S, PMIIA[v], PMIIA_MIP[v], Ep)
            updateAlpha(alpha, v, S, PMIIA[v], PMIIA_MIP[v], Ep, ap)
            # add new incremental influence
            for w in PMIIA[v]:
                if w not in S:
                    IncInf[w] += alpha[(PMIIA[v], w)]*(1 - ap[(w, PMIIA[v])])

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == "__main__":
    start = time.time()

    dataset = "hep"
    model = "Categories"
    print dataset, model

    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    G = nx.read_gpickle("../../graphs/%s.gpickle" %dataset)
    print 'Read graph G'
    print time.time() - start

    Ep = dict()
    with open("Ep_%s_%s1.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    R = 500
    I = 1000
    theta = 1/20
    ALGO_NAME = "PMIA"
    FOLDER = "Data4InfMax/"
    REVERSE_FOLDER = "Reverse"
    STEPS_FOLDER = "Steps"
    DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
    reverse_filename = FOLDER + REVERSE_FOLDER + "/%s_%s_%s_%s.txt" %(REVERSE_FOLDER, ALGO_NAME, dataset, model)
    steps_filename = FOLDER + STEPS_FOLDER + "/%s_%s_%s_%s.txt" %(STEPS_FOLDER, ALGO_NAME, dataset, model)
    pool = multiprocessing.Pool(processes = 4)

    for T in range(2100, 3000, 100):
        time2T = time.time()
        print "T:", T
        Coverages = {0:0}

        print 'Start Initialization for PMIA...'
        S = []
        IncInf = dict(zip(G.nodes(), [0]*len(G)))
        PMIIA = dict() # node to tree
        PMIOA = dict()
        PMIIA_MIP = dict() # node to MIPs (dict)
        PMIOA_MIP = dict()
        ap = dict()
        alpha = dict()
        IS = dict()
        for v in G:
            IS[v] = []
            PMIIA[v], PMIIA_MIP[v] = computePMIIA(G, IS[v], v, theta, S, Ep)
            for u in PMIIA[v]:
                ap[(u, PMIIA[v])] = 0 # ap of u node in PMIIA[v]
            updateAlpha(alpha, v, S, PMIIA[v], PMIIA_MIP[v], Ep, ap)
            for u in PMIIA[v]:
                IncInf[u] += alpha[(PMIIA[v], u)]*(1 - ap[(u, PMIIA[v])])
        print 'Finished initialization'


        print 'Selecting seed set S...'
        time2select = time.time()
        # add first node to S
        updateS(S, IncInf, alpha, ap, IS, PMIOA_MIP, PMIIA_MIP)
        time2Ts = time.time()
        Ts = pool.map(getCoverage, ((G, S, Ep) for i in range(I)))
        coverage = sum(Ts)/len(Ts)
        Coverages[len(S)] = coverage
        time2coverage = time.time() - time2Ts
        print '|S|: %s --> %s nodes | %s sec' %(len(S), coverage, time2coverage)

        Low = 0
        High = 1

        # find Low and High
        while coverage < T:
            Low = len(S)
            High = 2*Low
            while len(S) < High:
                updateS(S, IncInf, alpha, ap, IS, PMIOA_MIP, PMIIA_MIP)
            time2Ts = time.time()
            Ts = pool.map(getCoverage, ((G, S, Ep) for i in range(I)))
            coverage = sum(Ts)/len(Ts)
            Coverages[len(S)] = coverage
            time2coverage = time.time() - time2Ts
            print '|S|: %s --> %s nodes | %s sec' %(len(S), coverage, time2coverage)

        # find boundary using binary search
        lastS = deepcopy(S) # S gives us solution for k = 1..len(S)
        while Low + 1 != High:
            time2double = time.time()
            new_length = Low + (High - Low)//2
            lastS = S[:new_length]
            time2Ts = time.time()
            Ts = pool.map(getCoverage, ((G, lastS, Ep) for i in range(I)))
            coverage = sum(Ts)/len(Ts)
            Coverages[new_length] = coverage
            time2coverage = time.time() - time2Ts
            print '|S|: %s --> %s nodes | %s sec' %(len(lastS), coverage, time2coverage)

            if coverage < T:
                Low = new_length
            else:
                High = new_length

        assert Coverages[Low] < T
        assert Coverages[High] >= T
        finalS = S[:High]

        print 'Finished selecting seed set S: %s sec' %(time.time() - time2select)
        with open(steps_filename, 'a+') as fp:
                print >>fp, T, len(Coverages) - 1
        print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(finalS), T)
        with open(reverse_filename, 'a+') as fp:
                print >>fp, T, High

        print 'Finished seed minimization for T = %s in %s sec' %(T, time.time() - time2T)
        print '----------------------------------------------'

    print 'Total time: %s' %(time.time() - start)