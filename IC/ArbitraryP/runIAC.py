'''
Independent Arbitrary Cascade (IAC) is a independent cascade model with arbitrary
 propagation probabilities.
'''
from __future__ import division
from copy import deepcopy
import random, multiprocessing, os
import networkx as nx

def uniformEp(G, p = .01):
    '''
    Every edge has the same probability p.
    '''
    if type(G) == type(nx.DiGraph()):
        Ep = dict(zip(G.edges(), [p]*len(G.edges())))
    elif type(G) == type(nx.Graph()):
        Ep = dict()
        for (u, v) in G.edges():
            Ep[(u, v)] = p
            Ep[(u, v)] = p
    else:
        raise ValueError, "Provide either nx.Graph or nx.DiGraph object"
    return Ep

def randomEp(G, maxp):
    '''
    Every edge has random propagation probability <= maxp <= 1
    '''
    assert maxp <= 1, "Maximum probability cannot exceed 1."
    Ep = dict()
    if type(G) == type(nx.DiGraph()):
        for v1,v2 in G.edges():
            p = random.uniform(0, maxp)
            Ep[(v1,v2)] = p
    elif type(G) == type(nx.Graph()):
        for v1,v2 in G.edges():
            p = random.uniform(0, maxp)
            Ep[(v1,v2)] = p
            Ep[(v2,v1)] = p
    else:
        raise ValueError, "Provide either nx.Graph or nx.DiGraph object"
    return Ep

def random_from_range (G, prange):
    '''
    Every edge has propagation probability chosen from prange uniformly at random.
    '''
    for p in prange:
        if p > 1:
            raise ValueError, "Propagation probability inside range should be <= 1"
    Ep = dict()
    if type(G) == type(nx.DiGraph()):
        for v1,v2 in G.edges():
            p = random.choice(prange)
            Ep[(v1,v2)] = p
    elif type(G) == type(nx.DiGraph()):
        for v1,v2 in G.edges():
            p = random.choice(prange)
            Ep[(v1,v2)] = p
            Ep[(v2,v1)] = p
    return Ep

def runIAC (G, S, Ep):
    ''' Runs independent arbitrary cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    Ep -- propagation probabilities
    Output: T -- resulted influenced set of vertices (including S)

    NOTE:
    Ep is a dictionary for each edge it has associated probability
    If graph is undirected for each edge (v1,v2) with probability p,
     we have Ep[(v1,v2)] = p, Ep[(v2,v1)] = p.
    '''
    T = deepcopy(S) # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                p = Ep[(T[i],v)] # propagation probability
                if random.random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
        i += 1
    return T

def avgIAC (G, S, Ep, I):
    '''
    Input:
        G -- undirected graph
        S -- seed set
        Ep -- propagation probabilities
        I -- number of iterations
    Output:
        avg -- average size of coverage
    '''
    avg = 0
    for i in range(I):
        avg += float(len(runIAC(G,S,Ep)))/I
    return avg

def findCC(G, Ep):
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1-Ep[e])**(E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)

    # initialize CC
    CC = dict() # each component is reflection os the number of a component to its members
    explored = dict(zip(E.nodes(), [False]*len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CC[c] = [node]
            component = E[node].keys()
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CC[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    return CC

def getScores(G, k, Ep):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     k -- number of nodes in seed set (int)
     p -- propagation probability among all edges (int)
     Output:
     scores -- scores of nodes according to some weight function (dict)
    '''
    scores = dict(zip(G.nodes(), [0]*len(G))) # initialize scores

    CC = findCC(G, Ep)

    # find ties for components of rank k and add them all as qualified
    sortedCC = sorted([(len(dv), dk) for (dk, dv) in CC.iteritems()], reverse=True)
    topCCnumbers = sortedCC[:k] # CCs we assign scores to
    L = sum([l for (l,_) in topCCnumbers])

    increment = 0
    # add ties of rank k (if len of kth CC == 1 then add all CCs)
    while k+increment < len(sortedCC) and sortedCC[k + increment][0] == sortedCC[k-1][0]:
        topCCnumbers.append(sortedCC[k + increment])
        increment += 1

    # assign scores to nodes in top Connected Components
    # prev_length  = topCCnumbers[0][0]
    # rank = 1
    for length, numberCC in topCCnumbers:
        # if length != prev_length:
        #     prev_length = length
        #     rank += 1
        # weighted_score = length
        weighted_score = 1.0/length
        # weighted_score = 1
        for node in CC[numberCC]:
            scores[node] += weighted_score
    return scores

def CCWP(G, k, Ep, R):
    def map_CCWP(it):
        return getScores(G, k, Ep)
    Scores = map(map_CCWP, range(R))

    scores = {v: sum([s[v] for s in Scores]) for v in G}
    scores_copied = deepcopy(scores)
    S = []
    # penalization phase
    for it in range(k):
        maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv)
        # print maxv,
        S.append(maxk)
        scores_copied.pop(maxk) # remove top element from dict
        for v in G[maxk]:
            if v not in S:
                # weight = scores_copied[v]/maxv
                # print weight,
                penalty = (1-Ep[(maxk, v)])**(G[maxk][v]['weight'])
                scores_copied[v] = penalty*scores_copied[v]
    return S

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    # G = nx.DiGraph()
    # with open('../../graphdata/epi.txt') as f:
    #     n, m = f.readline().split()
    #     for line in f:
    #         try:
    #             u, v = map(int, line.split())
    #         except ValueError:
    #             continue
    #         try:
    #             G[u][v]['weight'] += 1
    #             G[v][u]["weight"] += 1
    #         except:
    #             G.add_edge(u, v, weight=1)
    #             G.add_edge(v, u, weight=1)
    # print 'Built graph G'
    # print time.time() - start
    #
    #
    # nx.write_gpickle(G, "../../graphs/epi.gpickle")
    # print 'Wrote graph G'
    # print time.time() - start
    G = nx.read_gpickle("../../graphs/epi.gpickle")
    print 'Read graph G'
    print time.time() - start

    # random.seed(1)
    #
    # time2probability = time.time()
    # prange = [.01, .02, .04, .08]
    # Ep = random_from_range(G, prange)
    # print 'Built probabilities Ep'
    # print time.time() - time2probability
    #
    # with open("Ep_epi_range1.txt", "w+") as f:
    #     for key, value in Ep.iteritems():
    #         f.write(str(key[0]) + " " + str(key[1]) + " " + str(value) + os.linesep)
    #
    #
    # time2probability = time.time()
    # Ep = randomEp(G, .1)
    # print 'Built probabilities Ep'
    # print time.time() - time2probability
    #
    # with open("Ep_epi_random1.txt", "w+") as f:
    #     for key, value in Ep.iteritems():
    #         f.write(str(key[0]) + " " + str(key[1]) + " " + str(value) + os.linesep)
    #
    # time2probability = time.time()
    # Ep = uniformEp(G, .01)
    # print 'Built probabilities Ep'
    # print time.time() - time2probability
    #
    # with open("Ep_epi_uniform1.txt", "w+") as f:
    #     for key, value in Ep.iteritems():
    #         f.write(str(key[0]) + " " + str(key[1]) + " " + str(value) + os.linesep)

    # import json
    # coverage2length = [[0,0]]
    # with open("plotdata/rawCCWPforDirect2.txt") as f:
    #     for line in f:
    #         [(cov, S)] = json.loads(line).items()
    #         coverage2length.append([len(S), int(cov)])
    #
    # coverage2length.sort(key=lambda (l,_): l)
    #
    # with open("plotdata/plotReverseCCWPforReverse2_v2.txt", "w+") as f:
    #     json.dump(coverage2length, f)


    console = []
