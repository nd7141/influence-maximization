
'''
CCWP heuristic for arbitrary propagation probabilities.
'''

from __future__ import division
import networkx as nx
from heapq import nlargest
from copy import deepcopy
import os, json, multiprocessing, random
from runIAC import *

def weighted_choice(choices):
    '''
    http://stackoverflow.com/a/3679747/2069858'''
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w > r:
            return c
        upto += w
    assert False, "Shouldn't get here"

def findCC(G, Ep, cascade = "IC"):
    '''
    G is undirected graph
    '''

    # remove blocked edges from graph G
    E = deepcopy(G)
    if cascade == "IC":
        edge_rem = [e for e in E.edges() if random.random() < (1-Ep[e])**(E[e[0]][e[1]]['weight'])]
    elif cascade == "LT":
        for u in G:
            W = [Ep[e] for e in G.edges(u)]
            choices = zip(G.edges(u), W)
            live_edge = weighted_choice(choices)

    E.remove_edges_from(edge_rem)

    # initialize CC
    CC = dict() # each component is reflection of the number of a component to its members
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

def CCWP((G, k, Ep)):
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
    sortedCC = sorted([(len(cc), cc_number) for (cc_number, cc) in CC.iteritems()], reverse=True)
    topCCnumbers = sortedCC[:k] # CCs we assign scores to
    QN = sum([l for (l, _) in topCCnumbers]) # number of qualified nodes

    increment = 0
    try:
        while k+increment < len(sortedCC) and sortedCC[k + increment][0] == sortedCC[k-1][0]:
            topCCnumbers.append(sortedCC[k + increment])
            increment += 1
            QN += sortedCC[k + increment][0]
    except IndexError:
        pass
    # assign scores to nodes in top Connected Components
    prev_length  = topCCnumbers[0][0]
    rank = 1
    QCC = len(topCCnumbers)
    for length, numberCC in topCCnumbers:
        if length != prev_length:
            prev_length = length
            rank += 1
        weighted_score = 1.0/length # updatef = 1
        for node in CC[numberCC]:
            scores[node] += weighted_score
    return scores

def CCWP_directed((G, k, Ep)):
    '''
    Implements Harvester for directed graphs
    Model: IC
    '''

    # remove blocked edges
    E = deepcopy(G)
    edge_rem = [edge for edge in E.edges() if random.random() < (1-Ep[edge])**(E[edge[0]][edge[1]]['weight'])]
    E.remove_edges_from(edge_rem)

    # find score for each node
    scores = dict(zip(E.nodes(), [0]*len(E)))
    reachability = dict()
    for node in E:
        reachable_nodes = [node]

        # Do BFS
        out_edges = E.out_edges(node)
        i = 0
        while i < len(out_edges):
            e = out_edges[i]
            if e[1] not in reachable_nodes:
                reachable_nodes.append(e[1])
                out_edges.extend(E.out_edges(e[1]))
            i += 1
        reachability[node] = reachable_nodes
        scores[node] = len(reachable_nodes)

    # enhance scores
    enhanced_scores = dict(zip(E.nodes(), [0]*len(E))) # resulted scores
    sorted_scores = sorted(scores.iteritems(), key = lambda (dk, dv): dv)
    reached_nodes = []

    already_selected = 0
    last_score = 0
    for node, score in sorted_scores:
        if already_selected <= k:
            if node not in reached_nodes:
                enhanced_scores[node] = score
                reached_nodes.extend(reachability[node])
                last_score = score
        else:
            if score == last_score:
                if node not in reached_nodes:
                    enhanced_scores[node] = score
                    reached_nodes.extend(reachability[node])
            else:
                break

    return enhanced_scores

def frange(begin, end, step):
    x = begin
    y = end
    while x < y:
        yield x
        x += step

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == '__main__':
    import time
    start = time.time()

    dataset = "gnu09"
    model = "MultiValency"
    print model, dataset

    if model == "MultiValency":
        ep_model = "range1_directed"
    elif model == "Random":
        ep_model = "random1_directed"
    elif model == "Categories":
        ep_model = "degree1_directed"

    G = nx.read_gpickle("../../graphs/%s.gpickle" %dataset)
    print 'Read graph G'
    print time.time() - start

    Ep = dict()
    with open("Ep_%s_%s.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    #calculate initial set
    R = 200
    I = 500
    ALGO_NAME = "CCWP"
    FOLDER = "Data4InfMax"
    SEEDS_FOLDER = "Seeds"
    TIME_FOLDER = "Time"
    DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
    # seeds_filename = FOLDER + "/" + SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
    seeds_filename = FOLDER + "/" + SEEDS_FOLDER + "/%s_%s_%s_%s_directed.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
    time_filename = FOLDER + "/" + TIME_FOLDER + "/%s_%s_%s_%s.txt" %(TIME_FOLDER, ALGO_NAME, dataset, model)
    logfile = open('log.txt', 'w+')
    # print >>logfile, '--------------------------------'
    # print >>logfile, time.strftime("%c")

    l2c = []
    pool = None
    pool2 = None
    # open file for writing output
    seeds_file = open("%s" %seeds_filename, "a+")
    time_file = open("%s" %time_filename, "a+")
    dbox_seeds_file = open("%s/%s" %(DROPBOX_FOLDER, seeds_filename), "a+")
    dbox_time_file = open("%s/%s" %(DROPBOX_FOLDER, time_filename), "a+")
    for length in range(151, 152, 10):
        time2length = time.time()
        print 'Start finding solution for length = %s' %length
        print >>logfile, 'Start finding solution for length = %s' %length
        time2S = time.time()

        print 'Start mapping...'
        time2map = time.time()
        # define map function
        # def map_CCWP(it):
        #     return CCWP(G, length, Ep)
        if pool == None:
            pool = multiprocessing.Pool(processes=2)
        Scores = pool.map(CCWP_directed, ((G, length, Ep) for i in range(R)))
        # print 'Finished mapping in', time.time() - time2map

        print 'Start reducing...'
        time2reduce = time.time()
        scores = {v: sum([s[v] for s in Scores]) for v in G}
        scores_copied = deepcopy(scores)
        S = []
        # penalization phase
        for it in range(length):
            maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv)
            S.append(maxk)
            scores_copied.pop(maxk) # remove top element from dict
            for v in G[maxk]:
                if v not in S:
                    # weight = scores_copied[v]/maxv
                    # print weight,
                    penalty = (1-Ep[(maxk, v)])**(G[maxk][v]['weight'])
                    scores_copied[v] = penalty*scores_copied[v]

        print S
        print avgIAC(G, S, Ep, 500)
        # print >>logfile, json.dumps(S)
        time2complete = time.time() - time2S
        # with open("%s" %time_filename, "a+") as time_file:
        #     print >>time_file, (time2complete)
        # with open("%s/%s" %(DROPBOX_FOLDER, time_filename), "a+") as dbox_time_file:
        #     print >>dbox_time_file, (time2complete)
        print 'Finish finding S in %s sec...' %(time2complete)

        # print 'Writing S to files...'
        # with open("%s" %seeds_filename, "a+") as seeds_file:
        #     print >>seeds_file, json.dumps(S)
        # with open("%s/%s" %(DROPBOX_FOLDER, seeds_filename), "a+") as dbox_seeds_file:
        #     print >>dbox_seeds_file, json.dumps(S)

        # print 'Start calculating coverage...'
        # coverage = sum(pool.map(getCoverage, ((G, S, Ep) for _ in range(I))))/I
        # print 'S:', S
        # print 'Coverage', coverage

        print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        print '----------------------------------------------'

    seeds_file.close()
    dbox_seeds_file.close()
    time_file.close()
    dbox_time_file.close()
    logfile.close()

    print 'Total time: %s' %(time.time() - start)

    console = []