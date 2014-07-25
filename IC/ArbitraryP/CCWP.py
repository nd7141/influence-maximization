
'''
CCWP heuristic for arbitrary propagation probabilities.
'''

from __future__ import division
import networkx as nx
from heapq import nlargest
from copy import deepcopy
import os, json, multiprocessing, random
from runIAC import *

def findCC(G, Ep):
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1-Ep[e])**(E[e[0]][e[1]]['weight'])]
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
    # add ties of rank k (if len of kth CC == 1 then add all CCs)
    #TODO find how it can be IndexError (case study: fb MultiValency k = 186)
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
        # weighted_score = 1 # updatef = 2
        # weighted_score = length # updatef = 3
        # weighted_score = 1.0/length**.5 # updatef = 4
        # weighted_score = 1.0/length**2  # updatef = 5
        # weighted_score = QCC/length # updatef = 6
        # weighted_score = 1.0/(length*QCC) # updatef = 7
        # weighted_score = 1.0/(1 - (1 - length)*(1 - rank)/(1 - k)) # updatef = 8
        # weighted_score = 1 - (length - 1)*(1 - rank)/(length*(1 - k)) # updatef = 9
        # weighted_score = 1/QN #updatef = 10
        for node in CC[numberCC]:
            scores[node] += weighted_score
    return scores
# TODO reduce number of binary steps by changing between binary and incremental search
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

    model = "MultiValency"
    print model

    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    dataset = "gnu09"

    G = nx.read_gpickle("../../graphs/%s.gpickle" %dataset)
    print 'Read graph G'
    print time.time() - start

    Ep = dict()
    with open("Ep_%s_%s1.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    #calculate initial set
    R = 500
    I = 1000
    ALGO_NAME = "CCWP"
    FOLDER = "Data4InfMax"
    SEEDS_FOLDER = "Seeds"
    TIME_FOLDER = "Time"
    DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
    seeds_filename = SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
    time_filename = TIME_FOLDER + "/%s_%s_%s_%s.txt" %(TIME_FOLDER, ALGO_NAME, dataset, model)
    logfile = open('log.txt', 'w+')
    # print >>logfile, '--------------------------------'
    # print >>logfile, time.strftime("%c")

    l2c = []
    pool = None
    pool2 = None
    # open file for writing output
    seeds_file = open(seeds_filename, "a+")
    time_file = open(time_filename, "a+")
    dbox_seeds_file = open("%/%", DROPBOX_FOLDER, seeds_filename, "a+")
    dbox_time_file = open("%/%", DROPBOX_FOLDER, time_filename, "a+")
    for length in range(1, 250, 5):
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
            pool = multiprocessing.Pool(processes=None)
        Scores = pool.map(CCWP, ((G, length, Ep) for i in range(R)))
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
        print >>logfile, json.dumps(S)
        time2complete = time.time() - time2S
        print >>time_file, (time2complete)
        print >>dbox_time_file, (time2complete)
        print 'Finish finding S in %s sec...' %(time2complete)

        print 'Writing S to files...'
        print >>seeds_filename, json.dumps(S)
        print >>dbox_seeds_file, json.dumps(S)

        # print 'Start calculating coverage...'
        # def map_AvgIAC (it):
        #     return avgIAC(G, S, Ep, I)
        # # if pool2 == None:
        # #     pool2 = multiprocessing.Pool(processes=None)
        # avg_size = 0
        # time2avg = time.time()
        # Ts = pool.map(getCoverage, ((G, S, Ep) for i in range(I)))
        # avg_size = sum(Ts)/len(Ts)
        # print 'Average coverage of %s nodes is %s' %(length, avg_size)
        # print 'Finished averaging seed set size in', time.time() - time2avg
        #
        # l2c.append([length, avg_size])
        # with open('plotdata/plot' + FILENAME, 'w+') as fp:
        #     json.dump(l2c, fp)
        # with open(DROPBOX + 'plotdata/plot' + FILENAME, 'w+') as fp:
        #     json.dump(l2c, fp)

        print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        print '----------------------------------------------'

    seeds_file.close()
    dbox_seeds_file.close()
    time_file.close()
    dbox_time_file.close()
    logfile.close()

    print 'Total time: %s' %(time.time() - start)

    console = []