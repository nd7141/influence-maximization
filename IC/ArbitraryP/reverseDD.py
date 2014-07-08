from __future__ import division
import networkx as nx
from DD import DD
from runIAC import *
import os, json, math, operator, multiprocessing
from priorityQueue import PriorityQueue as PQ
from pprint import pprint

def getScores(G, Ep):
    '''Finds scores for GDD.
    Scores are degree for each node.
    '''

    scores = PQ() # degree discount
    active = dict()
    inactive = dict()

    # initialize degree discount
    for u in G:
        active[u] = 1
        # inactive[u] = sum([Ep[(u,v)]*G[u][v]['weight'] for v in G[u]])
        inactive[u] = sum([1 - (1 - Ep[(u,v)])**G[u][v]["weight"] for v in G[u]])
        priority = active[u]*(1 + inactive[u])
        scores.add_task(u, -priority) # add degree of each node

    return scores, active, inactive

def updateScores(scores_copied, active, inactive, S, Ep):
    u, priority = scores_copied.pop_item() # extract node with maximal degree discount
    S.append(u)
    for v in G[u]:
        if v not in S:
            active[v] *= (1-Ep[(u,v)])**G[u][v]['weight']
            inactive[v] -= 1 - (1 - Ep[(u,v)])**G[u][v]['weight']
            priority = active[v]*(1 + inactive[v])
            scores_copied.add_task(v, -priority)


if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.read_gpickle("../../graphs/hep.gpickle")
    print 'Read graph G'
    print time.time() - start

    # get propagation probabilities
    Ep = dict()
    with open("Ep_hep_range1.txt") as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    T = 750
    I = 250

    length_to_coverage = {0:0}
    norm_parameters = dict()
    pool = None
    Coverages = {0:0}
    coverage = 0
    S = []

    model = "MultiValency"
    DROPBOX = "/home/sergey/Dropbox/Influence Maximization/"
    FILENAME = "reverseGDD_%s.txt" %model
    ftime = "time2kGDD_%s.txt" %model

    def mapAvgSize (S):
        return avgIAC(G, S, Ep, I)
    if pool == None:
        pool = multiprocessing.Pool(processes=None)

    print "Initializing scores..."
    scores, active, inactive = getScores(G, Ep)
    scores_copied = deepcopy(scores)

    print 'Selecting seed set S...'
    time2select = time.time()
    # add first node to S
    updateScores(scores_copied, active, inactive, S, Ep)
    time2Ts = time.time()
    Ts = pool.map(mapAvgSize, [S]*4)
    coverage = sum(Ts)/len(Ts)
    Coverages[len(S)] = coverage
    time2coverage = time.time() - time2Ts
    print '|S|: %s --> %s nodes | %s sec' %(len(S), coverage, time2coverage)
    with open("plotdata/" + ftime, 'a+') as fp:
        print >>fp, len(S), time2coverage
    with open(DROPBOX + "plotdata/" + ftime, 'a+') as fp:
        print >>fp, len(S), time2coverage

    Low = 0
    High = 1

    # find Low and High
    while coverage < T:
        Low = len(S)
        High = 2*Low
        while len(S) < High:
            updateScores(scores_copied, active, inactive, S, Ep)
        time2Ts = time.time()
        Ts = pool.map(mapAvgSize, [S]*4)
        coverage = sum(Ts)/len(Ts)
        Coverages[len(S)] = coverage
        time2coverage = time.time() - time2Ts
        print '|S|: %s --> %s nodes | %s sec' %(len(S), coverage, time2coverage)
        with open("plotdata/" + ftime, 'a+') as fp:
            print >>fp, len(S), time2coverage
        with open(DROPBOX + "plotdata/" + ftime, 'a+') as fp:
            print >>fp, len(S), time2coverage

    # find boundary using binary search
    lastS = deepcopy(S) # S gives us solution for k = 1..len(S)
    while Low + 1 != High:
        time2double = time.time()
        new_length = Low + (High - Low)//2
        lastS = S[:new_length]
        time2Ts = time.time()
        Ts = pool.map(mapAvgSize, [lastS]*4)
        coverage = sum(Ts)/len(Ts)
        Coverages[new_length] = coverage
        print '|S|: %s --> %s' %(len(lastS), coverage)
        time2coverage = time.time() - time2Ts
        print '|S|: %s --> %s nodes | %s sec' %(len(S), coverage, time2coverage)
        with open("plotdata/" + ftime, 'a+') as fp:
            print >>fp, new_length, time2coverage
        with open(DROPBOX + "plotdata/" + ftime, 'a+') as fp:
            print >>fp, new_length, time2coverage

        if coverage < T:
            Low = new_length
        else:
            High = new_length

    assert Coverages[Low] < T
    assert Coverages[High] >= T
    finalS = S[:High]

    print 'Finished selecting seed set S: %s sec' %(time.time() - time2select)
    # with open("plotdata/timeReverseDDforReverse3.txt", "w+") as fp:
    #     fp.write("%s" %(time.time() - time2select))
    print 'Coverage: ', Coverages[len(finalS)]
    print finalS
    print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(finalS), T)


    # # map length: [0,len(finalS)] to coverage
    # print 'Start estimating coverages...'
    # step = 5
    # for length in range(1, len(finalS)+1, step):
    #     if length in Coverages:
    #         norm_parameters[length] = norm_parameters.get(length,0) + 1
    #         length_to_coverage[length] = length_to_coverage.get(length, 0) + Coverages[length]
    #         print '|S|: %s --> %s' %(length, Coverages[length])
    #     else:
    #         norm_parameters[length] = norm_parameters.get(length,0) + 1
    #         # calculate coverage
    #         Ts = pool.map(mapAvgSize, [finalS[:length]]*4)
    #         coverage = sum(Ts)/len(Ts)
    #         length_to_coverage[length] = length_to_coverage.get(length, 0) + coverage
    #         print '|S|: %s --> %s' %(length, coverage)
    #
    # # if we haven't added result for T, then add it
    # if (len(finalS) - 1)%step != 0:
    #     norm_parameters[len(finalS)] = norm_parameters.get(len(finalS),0) + 1
    #     length_to_coverage[len(finalS)] = length_to_coverage.get(len(finalS), 0) + Coverages[len(finalS)]
    #     print '|S|: %s --> %s' %(len(finalS), Coverages[len(finalS)])
    #
    # print '------------------------------------------------'
    #
    # # normalizing coverages
    # for length in norm_parameters:
    #     length_to_coverage[length] /= norm_parameters[length]
    #
    # length_to_coverage = sorted(length_to_coverage.iteritems(), key = lambda (dk, dv): dk)

    # with open("plotdata/plotReverseDDforReverse3.txt", "w+") as fp:
    #     json.dump(length_to_coverage, fp)

    print 'Total time:', time.time() - start
    console = []