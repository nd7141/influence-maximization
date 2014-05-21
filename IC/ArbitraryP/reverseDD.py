from __future__ import division
import math
import networkx as nx
from DD import DD
from runIAC import *
import os, json, operator, multiprocessing

def getScores(G):
    '''Finds scores for DD.
    Score are degree for each node.
    '''

    scores = dict()
    t = dict()
    for node in G:
        scores[node] = sum(G[node][v]['weight'] for v in G[node])
        t[node] = 0 # number of activated neighbors

    return scores, t

def updateScores(scores_copied, t, S, Ep):
    maxk, maxv = max(scores_copied.iteritems(), key = operator.itemgetter(1))
    S.append(maxk)
    scores_copied.pop(maxk)
    neighbors_weights = [G[u][v]['weight'] for v in G[u] if v not in S]
    w_avg = sum(neighbors_weights)/len(neighbors_weights)
    for v in G[maxk]:
        if v not in S:
            t[v] += G[maxk][v]['weight'] # increase number of selected neighbors
            p = Ep[(maxk,v)]
            scores_copied[v] -= 2*t[v] + (scores_copied[v] - t[v])*t[v]*p**w_avg # discount of degree

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('../../graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    fileno = "test"
    time2result = time.time()
    tsize = 1000
    prange = [.01, .02, .04, .08]
    I = 1000
    avgS = 0
    M = 1
    length_to_coverage = {0:0}
    norm_parameters = dict()

    for ep_it in range(M):
        print 'Iteration ep_it:', ep_it
        scores, t = getScores(G)
        scores_copied = deepcopy(scores)
        Coverages = {}
        coverage = 0
        S = []
        pool = None
        Ep = random_from_range(G, prange)
        def mapAvgSize (S):
            return avgIAC(G, S, Ep, I)
        if pool == None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        print 'Selecting seed set S...'
        time2select = time.time()
        # add first node to S
        updateScores(scores_copied, t, S, Ep)

        # find Low and High
        while coverage < tsize:
            Low = len(S)
            High = 2*Low
            while len(S) < 2*Low:
                updateScores(scores_copied, t, S, Ep)
            T = pool.map(mapAvgSize, [S]*4)
            coverage = sum(T)/len(T)
            Coverages[len(S)] = coverage
            print '|S|: %s --> %s' %(len(S), coverage)

        # find boundary using binary search
        lastS = S # S gives us solution for k = 1..len(S)
        while Low + 1 != High:
            time2double = time.time()
            new_length = Low + (High - Low)//2
            lastS = S[:new_length]

            T = pool.map(mapAvgSize, [lastS]*4)
            coverage = sum(T)/len(T)
            Coverages[new_length] = coverage
            print '|S|: %s --> %s' %(len(lastS), coverage)

            if coverage < tsize:
                Low = len(lastS)
            else:
                High = len(lastS)

        # final check for k that reach coverage
        if Coverages[Low] >= tsize:
            finalS = S[:Low]
        elif Coverages[High] >= tsize:
            finalS = S[:High]

        print 'Finished selecting seed set S: %s sec' %(time.time() - time2select)

        print 'Coverage: ', Coverages[len(finalS)]
        print finalS
        print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(finalS), tsize)

        # map length: [0,len(finalS)] to coverage
        print 'Start estimating coverages...'
        step = 5
        for length in range(1, len(finalS)+1, step):
            if length in Coverages:
                norm_parameters[length] = norm_parameters.get(length,0) + 1
                length_to_coverage[length] = length_to_coverage.get(length, 0) + Coverages[length]
            else:
                norm_parameters[length] = norm_parameters.get(length,0) + 1
                # calculate coverage
                T = pool.map(mapAvgSize, [finalS[:length]]*4)
                coverage = sum(T)/len(T)
                length_to_coverage[length] = length_to_coverage.get(length, 0) + coverage

        # if we haven't added result for tsize, then add it
        if (len(finalS) - 1)%step != 0:
            norm_parameters[len(finalS)] = norm_parameters.get(len(finalS),0) + 1
            length_to_coverage[len(finalS)] = length_to_coverage.get(len(finalS), 0) + Coverages[len(finalS)]

        print 'Time to find result for %s iteration: %s sec' %(ep_it + 1, time.time() - time2result)
        print '------------------------------------------------'

    # normalizing coverages
    for length in norm_parameters:
        length_to_coverage[length] /= norm_parameters[length]

    length_to_coverage = sorted(length_to_coverage.iteritems(), key = lambda (dk, dv): dk)

    fp = open("plotDD.txt", "w+")
    json.dump(length_to_coverage, fp)

    print 'Total time:', time.time() - start
    console = []