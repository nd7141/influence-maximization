from __future__ import division
import networkx as nx
from DD import DD
from runIAC import *
import os, json, math, operator, multiprocessing, time
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

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == '__main__':
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

    G = nx.read_gpickle("../../graphs/U%s.gpickle" %dataset)
    print 'Read graph G'
    print time.time() - start

    Ep = dict()
    with open("Ep_%s_%s1.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    R = 500
    I = 1000
    ALGO_NAME = "GDD"
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
        S = []

        print "Initializing scores..."
        scores, active, inactive = getScores(G, Ep)
        scores_copied = deepcopy(scores)

        print 'Selecting seed set S...'
        time2select = time.time()
        # add first node to S
        updateScores(scores_copied, active, inactive, S, Ep)
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
                updateScores(scores_copied, active, inactive, S, Ep)
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

    print 'Total time:', time.time() - start
    console = []