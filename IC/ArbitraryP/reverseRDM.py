''' Implementation of Degree heuristic.
'''

from __future__ import division
import networkx as nx
import math, time, random
from copy import deepcopy
from runIAC import *
import multiprocessing, json

def updateS (S, Vcur):
    u = random.choice(list(Vcur))
    S.append(u)
    Vcur.remove(u)

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
    ALGO_NAME = "RDM"
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

        print 'Start Initialization for RDM...'
        S = []
        Vcur = set(G.nodes())
        print 'Finished initialization'


        print 'Selecting seed set S...'
        time2select = time.time()
        # add first node to S
        updateS(S, Vcur)
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
                updateS(S, Vcur)
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