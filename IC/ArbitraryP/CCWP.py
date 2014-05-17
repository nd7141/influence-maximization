
#TODO implement CCWP for arbitrary probabilities using CCparallel and test_in_parallel.py
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

def CCWP(G, k, Ep):
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
    # add ties of rank k
    increment = 0
    while sortedCC[k + increment][0] == sortedCC[k-1][0]:
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

def frange(begin, end, step):
    x = begin
    y = end
    while x < y:
        yield x
        x += step


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
                G.add_edge(u,v, weight=1)
            # G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    #calculate initial set
    seed_size = 60
    R = 500
    I = 250
    fileno = 0

    beste = -1
    maxIS = -1

    # get propagation probabilities
    Ep = random_from_range(G, [.01, .02, .04, .08])

    pool = pool2 = None
    for e in frange(1,1.5,.5):
        time2e = time.time()
        print 'Start finding solution for e = %s' %e
        print 'Start mapping...'
        time2map = time.time()
        # define map function
        def map_CCWP(it):
            # print it
            return CCWP(G, seed_size, Ep)
        if pool == None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        Scores = pool.map(map_CCWP, range(R))
        print 'Finished mapping in', time.time() - time2map
        time2reduce = time.time()
        print 'Reducing scores...'
        scores = {v: sum([s[v] for s in Scores]) for v in G}
        scores_copied = deepcopy(scores)
        S = []
        # penalization phase
        for it in range(seed_size):
            maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv)
            # print maxv,
            S.append(maxk)
            scores_copied.pop(maxk) # remove top element from dict
            for v in G[maxk]:
                if v not in S:
                    # weight = scores_copied[v]/maxv
                    # print weight,
                    penalty = (1-Ep[(maxk, v)])**(e*G[maxk][v]['weight'])
                    scores_copied[v] = penalty*scores_copied[v]
        print S
        # topScores = nlargest(seed_size, scores.iteritems(), key = lambda (dk,dv): dv)
        # S = [v for (v,_) in topScores]
        def map_AvgIAC (it):
            # print it
            return avgIAC(G, S, Ep, I)
        if pool2 == None:
            pool2 = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        print 'Finished reducing in', time.time() - time2reduce
        avg_size = 0
        time2avg = time.time()
        T = pool2.map(map_AvgIAC, range(4))
        print 'Finished averaging seed set size in', time.time() - time2avg
        print T
        avg_size = sum(T)/len(T)
        print 'Average size of %s nodes is %s' %(seed_size, avg_size)

        # write results to file
        print 'Writing result to file...'
        with open('../directCCWP/reverseCCWP%s.txt' %fileno, 'w+') as f:
                f.write('e: ' + str(e) + ' k: ' + str(seed_size) + os.linesep)
                f.write('S: ' + json.dumps(S) + os.linesep)
                f.write('IS:' + str(avg_size))
        fileno += 1

        if avg_size >= maxIS:
            print 'Changing best e to %s' %e
            maxIS = avg_size
            beste = e
            bestS = S
        print 'Total time for e = %s: %s sec' %(e, time.time() - time2e)
        print '----------------------------------------------'
        # delete existing variables
        # pool = None
        # pool2 = None

    print 'maxIS:', maxIS
    print 'beste:', beste
    print 'bestS:', bestS
    print '|S|:', len(S)

    print 'Total time: %s' %(time.time() - start)

    console = []