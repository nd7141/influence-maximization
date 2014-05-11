from __future__ import division
from copy import deepcopy
import time, random, operator
from heapq import nlargest
from IC import avgSize
import networkx as nx
import multiprocessing

def reverseCCWP(G, tsize, p, r):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     tsize -- coverage size (int)
     p -- propagation probability among all edges (int)
     r -- ratio for selecting number of components (float)
     Output:
     S -- seed set
    '''
    scores = dict() # initialize scores
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1-p)**(E[e[0]][e[1]]['weight'])]
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

    # find top components that can reach tsize activated nodes
    sortedCC = sorted([(len(dv), dk) for (dk, dv) in CC.iteritems()], reverse=True)
    cumsum = 0 # sum of top components
    L = 0 # current number of CC that achieve tsize
    # find L first
    for length, numberCC in sortedCC:
        L += 1
        cumsum += length
        if cumsum >= tsize:
            break
    # assign scores to L components
    for length, numberCC in sortedCC[:int(r*L)]:
        weighted_score = 1.0/(length*L)
        for node in CC[numberCC]:
            scores[node] = weighted_score
    return scores, L

# range for floats: http://stackoverflow.com/a/7267280/2069858
def frange(begin, end, step):
    x = begin
    y = end
    while x < y:
        yield x
        x += step

if __name__ == "__main__":
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('../graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    tsize = 150
    p = .01
    # r = 2
    R = 1000
    I = 250
    r_results = dict()
    best_r = -1
    best_S = []
    min_lenS = float("Inf")
    for r in frange(1.5,3.1,.1):
        time2r = time.time()
        print 'Finding solution for ratio r = %s...' %r

        def mapAvgSize (S):
            return avgSize(G, S, p, I)
        def mapReverseCCWP (it):
            print it
            return reverseCCWP(G, tsize, p, r)
        pool2algo = multiprocessing.Pool(processes=multiprocessing.cpu_count()*2)
        pool2average = multiprocessing.Pool(processes=multiprocessing.cpu_count()*2)

        time2map = time.time()
        print 'Start mapping...'
        result = pool2algo.map(mapReverseCCWP, range(R)) # result is [(scores1, L1), (scores2, L2), ...]
        print 'Finished mapping in %s sec' %(time.time() - time2map)

        time2reduce = time.time()
        print 'Start reducing scores...'
        scores = dict(zip(G.nodes(), [0]*len(G)))
        maxL = -1
        minL = float("Inf")
        avgL = 0
        for (Sc, L) in result:
            avgL += L
            if L > maxL:
                maxL = L
            if L < minL:
                minL = L
            for (node, score) in Sc.iteritems():
                scores[node] += score
        print 'Finished reducing in %s sec' %(time.time() - time2reduce)

        time2select = time.time()
        print 'Start selecting seed set S...'
        # select first nodes
        orderedScores = sorted(scores.iteritems(), key = operator.itemgetter(1), reverse=True)
        topScores = orderedScores[:minL] # change minL to maxL, avgL, r*avgL, 1, etc.
        S = [node for (node,_) in topScores]
        T = pool2average.map(mapAvgSize, [S]*4)
        coverage = sum(T)/len(T)
        print '|S| = %s --> %s' %(len(S), coverage)
        # Penalization phase
        scores_copied = deepcopy(scores)
        # remove all nodes that are already in S
        for node in S:
            scores_copied.pop(node)
        # add new node by one penalizing scores at the same time
        while coverage < tsize:
            maxk, maxv = max(scores_copied.iteritems(), key = operator.itemgetter(1))
            S.append(maxk)
            scores_copied.pop(maxk)
            T = pool2average.map(mapAvgSize, [S]*4)
            coverage = sum(T)/len(T)
            print '|S| = %s --> %s' %(len(S), coverage)
            for v in G[maxk]:
                if v not in S:
                    penalty = (1-p)**G[maxk][v]['weight']
                    scores_copied[v] *= penalty
        print 'Finished selecting seed set S in %s sec' %(time.time() - time2select)

        print 'S:', S
        print 'len(S):', len(S)
        r_results[round(r,1)] = len(S)
        if len(S) < min_lenS:
            min_lenS = len(S)
            best_S = deepcopy(S)
            best_r = r
            print 'New best r = %s with len(S) = %s' %(best_r, min_lenS)
        print 'Time for r = %s: % sec' %(r, time.time() - time2r)
        print '--------------------------------------------------------------------------------'

    print 'Best S:', best_S
    print 'len(S):', min_lenS
    print 'Best r:', best_r
    print 'r_results:', r_results
    print 'Total time: %s sec' %(time.time() - start)

    console = []
