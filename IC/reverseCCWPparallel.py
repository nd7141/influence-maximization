from __future__ import division
from copy import deepcopy
import time, random, operator, os, json
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
    # find number of next-i CCs
    nexti_length = sortedCC[int(r*L) + 0][0] # change to 1,2,3, ..., i
    cur_length = nexti_length
    number_of_ties = 0 # number of additional CCs to assign score to
    while cur_length == nexti_length:
        number_of_ties += 1
        cur_length = sortedCC[int(r*L) + number_of_ties][0]
    # assign scores to rL + next-i components
    numer = 0
    for length, numberCC in sortedCC[:int(r*L) + number_of_ties]:
        print numer, length
        numer += 1
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

    fileno = 0
    tsize = 150
    p = .01
    # r = 2
    R = 10
    I = 20
    e_results = dict()
    best_r = -1
    best_S = []
    min_lenS = float("Inf")
    for e in frange(1,1.2,.1):
        r_results = dict()
        for r in frange(2.9,3.1,.1):
            time2r = time.time()
            print 'Finding solution for ratio r = %s...' %r

            def mapAvgSize (S):
                return avgSize(G, S, p, I)
            def mapReverseCCWP (it):
                print it
                return reverseCCWP(G, tsize, p, r)
            pool2algo = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
            pool2average = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

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
                avgL += L/len(result)
                if L > maxL:
                    maxL = L
                if L < minL:
                    minL = L
                for (node, score) in Sc.iteritems():
                    scores[node] += score
            print 'Finished reducing in %s sec' %(time.time() - time2reduce)
            print 'avgL', avgL

            time2select = time.time()
            print 'Start selecting seed set S...'
            # select first top-L nodes with penalization
            scores_copied = deepcopy(scores)
            S = []
            for i in range(int(1.5*avgL)): # change range limit to minL, maxL, avgL, 1, etc.
                maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv) # top node in the order
                S.append(maxk)
                scores_copied.pop(maxk)
                for v in G[maxk]:
                    if v not in S:
                        penalty = (1-p)**(e*G[maxk][v]['weight'])
                        scores_copied[v] *= penalty
            # calculate spread for top-L nodes
            T = pool2average.map(mapAvgSize, [S]*4)
            coverage = sum(T)/len(T)
            print '|S| = %s --> %s' %(len(S), coverage)
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
                        penalty = (1-p)**(e*G[maxk][v]['weight'])
                        scores_copied[v] *= penalty
            print 'Finished selecting seed set S in %s sec' %(time.time() - time2select)

            print 'S:', S
            print 'len(S):', len(S)
            r_results[round(r,1)] = len(S)
            # write results to file
            print 'Writing result to file...'
            with open('reverseCCWP/reverseCCWP%s.txt' %fileno, 'w+') as f:
                f.write('e: ' + str(e) + ' r:' + str(r) + os.linesep)
                f.write('S: ' + json.dumps(S) + os.linesep)
                f.write('|S|:' + str(len(S)))
            fileno += 1
            # compare to best found result so far
            if len(S) < min_lenS:
                min_lenS = len(S)
                best_S = deepcopy(S)
                best_er = (e,r)
                print 'New best (e,r) = %s with len(S) = %s' %(best_er, min_lenS)
            print 'Time for r = %s: % sec' %(r, time.time() - time2r)
            print '--------------------------------------------------------------------------------'
        e_results[e] = r_results

    print 'Best S:', best_S
    print 'len(S):', min_lenS
    print 'Best er:', best_er
    print 'e_results:', sorted(e_results.iteritems(), key = lambda (dk, dv): dk)
    print 'Total time: %s sec' %(time.time() - start)

    console = []
