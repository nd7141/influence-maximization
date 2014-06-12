from __future__ import division
from copy import deepcopy
import time, random, operator, os, json
import networkx as nx
import multiprocessing, numpy
from runIAC import *
import matplotlib.pyplot as plt

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

def findL(CC, tsize):
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
    return L, sortedCC

def reverseCCWP(G, tsize, Ep, min_length):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     tsize -- coverage size (int)
     Ep -- propagation probabilities (dict)
     r -- ratio for selecting number of components (float)
     Output:
     S -- seed set
    '''
    scores = dict() # initialize scores

    # find CC
    CC = findCC(G, Ep)

    # find L and sortedCC
    L, sortedCC = findL(CC, tsize)

    # find number of CC assign scores to
    total_number_of_CC = 0
    N = 0
    for length, _ in sortedCC:
        if length >= min_length:
            total_number_of_CC += 1
            N += length
        else:
            break

    for length, numberCC in sortedCC[:total_number_of_CC]:
        weighted_score = 1.0/(length*total_number_of_CC)
        # weighted_score = 1
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

    R = 500
    I = 250
    tsize = 351
    best_S = []
    min_lenS = float("Inf")
    pool2algo = None
    pool2average = None

    length_to_coverage = {0:0}
    norm_parameters = dict()

    # get propagation probabilities
    Ep = dict()
    with open("Ep_hep_random1.txt") as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    time2preprocess = time.time()
    print 'Preprocessing to find minimal size of CC...'
    min_length = float("Inf")
    # find min_length to select CC within
    for length_it in range(5):
        CC = findCC(G, Ep)
        L, sortedCC = findL(CC, tsize)
        Llength = sortedCC[L-1][0]
        if Llength < min_length:
            min_length = Llength
    print 'Min |L|:', min_length
    print 'Finished preprocessing in %s sec' %(time.time() - time2preprocess)

    def mapAvgSize (S):
        return avgIAC(G, S, Ep, I)
    def mapReverseCCWP (it):
        return reverseCCWP(G, tsize, Ep, min_length)
    if pool2algo == None:
        pool2algo = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    if pool2average == None:
        pool2average = multiprocessing.Pool(processes=multiprocessing.cpu_count())

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
    print 'minL', minL
    print 'maxL', maxL

    time2select = time.time()
    print 'Start selecting seed set S...'

    # select first top-L nodes with penalization
    scores_copied = deepcopy(scores)
    S = []
    for i in range(int(minL)): # change range limit to minL, maxL, avgL, 1, etc.
        maxk, maxv = max(scores_copied.iteritems(), key=lambda (dk, dv): dv) # top node in the order
        S.append(maxk)
        scores_copied.pop(maxk)
        for v in G[maxk]:
            if v not in S:
                p = Ep[(maxk,v)]
                penalty = (1-p)**(G[maxk][v]['weight'])
                scores_copied[v] *= penalty
    # calculate spread for top-L nodes
    T = pool2average.map(mapAvgSize, [S]*4)
    coverage = sum(T)/len(T)
    print '|S| = %s --> %s' %(len(S), coverage)

    # add new nodes using binary search
    # first, search for 2 boundaries
    Coverages = dict()
    Low = len(S)
    lastS = deepcopy(S)
    Coverages[len(lastS)] = coverage
    while coverage < tsize:
        Low = len(lastS)
        High = 2*Low
        # select new nodes
        while len(lastS) < 2*Low:
            maxk, maxv = max(scores_copied.iteritems(), key = operator.itemgetter(1))
            lastS.append(maxk)
            scores_copied.pop(maxk)
            for v in G[maxk]:
                if v not in lastS:
                    p = Ep[(maxk, v)]
                    penalty = (1-p)**(G[maxk][v]['weight'])
                    scores_copied[v] *= penalty
        T = pool2average.map(mapAvgSize, [lastS]*4)
        coverage = sum(T)/len(T)
        print '|S| = %s --> %s' %(len(lastS), coverage)
        Coverages[len(lastS)] = coverage

    # second, search for minimal number of nodes
    S = deepcopy(lastS)
    step = len(S) - Low
    while step > 1:
        if coverage <= tsize:
            # calculate new S
            Low = len(S)
            new_length = Low + (High - len(S))//2
            S = lastS[:new_length]
            # calculate coverage
            T = pool2average.map(mapAvgSize, [S]*4)
            coverage = sum(T)/len(T)
            print '|S| = %s --> %s' %(len(S), coverage)
            Coverages[len(S)] = coverage
            # choose step
            if coverage >= tsize:
                step = min(len(S) - Low, High - len(S))
            else:
                step = max(len(S) - Low, High - len(S))
        else:
            # calculate new S
            High = len(S)
            new_length = High - (len(S) - Low)//2
            S = lastS[:new_length]
            # calculate coverage
            T = pool2average.map(mapAvgSize, [S]*4)
            coverage = sum(T)/len(T)
            print '|S| = %s --> %s' %(len(S), coverage)
            Coverages[len(S)] = coverage
            # choose step
            if coverage < tsize:
                step = min(len(S) - Low, High - len(S))
            else:
                step = max(len(S) - Low, High - len(S))

    # additional check that we achieved coverage
    if coverage <= tsize:
        print 'Increase S by 1'
        S = lastS[:new_length + 1]
    finalS = deepcopy(S)
    print finalS
    print 'Necessary %s initial nodes to target %s nodes in graph G' %(len(finalS), tsize)
    print 'Finished selecting seed set S in %s sec' %(time.time() - time2select)
    # with open("plotdata/timeReverseCCWPforReverse3.txt", "w+") as fp:
    #     fp.write("%s" %(time.time() - time2select))

    with open("plotdata/rawCCWPforDirect2.txt", "a+") as f:
        json.dump({tsize: finalS}, f)
        print >>f

    with open("plotdata/rawCCWPTimeforDirect2.txt", "a+") as f:
        json.dump({tsize: time.time() - start}, f)
        print >>f

    # # map length: [0,len(S)] to coverage
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
    #         T = pool2average.map(mapAvgSize, [finalS[:length]]*4)
    #         coverage = sum(T)/len(T)
    #         length_to_coverage[length] = length_to_coverage.get(length, 0) + coverage
    #         print '|S|: %s --> %s' %(length, coverage)
    #
    # # if we haven't added result for tsize then add it
    # if (len(finalS) - 1)%step != 0:
    #     norm_parameters[len(finalS)] = norm_parameters.get(len(finalS),0) + 1
    #     length_to_coverage[len(finalS)] = length_to_coverage.get(len(finalS), 0) + Coverages[len(finalS)]
    #     print '|S|: %s --> %s' %(len(finalS), Coverages[len(finalS)])
    #
    # # normalizing coverages
    # for length in norm_parameters:
    #     length_to_coverage[length] /= norm_parameters[length]
    #
    # length_to_coverage = sorted(length_to_coverage.iteritems(), key = lambda (dk, dv): dk)
    #
    # with open("plotdata/plotReverseCCWPforReverse3.txt", "w+") as fp:
    #     json.dump(length_to_coverage, fp)
    #
    print 'Total time: %s sec' %(time.time() - start)

    console = []
