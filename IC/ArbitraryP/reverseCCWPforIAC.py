from __future__ import division
from copy import deepcopy
import time, random, operator, os, json
import networkx as nx
import multiprocessing
import runIAC

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

def reverseCCWP(G, tsize, Ep, min_length, r):
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

    fileno = 1
    tsize = 150

    # r = 2
    Ep = runIAC.uniformEp(G, .01)
    R = 500
    I = 250
    e_results = dict()
    best_r = -1
    best_S = []
    min_lenS = float("Inf")
    pool2algo = None
    pool2average = None

    time2preprocess = time.time()
    print 'Preprocessing to find minimal size of CC...'
    min_length = float("Inf")
    for it in range(20):
        CC = findCC(G, Ep)
        L, sortedCC = findL(CC, tsize)
        Llength = sortedCC[L-1][0]
        if Llength < min_length:
            min_length = Llength
    print 'Min |L|:', min_length
    print 'Finished preprocessing in % sec' %(time.time() - time2preprocess)

    for e in frange(1,1.5,.5):
        r_results = dict()
        for r in frange(1.,1.1,.2):
            time2r = time.time()
            print 'Finding solution for ratio e = %s, r = %s...' %(e,r)

            def mapAvgSize (S):
                return runIAC.avgIAC(G, S, Ep, I)
            def mapReverseCCWP (it):
                # print it
                return reverseCCWP(G, tsize, Ep, min_length, r)
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

            time2select = time.time()
            print 'Start selecting seed set S...'
            # select first top-L nodes with penalization
            scores_copied = deepcopy(scores)
            S = []
            for i in range(int(maxL)): # change range limit to minL, maxL, avgL, 1, etc.
                maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv) # top node in the order
                S.append(maxk)
                scores_copied.pop(maxk)
                for v in G[maxk]:
                    if v not in S:
                        p = Ep[(maxk,v)]
                        penalty = (1-p)**(e*G[maxk][v]['weight'])
                        scores_copied[v] *= penalty
            with open('../reverseCCWP/reverseCCWP%s.txt' %fileno, 'w+') as f:
                f.write('e: ' + str(e) + ' r:' + str(r) + os.linesep)
            # calculate spread for top-L nodes
            T = pool2average.map(mapAvgSize, [S]*4)
            coverage = sum(T)/len(T)
            with open('../reverseCCWP/reverseCCWP%s.txt' %fileno, 'a+') as f:
                f.write('|S| = %s --> %s' %(len(S), coverage) + os.linesep)
            print '|S| = %s --> %s' %(len(S), coverage)
            # add new node by one penalizing scores at the same time
            while coverage < tsize:
                maxk, maxv = max(scores_copied.iteritems(), key = operator.itemgetter(1))
                S.append(maxk)
                scores_copied.pop(maxk)
                T = pool2average.map(mapAvgSize, [S]*4)
                coverage = sum(T)/len(T)
                with open('../reverseCCWP/reverseCCWP%s.txt' %fileno, 'a+') as f:
                    f.write('|S| = %s --> %s' %(len(S), coverage) + os.linesep)
                print '|S| = %s --> %s' %(len(S), coverage)
                for v in G[maxk]:
                    if v not in S:
                        p = Ep[(maxk, v)]
                        penalty = (1-p)**(e*G[maxk][v]['weight'])
                        scores_copied[v] *= penalty
            print 'Finished selecting seed set S in %s sec' %(time.time() - time2select)

            print 'S:', S
            print 'len(S):', len(S)
            r_results[round(r,1)] = len(S)
            # write results to file
            print 'Writing result to file...'
            with open('../reverseCCWP/reverseCCWP%s.txt' %fileno, 'a+') as f:
                f.write('S: ' + json.dumps(S) + os.linesep)
                f.write('|S|:' + str(len(S)))
            fileno += 1
            # compare to best found result so far
            if len(S) < min_lenS:
                min_lenS = len(S)
                best_S = deepcopy(S)
                best_er = (e,r)
                print 'New best (e,r) = %s with len(S) = %s' %(best_er, min_lenS)
            print 'Time for e = %s, r = %s: %s sec' %(e, r, time.time() - time2r)
            print '--------------------------------------------------------------------------------'
        e_results[e] = r_results

    print 'Best S:', best_S
    print 'len(S):', min_lenS
    print 'Best er:', best_er
    print 'e_results:', sorted(e_results.iteritems(), key = lambda (dk, dv): dk)
    print 'Total time: %s sec' %(time.time() - start)

    console = []
