''' File for testing different files in parallel
'''

import networkx as nx

from IC import runIC, avgSize
from CCparallel import CC_parallel
import multiprocessing
from heapq import nlargest

#import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    import time
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
                G.add_edge(u,v, weight=1)
            # G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - start

    #calculate initial set
    seed_size = 50

    print 'Start mapping...'
    time2map = time.time()
    R = 200
    # define map function
    def map_CC(it):
        print it
        return CC_parallel(G, seed_size, .01)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()*2)
    Scores = pool.map(map_CC, range(R))
    print 'Finished mapping in', time.time() - time2map
    time2reduce = time.time()
    print 'Reducing scores...'
    scores = {v: sum([s[v] for s in Scores]) for v in G}
    topScores = nlargest(seed_size, scores.iteritems(), key = lambda (dk,dv): dv)
    S = [v for (v,_) in topScores]
    print 'Time to reduce', time.time() - time2reduce

    print 'Average size is', avgSize(G,S,.01,200)
    print 'Average size of 10 nodes is', avgSize(G,S[:10],.01,200)

    print 'Total time:', time.time() - start

    # # write results S to file
    # with open('visualisation.txt', 'w') as f:
    #     for node in S:
    #         f.write(str(node) + os.linesep)

    console = []
