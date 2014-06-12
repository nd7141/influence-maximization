from __future__ import division
from priorityQueue import PriorityQueue as PQ # priority queue
import networkx as nx
from runIAC import *
import json, os

def DD(G, k, Ep):
    ''' Degree Discount heuristic for AIP cascade (naive approach).
    Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    Ep -- propagation probabilities
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # priority = d[u]
        priority = (1 + sum([Ep[(u,v)]*G[u][v]['weight'] for v in G[u]]))
        dd.add_task(u, -priority) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                p = Ep[(u,v)]
                # priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p# discount of degree
                priority = (1-p)**t[v]*(1 + (d[v] - t[v])*p)
                dd.add_task(v, -priority)
    return S

def GDD(G, k, Ep):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    Ep -- propagation probabilities
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    active = dict()
    inactive = dict()

    # initialize degree discount
    for u in G:
        active[u] = 1
        inactive[u] = sum([Ep[(u,v)]*G[u][v]['weight'] for v in G[u]])
        priority = active[u]*(1 + inactive[u])
        dd.add_task(u, -priority) # add degree of each node

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                active[v] *= (1-Ep[(u,v)])**G[u][v]['weight']
                inactive[v] -= Ep[(u,v)]*G[u][v]['weight']
                priority = active[v]*(1 + inactive[v])
                dd.add_task(v, -priority)
    return S

# range for floats: http://stackoverflow.com/a/7267280/2069858
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
    I = 250
    ftime = open('plotdata/timeDirectGDDforDirect2.txt', 'w+')

    # get propagation probabilities
    Ep = dict()
    with open("Ep_hep_random1.txt") as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    length_to_coverage = {0:0}
    l2c = [[0,0]]
    for length in range(1,251,5):

        time2length = time.time()

        print 'Start finding solution for length = %s' %length
        time2S = time.time()
        S = GDD(G, length, Ep)
        print S
        print >>ftime, "%s %s" %(length, time.time() - time2S)
        print 'Finish finding S in %s sec...' %(time.time() - time2S)

        print 'Start calculating coverage...'
        def map_AvgIAC (it):
            return avgIAC(G, S, Ep, I)
        avg_size = 0
        time2avg = time.time()
        T = map(map_AvgIAC, range(4))
        # print T
        avg_size = sum(T)/len(T)
        print 'Average coverage of %s nodes is %s' %(length, avg_size)
        print 'Finished calculating coverage in', time.time() - time2avg

        length_to_coverage[length] = avg_size
        l2c.append([length, avg_size])
        with open("plotdata/plotDirectGDDforDirect2.txt", "w+") as fp:
            json.dump(l2c, fp)

        print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        print '----------------------------------------------'

    print 'Total time: %s' %(time.time() - start)

    console = []