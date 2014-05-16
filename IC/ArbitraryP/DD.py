from __future__ import division
from priorityQueue import PriorityQueue as PQ # priority queue

def DD(G, k, Ep):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    Ep -- propagation probabilities
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    # dd = dict()
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        # dd[u] = -d[u]
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        # time2min = time.time()
        u, priority = dd.pop_item() # extract node with maximal degree discount
        # u, priority = min(dd.iteritems(), key = lambda(dk,dv): dv)
        # dd.pop(u)
        # print 'Took %s sec to find min' %(time.time() - time2min)
        S.append(u)
        neighbors_weights = [G[u][v]['weight'] for v in G[u] if v not in S]
        w_avg = sum(neighbors_weights)/len(neighbors_weights)
        # time2update = time.time()
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                p = Ep[(u,v)]
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p**w_avg# discount of degree
                dd.add_task(v, -priority)
                # dd[v] = -priority
        # print 'Took %s sec to update' %(time.time() - time2update)
    return S
