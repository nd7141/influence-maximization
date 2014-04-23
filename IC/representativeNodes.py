''' Implementation of heuristic that searches for broadest set of vertices
of size k using priority queues
'''
__author__ = 'ivanovsergey'

from priorityQueue import PriorityQueue as PQ # priority queue class

def _sumDist (G, S, no):
    ''' Compute cumulative distance from node to set S
    '''
    cum = 0
    for u in S:
        try:
            cum += G[no][u]['weight']
        except:
            pass
    return cum

def _minDist (G, S, no):
    ''' Compute minimal distance from node to S
    '''
    min_dist = float('Inf')
    for u in S:
        try:
            min_dist = min(G[no][u]['weight'], min_dist)
        except:
            pass
    if min_dist == float('Inf'):
        return -1
    else:
        return min_dist

def representativeNodes(G, k, metric=1):
    ''' Finds the most distinguishable (representative) nodes in graph G greedily.
    Takes the most furthest node to the already chosen nodes at each step.

    Input: G -- networkx object graph with weighted edges
    k -- number of nodes needed
    metric -- parameter for differentiating representative qualities
    metric == 1 trying to maximize total distance in the chosen set of k nodes
    metric == 2 trying to maximize minimal distance between a pair of k nodes
    Output:
    S -- chosen k nodes
    objv -- objective value according to the chosen metric and set of nodes
    '''

    S = [] # set of chosen nodes
    S_dist = PQ() # distances from each node in G to set S according to metric

    # initialize S with furthest vertices
    try:
        u,v,d = max(G.edges(data=True), key=lambda (u, v, d): d['weight'])
    except KeyError:
        raise KeyError, 'Most likely you have no weight attribute'
    S.extend([u,v])

    # compute distances from each node in G to S
    for v in G.nodes():
        if v not in S: # calculate only for nodes in G
            if metric == 1:
                S_dist.add_task(v, - _sumDist(G, S, v)) # take minus to pop the maximum value from priority queue
            elif metric == 2:
                S_dist.add_task(v, - _minDist(G, S, v)) # take minus to pop the maximum value from priority queue

    # add new nodes to the set greedily
    while len(S) < k:
        u, priority = S_dist.pop_item() # find maximum value of distance to set S
        S.append(u) # append that node to S

        # only increase distance for nodes that are connected to u
        for v in G[u].keys():
            if v not in S: # add only remained nodes
                [priority, count, task] = S_dist.entry_finder[v] # finds distance for the previous step
                try:
                    if metric == 1:
                        S_dist.add_task(v, priority-G[u][v]['weight']) # adds distance to the new member of S
                    elif metric == 2:
                        S_dist.add_task(v, max(priority, -G[u][v]['weight'])) # update min distance to the set S
                except:
                    raise u,v, "These are vertices that caused the problem"

    # extract objective value of the chosen set
    if metric == 1:
        objv = 0
        for u in S:
            objv += _sumDist(G, S, u)
    elif metric == 2:
        objv = float('Inf')
        for u in S:
            objv = min(objv, _minDist(G, S, u))

    return S, objv

if __name__ == '__main__':
    console = []