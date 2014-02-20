''' Implementation of degree discount heuristic [1] for Independent Cascade model
of influence propagation in graph G

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
'''
__author__ = 'ivanovsergey'

from priorityQueue import PriorityQueue as PQ # priority queue

def degreeDiscountIC (G, k, p=.1):
    ''' Finds initial set of nodes to propagate in Independent Cascade model
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S

    # initialize degree discount
    for u in G.nodes():
        dd.add_task(u, -len(G[u])) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u].keys() :
            if v in dd.entry_finder:
                t[v] += 1
                priority = (len(G[v].keys()) - 2*t[v] - (len(G[v].keys()) - t[v])*t[v]*p)
                dd.add_task(v, -priority)
    return S

if __name__ == '__main__':
    console = []