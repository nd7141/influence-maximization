''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)
'''
from __future__ import division
from copy import deepcopy # copy graph object
from random import random
from priorityQueue import PriorityQueue as PQ
import networkx as nx
from runIAC import avgIAC


def bfs(E, S):
    ''' Finds all vertices reachable from subset S in graph E using Breadth-First Search
    Input: E -- networkx graph object
    S -- list of initial vertices
    Output: Rs -- list of vertices reachable from S
    '''
    Rs = []
    for u in S:
        if u in E:
            if u not in Rs: Rs.append(u)
            for v in E[u].keys():
                if v not in Rs: Rs.append(v)
    return Rs

def findCCs(G, Ep):
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random.random() < (1-Ep[e])**(E[e[0]][e[1]]['weight'])]
    E.remove_edges_from(edge_rem)

    # initialize CC
    CCs = dict() # each component is reflection of the number of a component to its members
    explored = dict(zip(E.nodes(), [False]*len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CCs[c] = [node]
            component = E[node].keys()
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CCs[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    return CCs

def newGreedyIC (G, k, Ep, R = 20):

    S = []
    for i in range(k):
        print i
        time2k = time.time()
        scores = {v: 0 for v in G}
        for j in range(R):
            print j,
            CCs = findCCs(G, Ep)
            for CC in CCs:
                for v in S:
                    if v in CC:
                        break
                else: # in case CC doesn't have node from S
                    for u in CC:
                        scores[u] += float(len(CC))/R
        max_v, max_score = max(scores.iteritems(), key = lambda (dk, dv): dv)
        S.append(max_v)
        print
        print time.time() - time2k
    return S

# def newGreedyIC(G, k, p=.01):
#     ''' Finds initial set of nodes to propagate in Independent Cascade.
#     Input: G -- networkx graph object
#     k -- number of nodes needed
#     p -- propagation probability
#     Output: S -- set of k nodes chosen
#     '''
#
#     import time
#     start = time.time()
#
#     # assert type(S0) == list, "S0 must be a list. %s provided instead" % type(S0)
#     # S = S0 # set of selected nodes
#     # if len(S) >= k:
#     #     return S[:k]
#
#     S = []
#
#     iterations = k - len(S)
#     print 'iterations =', iterations
#     for i in range(iterations):
#         # s = PQ() # number of additional nodes each remained mode will bring to the set S in R iterations
#         s = dict()
#         Rv = dict() # number of reachable nodes for node v
#         # initialize values of s
#         for v in G:
#             if v not in S:
#                 # s.add_task(v, 0)
#                 s[v] = 0
#
#         # calculate potential additional spread for each vertex not in S
#         prg_idx = 1
#         idx = 1
#         prcnt = .1 # for progress to print
#         R = 20 # number of iterations to run RanCas
#         # spread from each node individually in pruned graph E
#         # Rv = dict()
#         # for v in G:
#         #     if v not in S:
#         #         Rv[v] = 0
#         for j in range(R):
#             # create new pruned graph E
#             E = deepcopy(G)
#             edge_rem = [] # edges to remove
#             for (u,v) in E.edges():
#                 w = G[u][v]['weight']
#                 if random() < (1 - p)**w:
#                     edge_rem.append((u,v))
#             E.remove_edges_from(edge_rem)
#             # find reachable vertices from S
#             # TODO make BFS happens only once for all nodes. Should take O(m) time.
#             Rs = bfs(E, S)
#             # find additional nodes each vertex would bring to the set S
#             time2update = time.time()
#             for v in G:
#                 if v not in S + Rs: # if node has not chosen in S and has chosen by spread from S
#                     # Rv[v] += float(len(bfs(E, [v])))/R
#                     # [priority, c, task] = s.entry_finder[v]
#                     # s.add_task(v, priority - float(len(bfs(E, [v])))/R)
#                     s[v] -= float(len(bfs(E, [v])))/R
#             # print 'Took %s sec to update' %(time.time() - time2update)
#
#             if idx == int(prg_idx*prcnt*R):
#                 print '%s%%...' %(int(prg_idx*prcnt*100)), time.time() - start
#                 prg_idx += 1
#             idx += 1
#         # add spread of nodes in G'
#         # for v in Rv:
#         #     s.add_task(v, -Rv[v])
#         # add vertex with maximum potential spread
#         time2min = time.time()
#         # task, priority = s.pop_item()
#         task, priority = min(s.iteritems(), key=lambda (dk,dv): dv)
#         s.pop(task)
#         # print 'Took %s sec to find min' %(time.time() - time2min)
#         S.append(task)
#         print i, k, task, -priority, time.time() - start
#     return S

if __name__ == "__main__":
    import time
    start = time.time()

    G = nx.read_gpickle("../../graphs/hep.gpickle")
    print 'Read graph G'
    print time.time() - start

    model = "MultiValency"

    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    # get propagation probabilities
    Ep = dict()
    with open("Ep_hep_%s1.txt" %ep_model) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    I = 1000

    S = newGreedyIC(G, 10, Ep)
    print S
    print avgIAC(G, S, Ep, I)