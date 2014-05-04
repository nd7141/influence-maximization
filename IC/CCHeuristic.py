''' Implementation of heuristic based on Connected Components (CC) for undirected graph.
We first delete an edge from original graph with probability (1-p)**w.
Then we calculate SCC for this graph.
Then for all vertices in a component we add to its score the number of nodes in this component.
Procedure repeats R times to get some average.
Then k nodes with top scores are selected.

References:
Kempe et al. "Maximizing the spread of influence through a social network" Claim 2.3
'''

from __future__ import division

import random, operator, time, os
from heapq import nlargest
from copy import deepcopy

def CC_heuristic (G, k, p, R=20):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     k -- number of nodes in seed set (int)
     p -- propagation probability among all edges (int)
     R -- number of iterations to estimate scores of nodes (int)
     Output:
     S -- seed set (list of tuples: for each tuple first argument is a node, second argument is its score)
    '''
    scores = dict(zip(G.nodes(), [0]*len(G))) # initialize scores
    start = time.time()
    for it in range(R):
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

        # add score only to top components
        topCC = nlargest(k, CC.iteritems(), key= lambda (dk,dv): len(dv))
        for (c, component) in topCC:
            # print c, len(component)
            weighted_score = 1.0/len(component)**(.5)
            # weighted_score = 1
            for node in component:
                if random.random() < weighted_score:
                    scores[node] += weighted_score

        # update scores
        # for c in CC:
        #     weighted_score = len(CC[c]) # score is size of a component
        #     for node in CC[c]:
        #         if random.random() < 1.0/weighted_score:
        #             scores[node] += weighted_score
        print it + 1, time.time() - start
    S = nlargest(k, scores.iteritems(), key=operator.itemgetter(1)) # select k nodes with top scores
    return S
