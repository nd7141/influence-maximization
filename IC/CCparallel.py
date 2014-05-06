'''
The same as CCHeuristic.py but tuned for multiprocessing.
Instead of returning seed set, it returns scores that are later reduced.
Runs only one iteration R.
'''

from __future__ import division

import random
from copy import deepcopy

def CC_parallel(G, k, p):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     k -- number of nodes in seed set (int)
     p -- propagation probability among all edges (int)
     Output:
     scores -- scores of nodes according to some weight function (dict)
    '''
    scores = dict(zip(G.nodes(), [0]*len(G))) # initialize scores

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

    # find ties for components of rank k and add them all as qualified
    sortedCC = sorted([(len(dv), dk) for (dk, dv) in CC.iteritems()], reverse=True)
    topCCnumbers = sortedCC[:k]
    L = sum([l for (l,_) in topCCnumbers])
    # add ties of rank k
    increment = 0
    while sortedCC[k + increment][0] == sortedCC[k-1][0]:
        topCCnumbers.append(sortedCC[k + increment])
        increment += 1

    # assign scores to nodes in top Connected Components
    # prev_length  = topCCnumbers[0][0]
    # rank = 1
    for length, numberCC in topCCnumbers:
        # if length != prev_length:
        #     prev_length = length
        #     rank += 1
        # weighted_score = (1-k)/(length - k + (1 - length)*rank)
        weighted_score = 1.0/length
        for node in CC[numberCC]:
            scores[node] += weighted_score

    return scores