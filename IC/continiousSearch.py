
from copy import deepcopy
import random
from heapq import nlargest
import time

def continiousSearch (G, k, p, R=200):
    scores = dict(zip(G.nodes(), [0]*len(G)))
    for it in range(R):
        time2iteration = time.time()
        # remove blocked edges from graph G
        E = deepcopy(G)
        edge_rem = [e for e in E.edges() if random.random() < (1-p)**(E[e[0]][e[1]]['weight'])]
        E.remove_edges_from(edge_rem)

        # initialize CC
        CC = dict() # each component is reflection of the number of a component to its members
        explored = dict(zip(E.nodes(), [False]*len(E)))
        c = 0 # number of a component

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
        # add ties of rank k
        increment = 0
        # components of rank k will be at positions from k-1 to the end of topCCnumbers
        while sortedCC[k + increment][0] == sortedCC[k-1][0]:
            topCCnumbers.append(sortedCC[k + increment])
            increment += 1

        # find highest scores among nodes in top-(k-1) components
        for length, numberCC in topCCnumbers[:k-1]:
            component = CC[numberCC]
            random.shuffle(component) # to randomize ties of max below
            max_node = max(component, key = lambda (node): scores[node]) # node with highest score
            scores[max_node] += length # even if all scores are zero we selected the random one among all by shuffling

        # deal with components of rank k separately
        # select only one component with highest score
        max_score_of_rank_k = -1
        for length, numberCC in topCCnumbers[k-1:]:
            component = CC[numberCC]
            random.shuffle(component) # to randomize ties of max below
            max_node = max(component, key = lambda (node): scores[node]) # node with highest score
            # it should pass following condition at least once
            if scores[max_node] > max_score_of_rank_k:
                max_score_of_rank_k = scores[max_node]
                max_node_of_rank_k = max_node
                max_node_length = length
        # update only one node with highest score among all CC of rank k
        scores[max_node_of_rank_k] += max_node_length

        print it+1, R, time.time() - time2iteration

    # select top scores
    # topScores = nlargest(k, scores.iteritems(), key = lambda (dk, dv): dv)
    # print topScores
    # S = [node for (node,_) in topScores]
    # select top scores wit penalization
    scores_copied = deepcopy(scores)
    S = []
    for it in range(k):
        maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv)
        # print maxv,
        S.append(maxk)
        scores_copied.pop(maxk) # remove top element from dict
        for v in G[maxk]:
            if v not in S:
                # weight = scores_copied[v]/maxv
                # print weight,
                penalty = (1-p)**(G[maxk][v]['weight'])
                scores_copied[v] = penalty*scores_copied[v]
    topScores = [(node, scores[node]) for node in S]
    print topScores

    return S




