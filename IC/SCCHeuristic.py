from __future__ import division

import random, operator, time, os
from heapq import nlargest
from copy import deepcopy

def SCC_heuristic (G, k, p, R=20):
    scores = dict(zip(G.nodes(), [0]*len(G)))
    start = time.time()
    for it in range(R):
        E = deepcopy(G)
        edge_rem = [e for e in E.edges() if random.random() < (1-p)**(E[e[0]][e[1]]['weight'])]
        E.remove_edges_from(edge_rem)

        SCC = dict()
        explored = dict(zip(E.nodes(), [False]*len(E)))
        c = 0
        for node in E:
            if not explored[node]:
                c += 1
                explored[node] = True
                SCC[c] = [node]
                # perform BFS(E, node)
                component = E[node].keys()
                for neighbor in component:
                    if not explored[neighbor]:
                        explored[neighbor] = True
                        SCC[c].append(neighbor)
                        component.extend(E[neighbor].keys())

        # topSCC = nlargest(k, SCC.iteritems(), key= lambda (dk,dv): len(dv))
        # for (c, component) in topSCC:
        #     print c, len(component)
        #     weighted_score = 1.0/len(component)
        #     for node in component:
        #         scores[node] += weighted_score
        for c in SCC:
            # print c, len(SCC[c])
            weighted_score = len(SCC[c])
            for node in SCC[c]:
                scores[node] += weighted_score
        print it + 1, time.time() - start
    S = nlargest(k, scores.iteritems(), key=operator.itemgetter(1))
    return S, scores
