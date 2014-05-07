

import networkx as nx
import random
import os

def generateGraph (n,m,filename='',pw=.75,maxw=5):
    G = nx.dense_gnm_random_graph(n,m)
    for e in G.edges():
        if random.random() < pw:
            G[e[0]][e[1]]['weight'] = 1
        else:
            G[e[0]][e[1]]['weight'] = random.randint(2,maxw)
    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' %(len(G.nodes()), len(G.edges()), os.linesep))
            for v1,v2,edata in G.edges(data=True):
                for it in range(edata['weight']):
                    f.write('%s %s%s' %(v1, v2, os.linesep))

if __name__ == '__main__':
    generateGraph(30, 120, 'small_graph.txt')