from __future__ import division
import networkx as nx
import random

def Multivalency(G):
    Ep = dict()
    probabilities = [.01, .02, .04, .08]
    if isinstance(G, nx.Graph):
        for e in G.edges_iter():
            p = random.choice(probabilities)
            Ep[(e[0], e[1])] = p
            Ep[(e[1], e[0])] = p
    elif isinstance(G, nx.DiGraph):
        for e in G.out_edges():
            p = random.choice(probabilities)
            Ep[(e[0], e[1])] = p
    else:
        raise NotImplementedError
    return Ep


