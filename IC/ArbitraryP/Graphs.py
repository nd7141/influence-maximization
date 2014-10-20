from __future__ import division
import networkx as nx

def read_undirected_graph(filename):
    G = nx.Graph()
    with open(filename) as f:
        for line in f:
            e0, e1 = map(int, line.split())
            try:
                G[e0][e1]["weight"] += 1
            except KeyError:
                G.add_edge(e0, e1, {"weight": 1})
    return G

def read_directed_graph(filename):
    G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            e0, e1 = map(int, line.split())
            try:
                G[e0][e1]["weight"] += 1
            except KeyError:
                G.add_edge(e0, e1, {"weight": 1})
    return G

GNUTELLA_NETWORK_FILENAME = "../../graphdata/gnutella09.txt"
HEP_NETWORK_FILENAME = "../../graphdata/hep.txt"

Gnutella = read_directed_graph(GNUTELLA_NETWORK_FILENAME)
Hep = read_undirected_graph(HEP_NETWORK_FILENAME)
