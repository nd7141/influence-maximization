from __future__ import division
import networkx as nx
from Harvester import *
from Models import *
import os

def read_graph(filename, directed=False):
    '''
    Read a graph from a file that may have multiple edges between the same nodes.
    '''
    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            e0, e1 = map(int, line.split())
            try:
                G[e0][e1]["weight"] += 1
            except KeyError:
                G.add_edge(e0, e1, {"weight": 1})
    return G

def read_graph_without_weights(filename, directed=False):
    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            e0, e1 = map(int, line.split())
            G.add_edge(e0, e1)
    return G

def convert_undirected_to_directed(G):
    assert type(G) == type(nx.Graph())
    # check if there are weights on edges
    e1, e2 = G.edges()[0]
    if "weight" in G[e1][e2]:
        weighted = True
    else:
        weighted = False
    directed_G = nx.DiGraph()
    if weighted:
        for e in G.edges():
            directed_G.add_weighted_edges_from([(e[0], e[1], G[e[0]][e[1]]["weight"]), (e[1], e[0], G[e[1]][e[0]]["weight"])])
    else:
        for e in G.edges():
            directed_G.add_edges_from([(e[0], e[1]), (e[1], e[0])])
    return directed_G

def read_adjacency_list(filename, directed=False):
    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    with open(filename) as f:
        for node, line in enumerate(f):
            neighbors = map(int, line.split())
            G.add_node(node)
            G.add_edges_from([(node, v) for v in neighbors])
    return G

def transform_panos_representation():
    sergei2pano = dict()
    pano2sergei = dict()
    counter = 0
    with open("Ep_HEP_Multivalency_with_weights.txt") as f:
        for line in f:
            edge = map(float, line.split())
            u = int(edge[0])
            v = int(edge[1])
            p = edge[2]
            if u not in sergei2pano:
                sergei2pano[u] = counter
                pano2sergei[counter] = u
                counter += 1
            if v not in sergei2pano:
                sergei2pano[v] = counter
                pano2sergei[counter] = v
                counter += 1

    for file_num in range(1,201):
        adj_list = dict()
        with open("./GAME_PWs/GAMEtr%s_Sergei.txt" %file_num) as f:
            for line_number, line in enumerate(f):
                s = " ".join(map(lambda v: str(pano2sergei[int(v)]), line.split())) + os.linesep
                adj_list[pano2sergei[line_number]] = s
        sorted_lines = sorted(adj_list.iteritems(), key = lambda (dk,dv): dk)
        with open("./GAME_PWs/GAME%s_Sergei_new.txt" %file_num, "w+") as f:
            for (_, line) in sorted_lines:
                f.write(line)

def gen_prb (n, mu, sigma, lower=0, upper=1):
    '''Generate probability from normal distribution in the range [0,1].
    '''
    import scipy.stats as stats
    X = stats.truncnorm(
         (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(n)

def wrt_prb(i_flnm, o_flnm, mu=0.09, sigma=0.06, directed=True):
    G = read_graph(i_flnm)
    m = len(G.edges())
    X = gen_prb(m, mu, sigma)
    with open(o_flnm, "w+") as f:
        for i, e in enumerate(G.edges()):
            f.write("%d %d %s\n" %(e[0], e[1], X[i]))
            if directed:
                f.write("%d %d %s\n" %(e[1], e[0], X[i]))

GNUTELLA_NETWORK_FILENAME = "../../graphdata/gnutella09.txt"
HEP_NETWORK_FILENAME = "../../graphdata/hep.txt"

Gnutella = read_graph(GNUTELLA_NETWORK_FILENAME, True)
Hep = read_graph_without_weights(HEP_NETWORK_FILENAME)
directed_Hep = convert_undirected_to_directed(Hep)

console = []