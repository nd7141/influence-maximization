__author__ = 'sergey'

from copy import deepcopy

import networkx as nx
import matplotlib.pylab as plt

from IC.IC import runIC


def neighborsOfS(G, S, radius):
    ''' Defines neighborhood for nodes in S
    Input: G -- networkx graph
    S -- set of nodes (list)
    Output: neighbors -- edges that define neighborhood of S (list)
    '''
    neighbor_edges = []
    edges = dict()
    # for radius = 0 only edges within nodes of S count
    for u in S:
        neighbor_edges.append((u,u,1))
        for v in S:
            try:
                w = G[u][v]['weight']
            except KeyError:
                pass
            else:
                if u not in edges or v not in edges[u]:
                    neighbor_edges.append((u,v,w))
                    edges.setdefault(u,[]).append(v)
                    edges.setdefault(v,[]).append(u)
    r = 0
    layer_prev = deepcopy(S)
    # find edges for radius > 0
    while r < radius:
        layer_new = []
        for u in layer_prev:
            for v in G[u]:
                # if we haven't explored node u or (u,v) hasn't been explored
                if u not in edges or v not in edges[u]:
                    edges.setdefault(u,[]).append(v)
                    edges.setdefault(v,[]).append(u)
                    neighbor_edges.append((u,v,G[u][v]['weight']))
                    layer_new.append(v) # add newly explored node
        layer_prev = deepcopy(layer_new)
        r += 1
    return neighbor_edges

def starS(G, S, radius):
    ''' Returns new graph object with nodes from S and its neighbors
    Input: G -- netowrkx graph
    S -- set of nodes (list)
    Output: E -- network graph Star
    '''
    E = nx.Graph()
    neighbors = neighborsOfS(G,S, radius)
    for u, v, d in neighbors:
        E.add_edge(u,v,weight=d)
    return E

def drawSpread(G,S,T):
    E = nx.Graph()
    # create neighborhood of S
    for u in S:
        for v in G[u]:
            try:
                E[u][v]['weight'] += 1
            except KeyError:
                E.add_edge(u,v,weight=1)
    # add activated nodes by neighbors of S
    activated = deepcopy(S)
    for v in T:
        if v not in activated:
            for u in activated:
                if u in G[v]:
                    E.add_edge(u,v,weight=1)
            activated.append(v)


    pos = nx.spring_layout(E)

    nx.draw_networkx_nodes(E, pos, node_color='k', node_size=10)
    nx.draw_networkx_nodes(E, pos, T, node_color='b', node_size=300)
    nx.draw_networkx_nodes(E, pos, S, node_color='w', node_size=300)

    nx.draw_networkx_labels(E, pos, labels=dict(zip(T,T)), font_color='r', font_size=10, font_weight='extra bold')

    nx.draw_networkx_edges(G, pos,edgelist=E.edges(), width=.5, edge_color='g', alpha=.5)

def drawStar(G,S,radius=1,layout='spring'):
    ''' Parameters for drawing a graph Star
     Input: G -- netowrkx graph
    S -- set of nodes (list)
    '''
    Star = starS(G,S,radius)
    if layout == 'spring':
        pos = nx.spring_layout(Star)
    elif layout == 'spectral':
        pos = nx.spectral_layout(Star)

    nx.draw_networkx_nodes(Star, pos, node_color='k', node_size=10)
    nx.draw_networkx_nodes(Star, pos, S, node_color='w', node_size=300)

    nx.draw_networkx_labels(Star, pos, labels=dict(zip(S,S)), font_color='r', font_size=10, font_weight='extra bold')

    nx.draw_networkx_edges(G, pos,edgelist=Star.edges(), width=.5, edge_color='g', alpha=.5)

    # edges = Star.edges(data=True)
    # edge_lbls = {(u,v): d['weight'] for (u,v,d) in edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lbls)

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('graphdata/../graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
    print 'Built graph G'
    print time.time() - start

    #read in initial set
    S = []
    with open('visualisation.txt') as f:
        for line in f:
            S.append(int(line))
    print 'Read initial seed sddet S'
    print time.time() - start

    #drawStar(G, S, radius=0, layout='spring')

    T = runIC(G,S)
    print 'Targeted %s nodes' %(len(T))
    print time.time() - start
    drawSpread(G,S,T)

    plt.axis('off')
    plt.show()

    console = []