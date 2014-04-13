__author__ = 'sergey'

import networkx as nx
import matplotlib.pylab as plt

def neighborsOfS(G, S):
    ''' Defines neighborhood for nodes in S
    Input: G -- networkx graph
    S -- set of nodes (list)
    Output: neighbors -- edges that define neighborhood of S (list)
    '''
    neighbors = []
    for v in S:
        for u in G[v]:
            if u not in S:
                neighbors.append((u,v,G[u][v]['weight']))
    return neighbors

def starS(G, S):
    ''' Returns new graph object with nodes from S and its neighbors
    Input: G -- netowrkx graph
    S -- set of nodes (list)
    Output: E -- network graph Star
    '''
    E = nx.Graph()
    neighbors = neighborsOfS(G,S)
    for u, v, d in neighbors:
        E.add_edge(u,v,weight=d)
    return E

def drawStar(G,S):
    ''' Parameters for drawing a graph Star
     Input: G -- netowrkx graph
    S -- set of nodes (list)
    '''
    Star = starS(G,S)
    pos = nx.spring_layout(Star)

    nx.draw_networkx_nodes(Star, pos, node_color='b', node_size=10)
    nx.draw_networkx_nodes(Star, pos, S, node_color='r', node_size=100)

    nx.draw_networkx_edges(G, pos,edgelist=Star.edges(), width=.5, edge_color='g', alpha=.5)

    # edges = Star.edges(data=True)
    # edge_lbls = {(u,v): d['weight'] for (u,v,d) in edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lbls)

if __name__ == '__main__':
    import time
    start = time.time()

    # read in graph
    G = nx.Graph()
    with open('graphdata/hep.txt') as f:
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
    with open('degreeDiscountIC.txt') as f:
        for line in f:
            S.append(int(line))
    print 'Read initial seed set S'
    print time.time() - start

    drawStar(G, S)

    plt.axis('off')
    plt.show()

    console = []