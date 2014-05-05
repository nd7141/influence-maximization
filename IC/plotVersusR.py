
from IC import avgSize
from CCHeuristic import CC_heuristic
import time
import networkx as nx
import matplotlib.pyplot as plt

def getDataTvsR(G, maxR, stepR, algo, k = 50, p=.01):
    data = dict()
    for R in range(1, maxR+2, stepR):
        S = algo(G, k, p, R)
        size = avgSize(G, S, p, R)
        data[R] = size
        print R
    return data

if __name__ == "__main__":
    time2implement = time.time()

    time2build = time.time()
    # read in graph
    G = nx.Graph()
    with open('../graphdata/hep.txt') as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
            # G.add_edge(u, v, weight=1)
    print 'Built graph G'
    print time.time() - time2build

    data = getDataTvsR(G, 10000, 1000, CC_heuristic, 50, .01)
    R = []
    T = []
    for k,v in sorted(data.items()):
        R.append(k)
        T.append(v)
    fig = plt.figure()
    plt.plot(R, T, 'r--')
    plt.xlabel("Number of Iterations, R")
    plt.ylabel("Spread of Influence")
    plt.title("CC algorithm: T vs R")
    plt.show()
    fig.savefig('plotforR.png', dpi=fig.dpi)

    console = []