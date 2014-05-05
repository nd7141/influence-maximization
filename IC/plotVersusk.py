
from IC import avgSize
import time
import matplotlib.pyplot as plt
from degreeDiscount import degreeDiscountIC
from CCHeuristic import CC_heuristic
from newGreedyIC import newGreedyIC
import networkx as nx
from plotVersusR import getDataTvsR

def getData (G, maxk, algo, p, axis):
    data = dict()
    for k in range(1,maxk+1):
        if axis == "size":
            S = algo(G, k, p)
            size = avgSize(G, S, p, 200)
            data[k] = size
        elif axis == "time":
            start = time.time()
            S = algo(G, k, p)
            finish = time.time()
            data[k] = finish - start
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

    print "Started calculating data for plot time vs k"
    data1 = getData(G, 50, degreeDiscountIC, .01, "time")
    data2 = getData(G, 50, CC_heuristic, .01, "time")
    data3 = getData(G, 50, newGreedyIC, .01, "time")
    fig = plt.figure()
    d1plt, = plt.semilogy(data1.keys(), data1.values(), 'r--')
    d2plt, = plt.semilogy(data2.keys(), data2.values(), 'b--')
    d3plt, = plt.semilogy(data3.keys(), data3.values(), 'g--')
    plt.xlabel("Seed set size k")
    plt.ylabel("Running Time (in sec)")
    plt.title("Running time for different k")
    plt.legend([d1plt,d2plt,d3plt], ["Degree Discount", "CC algorithm", "New Greedy"], loc=9)
    # plt.show()
    fig.savefig('time_vs_k.png', dpi=fig.dpi)

    print "Started calculating data for plot influence vs k"
    data1 = getData(G, 50, degreeDiscountIC, .01, "size")
    data2 = getData(G, 50, CC_heuristic, .01, "size")
    data3 = getData(G, 50, newGreedyIC, .01, "size")
    fig = plt.figure()
    d1plt, = plt.plot(data1.keys(), data1.values(), 'r--')
    d2plt, = plt.plot(data2.keys(), data2.values(), 'b--')
    d3plt, = plt.plot(data3.keys(), data3.values(), 'g--')
    plt.xlabel("Seed set size k")
    plt.ylabel("Influence spread (in sec)")
    plt.title("Running time for different k")
    plt.legend([d1plt,d2plt,d3plt], ["Degree Discount", "CC algorithm", "New Greedy"], loc=9)
    # plt.show()
    fig.savefig('influence_vs_k.png', dpi=fig.dpi)

    print "Started calculating data for plot influence vs R"
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
    # plt.show()
    fig.savefig('influence_vs_R.png', dpi=fig.dpi)

    print 'Total time:', time.time() - time2implement

    console = []