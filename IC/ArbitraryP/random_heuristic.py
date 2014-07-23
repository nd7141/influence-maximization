from __future__ import division
import networkx as nx
import random, multiprocessing, json
from runIAC import runIAC

def getSet (G, k):
    S = random.sample(G.nodes(), k)
    return S

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == "__main__":
    import time
    start = time.time()

    model = "Categories"

    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    dataset = "gnu09"

    G = nx.read_gpickle("../../graphs/U%s.gpickle" %dataset)
    print 'Read graph G'
    print time.time() - start

    Ep = dict()
    with open("Ep_%s_%s1.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    pool = None
    I = 250
    DROPBOX = "/home/sergey/Dropbox/Influence Maximization/"
    FILENAME = "DirectRandomfor_%s_%s.txt" %(dataset, model)
    ftime = open('plotdata/time' + FILENAME, 'a+')
    l2c = [[0, 0]]

    for length in range(1, 250, 5):

        S = getSet(G, length)

        if pool == None:
            pool = multiprocessing.Pool(processes=None)
        avg_size = 0
        time2avg = time.time()
        T = pool.map(getCoverage, ((G, S, Ep) for i in range(I)))
        avg_size = sum(T)/len(T)
        print "%s: Time to average: %s sec" %(length, time.time() - time2avg)

        l2c.append([length, avg_size])
        with open('plotdata/plot' + FILENAME, 'w+') as fp:
            json.dump(l2c, fp)
        with open(DROPBOX + 'plotdata/plot' + FILENAME, 'w+') as fp:
            json.dump(l2c, fp)
