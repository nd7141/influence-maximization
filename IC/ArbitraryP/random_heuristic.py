from __future__ import division
import networkx as nx
import random, multiprocessing, json
from runIAC import runIAC

def getSet (G, k):
    S = random.sample(G.nodes(), k)
    return S

def getDegreeS (G, k):
    from heapq import nlargest
    d = dict()
    for u in G:
        d[u] = sum([G[u][v]["weight"] for v in G[u]])
    [nodes, _] = zip(*nlargest(k, d.iteritems(), key = lambda (dk,dv): dv))
    S = list(nodes)
    return S

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == "__main__":
    import time
    start = time.time()

    dataset = "gnu09"
    model = "Categories"
    print dataset, model

    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    G = nx.read_gpickle("../../graphs/U%s.gpickle" %dataset)
    print 'Read graph G'
    print time.time() - start

    Ep = dict()
    with open("Ep_%s_%s1.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    #calculate initial set
    I = 1000
    ALGO_NAME = "Degree"
    FOLDER = "Data4InfMax"
    SEEDS_FOLDER = "Seeds"
    TIME_FOLDER = "Time"
    DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
    seeds_filename = FOLDER + "/" + SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
    time_filename = FOLDER + "/" + TIME_FOLDER + "/%s_%s_%s_%s.txt" %(TIME_FOLDER, ALGO_NAME, dataset, model)


    length_to_coverage = {0:0}
    l2c = [[0,0]]
    pool = None
    for length in range(1, 156, 5):

        time2length = time.time()

        print 'Start finding solution for length = %s' %length
        time2S = time.time()
        S = getDegreeS(G, length)
        time2complete = time.time() - time2S
        with open("%s" %time_filename, "a+") as time_file:
            print >>time_file, (time2complete)
        with open("%s/%s" %(DROPBOX_FOLDER, time_filename), "a+") as dbox_time_file:
            print >>dbox_time_file, (time2complete)
        print 'Finish finding S in %s sec...' %(time2complete)

        print 'Writing S to files...'
        with open("%s" %seeds_filename, "a+") as seeds_file:
            print >>seeds_file, json.dumps(S)
        with open("%s/%s" %(DROPBOX_FOLDER, seeds_filename), "a+") as dbox_seeds_file:
            print >>dbox_seeds_file, json.dumps(S)

        print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        print '----------------------------------------------'

    print 'Total time: %s' %(time.time() - start)

    console = []