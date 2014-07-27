from __future__ import division
from runIAC import runIAC
import networkx as nx
import json, time, multiprocessing

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == "__main__":
    start = time.time()

    input_filename = "Seeds_GDD_hep_Categories.txt"

    file_split = input_filename.split("_")
    algo = file_split[1]
    dataset = file_split[2]
    idx = file_split[3].find(".txt")
    model = file_split[3][:idx]
    print model, dataset, algo

    output_filename = "Spread_%s_%s_%s.txt" %(algo, dataset, model)

    FOLDER = "Data4InfMax"
    SEEDS_FOLDER = "Seeds"
    SPREAD_FOLDER = "Spread"
    # TIME_FOLDER = "Time"
    DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
    seeds_filename = FOLDER + "/" + SEEDS_FOLDER + "/" + input_filename
    spread_filename = FOLDER + "/" + SPREAD_FOLDER + "/" + output_filename

    #define G
    G = nx.read_gpickle("../../graphs/U%s.gpickle" %dataset)
    print 'Read graph G'

    # define Ep
    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    Ep = dict()
    with open("Ep_%s_%s1.txt" %(dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    #define I
    I = 10000

    pool = multiprocessing.Pool(processes=None)
    with open(seeds_filename) as fp:
        for line in fp:
            S = json.loads(line)
            print len(S),
            Ts = pool.map(getCoverage, ((G, S, Ep) for _ in range(I)))
            T = sum(Ts)/len(Ts)
            print T
            k = len(S)
            with open(spread_filename, "a+") as fp:
                print >>fp, k, " ", T


    console = []