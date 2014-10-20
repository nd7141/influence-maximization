from __future__ import division
import networkx as nx
import random
import multiprocessing
from Graphs import Gnutella, Hep
from Harvester import Harvester
from DD import GDD
from Models import Multivalency


def runIC(G, S, Ep):
    activated = dict(zip(G.nodes(), [False]*len(G)))
    activated.update(dict(zip(S, [True]*len(S))))

    new_activated_nodes = []
    for node in S:
        for out_node in G[node]:
            if not activated[out_node]:
                if random.random() <= 1 - (1 - Ep[(node, out_node)])**G[node][out_node]["weight"]:
                    activated[out_node] = True
                    new_activated_nodes.append(out_node)

    for activated_node in new_activated_nodes:
        for out_node in G[activated_node]:
            if not activated[out_node]:
                if random.random() <= 1 - (1 - Ep[(activated_node,out_node)])**G[activated_node][out_node]["weight"]:
                    activated[out_node] = True
                    new_activated_nodes.append(out_node)

    return len(S) + len(new_activated_nodes)

def getCoverage((G, S, Ep)):
    return runIC(G, S, Ep)

if __name__ == "__main__":

    # CONSTANTS
    I = 100
    MC = 100
    k = 100
    G = Hep
    Ep = Multivalency(G)
    pool = multiprocessing.Pool(processes=4)

    ### Harvester ###

    # Get seeds
    S = Harvester(G, k, Ep, MC)

    # Estimate coverage
    coverage_list = pool.map(getCoverage, ((G, S, Ep) for _ in range(I)))
    average_coverage = sum(coverage_list)/len(coverage_list)
    print "Harvester: k = %s coverage = %s" %(k, average_coverage)

    ### GDD ###

    # Get seeds
    S = GDD(G, k, Ep)

    # Estimate coverage
    coverage_list = pool.map(getCoverage, ((G, S, Ep) for _ in range(I)))
    average_coverage = sum(coverage_list)/len(coverage_list)
    print "GDD: k = %s coverage = %s" %(k, average_coverage)

    console = []