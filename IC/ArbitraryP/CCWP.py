
'''
CCWP heuristic for arbitrary propagation probabilities.
'''

from __future__ import division
import networkx as nx
from heapq import nlargest
from copy import deepcopy, copy
import os, json, multiprocessing, random
from runIAC import *

def CCWP((G, k, Ep)):
    '''
     Input:
     G -- undirected graph (nx.Graph)
     k -- number of nodes in seed set (int)
     p -- propagation probability among all edges (int)
     Output:
     scores -- scores of nodes according to some weight function (dict)
    '''
    scores = dict(zip(G.nodes(), [0]*len(G))) # initialize scores

    CC = findCC(G, Ep)

    # find ties for components of rank k and add them all as qualified
    sortedCC = sorted([(len(cc), cc_number) for (cc_number, cc) in CC.iteritems()], reverse=True)
    topCCnumbers = sortedCC[:k] # CCs we assign scores to
    QN = sum([l for (l, _) in topCCnumbers]) # number of qualified nodes

    increment = 0
    try:
        while k+increment < len(sortedCC) and sortedCC[k + increment][0] == sortedCC[k-1][0]:
            topCCnumbers.append(sortedCC[k + increment])
            increment += 1
            QN += sortedCC[k + increment][0]
    except IndexError:
        pass
    # assign scores to nodes in top Connected Components
    prev_length  = topCCnumbers[0][0]
    rank = 1
    QCC = len(topCCnumbers)
    for length, numberCC in topCCnumbers:
        if length != prev_length:
            prev_length = length
            rank += 1
        weighted_score = 1.0/length # updatef = 1
        for node in CC[numberCC]:
            scores[node] += weighted_score
    return scores

def timed(func):
    def measure_time(*args, **kw):
        start = time.time()
        res = func(*args, **kw)
        finish = time.time()

        print 'Time: %s %.4f sec' %(func.__name__, finish-start)
        return res
    return measure_time

def sort_dict(dictionary, reverse=True):
    return sorted(dictionary.iteritems(), key=lambda (key,value): value, reverse=reverse)

# @timed
def create_live_edge_graph(G, Ep):
    '''
    Create live-edge graph from G
    '''
    if isinstance(G, nx.DiGraph):
        E = nx.DiGraph()
    elif isinstance(G, nx.Graph):
        E = nx.Graph()
    else:
        raise ValueError, 'First argument should be nx.DiGraph or nx.Graph. Got %s instead' %(type(G))
    E.add_nodes_from(G.nodes()) # add all nodes in case of isolated components
    live_edges = [edge for edge in G.edges() if random.random() >= (1-Ep[edge])**(G[edge[0]][edge[1]]['weight'])]
    E.add_edges_from(live_edges)

    return E

# @timed
def find_strongly_connected_components(E):
    n2c = dict() # nodes to components
    c2n = dict() # component to nodes

    scc = nx.strongly_connected_components(E)
    for number, component in enumerate(scc):
        c2n[number] = component
        n2c.update(dict(zip(component, [number]*len(component))))

    return n2c, c2n

# @timed
def create_dags(E, n2c):
    dags = nx.DiGraph()
    for node in E:
        dags.add_node(n2c[node])
        for out_node in E[node]:
            if n2c[node] != n2c[out_node]:
                dags.add_edge(n2c[node], n2c[out_node])
    return dags

# @timed
def find_reach_topsort(dags, c2n):
    node_reach = dict()
    cluster_reach = dict()

    wccs = nx.weakly_connected_component_subgraphs(dags)

    for hub in wccs:
        # treat hubs of size 1 and 2 specially
        if len(hub) == 1:
            cluster = hub.nodes()[0]
            cluster_reach[cluster] = c2n[cluster]

            node_reach.update(dict(zip(c2n[cluster], [len(c2n[cluster])]*len(c2n[cluster]))))
        elif len(hub) == 2:
            cluster1, cluster2 = hub.edges()[0]

            cluster_reach[cluster2] = c2n[cluster2]
            cluster_reach[cluster1] = c2n[cluster1] + c2n[cluster2]

            node_reach.update(dict(zip(c2n[cluster1], [len(cluster_reach[cluster1])]*len(c2n[cluster1]))))
            node_reach.update(dict(zip(c2n[cluster2], [len(cluster_reach[cluster2])]*len(c2n[cluster2]))))
        else:
            hub_ts = nx.topological_sort(hub, reverse=True)
            for cluster in hub_ts:
                reach = set()
                for _, out_cluster in dags.out_edges(cluster):
                    reach.update(cluster_reach[out_cluster])
                reach.update(c2n[cluster])
                cluster_reach[cluster] = reach

                node_reach.update(dict(zip(c2n[cluster], [len(reach)]*len(c2n[cluster]))))
    return node_reach

# @timed
def find_reach_bfs (E):
    node_reach = dict()

    for node in E:
        reachable_nodes = [node]
        # Do BFS
        out_edges = E.out_edges(node)
        i = 0
        while i < len(out_edges):
            e = out_edges[i]
            if e[1] not in reachable_nodes:
                reachable_nodes.append(e[1])
                out_edges.extend(E.out_edges(e[1]))
            i += 1
        node_reach[node] = len(reachable_nodes)
    return node_reach

def find_reach_undirected(E):
    node_reach = dict()
    connected_components = nx.connected_components(E)
    for cc in connected_components:
        cc_size = len(cc)
        node_reach.update(dict(zip(cc, [cc_size]*cc_size)))

    return node_reach

# @timed
def assign_scores(node_reach, k):
    sorted_reach = sorted(node_reach.iteritems(), key= lambda (dk,dv): dv, reverse=True)
    # find the last index in sorted_reach to which assign a score
    min_value = sorted_reach[k-1][1]
    new_idx = k
    new_value = sorted_reach[k][1]
    while new_value == min_value:
        new_idx += 1
        try:
            new_value = sorted_reach[new_idx][1]
        except IndexError:
            break

    scores = dict(sorted_reach[:new_idx]) # score = |CC|
    # selected_nodes = [node for (node, reach) in sorted_reach[:new_idx]]
    # scores = dict(zip(selected_nodes, [1]*len(selected_nodes))) # score = 1
    return scores

def Harvester_topsort((G, k, Ep)):
    '''
    Implements Harvester for directed graphs using topological sort to find reach for each node.
    Model: IC
    '''
    assert isinstance(G, nx.DiGraph), '''First argument should be a directed graph, got %s instead.
    Use Harvester_undirected in case of undirected graph''' %type(G)

    E = create_live_edge_graph(G, Ep)
    n2c, c2n = find_strongly_connected_components(E)
    dags = create_dags(E, n2c)
    node_reach = find_reach_topsort(dags, c2n)
    scores = assign_scores(node_reach, k)
    # print sort_dict(scores)

    return scores

def Harvester_bfs ((G, k, Ep)):
    '''Implements Harvester for directed graphs using bfs to find reach for each node.
    Model: IC
    '''
    assert isinstance(G, nx.DiGraph), '''First argument should be a directed graph, got %s instead.
    Use Harvester_undirected in case of undirected graph''' %type(G)

    E = create_live_edge_graph(G, Ep)
    node_reach = find_reach_bfs(E)
    scores = assign_scores(node_reach, k)
    # print scores

    return scores

def Harvester_undirected((G, k, Ep)):
    '''Implements Harvester for undirected graphs using connected components to find reach for each node.
    Model: IC
    '''
    assert isinstance(G, nx.Graph), '''First argument should be a undirected graph, got %s instead.
    Use Harvester_bfs or Harvester_topsort in case of directed graph''' %type(G)

    E = create_live_edge_graph(G, Ep)
    node_reach = find_reach_undirected(E)
    scores = assign_scores(node_reach, k)

    # print scores
    return scores

def frange(begin, end, step):
    x = begin
    y = end
    while x < y:
        yield x
        x += step

def getCoverage((G, S, Ep)):
    return len(runIAC(G, S, Ep))

if __name__ == '__main__':
    import time
    start = time.time()

    dataset = "gnu09"
    directed = "U" # "D" for directed case; "U" for undirected case
    model = "Categories"
    print model, dataset

    if model == "MultiValency":
        ep_model = "range1"
    elif model == "Random":
        ep_model = "random1"
    elif model == "Categories":
        ep_model = "degree1"
    elif model == "%s_Weighted":
        ep_model = "weighted1"
    elif model == "Uniform":
        ep_model = "uniform1"

    G = nx.read_gpickle("../../graphs/%s%s.gpickle" %(directed, dataset))
    print 'Read graph G'
    print time.time() - start
    print len(G), len(G.edges())

    Ep = dict()
    with open("Ep/%s_Ep_%s_%s.txt" %(directed, dataset, ep_model)) as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])

    #calculate initial set
    R = 500
    I = 500
    ALGO_NAME = "CCWP"
    FOLDER = "Data4InfMax"
    SEEDS_FOLDER = "Seeds"
    TIME_FOLDER = "Time"
    DROPBOX_FOLDER = "/home/sergey/Dropbox/Influence Maximization"
    # seeds_filename = FOLDER + "/" + SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
    seeds_filename = FOLDER + "/" + SEEDS_FOLDER + "/%s_%s_%s_%s.txt" %(SEEDS_FOLDER, ALGO_NAME, dataset, model)
    time_filename = FOLDER + "/" + TIME_FOLDER + "/%s_%s_%s_%s.txt" %(TIME_FOLDER, ALGO_NAME, dataset, model)
    logfile = open('log.txt', 'w+')
    # print >>logfile, '--------------------------------'
    # print >>logfile, time.strftime("%c")

    l2c = [[0,0]]
    pool = None
    pool2 = None
    # open file for writing output
    seeds_file = open("%s" %seeds_filename, "a+")
    time_file = open("%s" %time_filename, "a+")
    dbox_seeds_file = open("%s/%s" %(DROPBOX_FOLDER, seeds_filename), "a+")
    dbox_time_file = open("%s/%s" %(DROPBOX_FOLDER, time_filename), "a+")

    for length in range(10, 200, 20):
        time2length = time.time()
        print 'Start finding solution for length = %s' %length
        print >>logfile, 'Start finding solution for length = %s' %length
        time2S = time.time()

        print 'Start mapping...'
        time2map = time.time()
        if pool == None:
            pool = multiprocessing.Pool(processes=4)
        Scores = map(Harvester_undirected, ((G, length, Ep) for i in range(R)))
        print 'Finished mapping in', time.time() - time2map

        print 'Start reducing...'
        time2reduce = time.time()

        scores = dict()
        for Score in Scores:
            for node in Score:
                try:
                    scores[node] += Score[node]
                except KeyError:
                    scores[node] = Score[node]
        scores_copied = deepcopy(scores)
        S = []
        # penalization phase
        for it in range(length):
            maxk, maxv = max(scores_copied.iteritems(), key = lambda (dk, dv): dv)
            S.append(maxk)
            scores_copied.pop(maxk) # remove top element from dict
            for v in G[maxk]:
                if v not in S and v in scores_copied:
                    # weight = scores_copied[v]/maxv
                    # print weight,
                    penalty = (1-Ep[(maxk, v)])**(G[maxk][v]['weight'])
                    scores_copied[v] *= penalty
        print 'Finished reducing in', time.time() - time2reduce

        print 'Total:', time.time() - start

        print S
        results = pool.map(getCoverage, ((G, S, Ep) for _ in range(I)))
        coverage = sum(results)/len(results)
        print "Coverage:", coverage
        l2c.append((length, coverage))
        with open("harvester_undirected.txt", "w+") as f:
            json.dump(l2c, f)
        # print avgIAC(G, S, Ep, 500)
        # # print >>logfile, json.dumps(S)
        # time2complete = time.time() - time2S
        # # with open("%s" %time_filename, "a+") as time_file:
        # #     print >>time_file, (time2complete)
        # # with open("%s/%s" %(DROPBOX_FOLDER, time_filename), "a+") as dbox_time_file:
        # #     print >>dbox_time_file, (time2complete)
        # print 'Finish finding S in %s sec...' %(time2complete)

        # print 'Writing S to files...'
        # with open("%s" %seeds_filename, "a+") as seeds_file:
        #     print >>seeds_file, json.dumps(S)
        # with open("%s/%s" %(DROPBOX_FOLDER, seeds_filename), "a+") as dbox_seeds_file:
        #     print >>dbox_seeds_file, json.dumps(S)

        # print 'Start calculating coverage...'
        # coverage = sum(pool.map(getCoverage, ((G, S, Ep) for _ in range(I))))/I
        # print 'S:', S
        # print 'Coverage', coverage

        # print 'Total time for length = %s: %s sec' %(length, time.time() - time2length)
        print '----------------------------------------------'

    # seeds_file.close()
    # dbox_seeds_file.close()
    # time_file.close()
    # dbox_time_file.close()
    # logfile.close()

    # Q = nx.DiGraph()
    # Q.add_path([1,2,3,4])
    # Q.add_edge(3,1)
    # Q.add_nodes_from([5,6])
    # Q.add_edge(7,8)
    #
    # start = time.time()
    # reachability_test = CCWP_test((G, 2, Ep))
    # print reachability_test
    # print 'test:', time.time() - start

    # start = time.time()
    # reachability_bench = CCWP_directed((G, 2, Ep))
    # reachability_test = CCWP_test((G, 10, Ep))
    # print reachability_test
    # print 'benchmark:', time.time() - start

    Q = nx.DiGraph()
    Q.add_edges_from([(1,2),(2,3),(3,4),(2,5),(5,6), (4,7), (6,7)])
    Q.add_node(0)
    Q.add_edges_from([(8,9),(8,10),(9,10)])
    Q.add_edges_from([(11,12), (11,13), (12,14), (13,14)])

    # print find_reach_bfs(Q)
    # n2c, c2n = find_connected_components(Q)
    # dags = create_dags(Q, n2c)
    # print find_reach_topsort(dags, c2n)

    # P = nx.DiGraph()
    # P.add_edges_from([(4,1), (4,3), (3,2), (2,1), (2,0)])
    # P.add_edges_from([(7,6), (7,5)])
    # print CCWP_test((G,3,Ep))
    # print 'Total time: %s' %(time.time() - start)

    console = []