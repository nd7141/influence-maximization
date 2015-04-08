from __future__ import division
import networkx as nx
import json, time, random, os, math
from Harvester import make_possible_world
import numpy as np
from Models import Uniform

def find_diameter (G, CCs):
    ds = []
    for CC in CCs:
        d = nx.diameter(G.subgraph(CC))
        ds.append(d)
    return ds

def almost_equal(a,b,eps=1e-6):
    if abs(a-b) < eps:
        return True
    return False

def write_PWs(edge_file, MC, output_file):
    for mc in range(MC):
        with open(edge_file) as f:
            with open(output_file + str(mc) + ".txt", "w+") as g:
                for line in f:
                    u,v,p = map(float, line.split())
                    if random.random() <= p:
                        g.write("%d %d %s\n" %(u,v,p))

def cleanGraph(infile, outfile, model="no_model"):
    '''
    Transform edge list graph representation to the standard form,
    where first encountered vertex has smaller index than later encountered vertex.
    Example: 0 3 --> 0 1
            1 2 --> 2 3
    model: additionally add edge probability based on the model
     no_model -- just transforms the graph
     multivalency -- assigns probability from [.01, .02, .04, .08] at random
     random -- assigns p from [0, .1] at random
     uniform -- assigns all p equals to 0.1
     wc -- assigns probability 1/i, where i the number of incoming edges for endpoint of an edge
    '''
    G = dict()
    # rewrite the file
    with open(outfile, 'w+') as _:
        pass
    time2write = time.time()
    content = [] # content to write to the file
    wc_content = []
    with open(infile) as f:
        old2new = dict()
        mapped = 0
        track = 1
        for i, line in enumerate(f):
            if i == track*100000:
                print 'Processed %s edges' %i
                track += 1
            try:
                u, v, p = map(float, line.split())
            except:
                continue
            if u not in old2new:
                old2new[u] = mapped
                mapped += 1
            if v not in old2new:
                old2new[v] = mapped
                mapped += 1
            G.setdefault(old2new[u], []).append(old2new[v])
            G.setdefault(old2new[v], []).append(old2new[u])


            # collect content
            if model == "no_model":
                content.append("%s %s %s" %(old2new[u], old2new[v], p))
            elif model == "wc":
                wc_content.append((old2new[u], old2new[v]))
            else:
                if model == "multivalency":
                    p = random.choice([.01, .02, .04, .08])
                elif model == "random":
                    p = 0.1*random.random()
                elif model == "uniform":
                    p = 0.1
                else:
                    raise ValueError, 'Please provide appropriate model (wc, multivalency, random, uniform, no_model)'
                content.append("%s %s %s" %(old2new[u], old2new[v], p))


        if model == "wc":
            for (u,v) in wc_content:
                p = 1./len(G[v])
                content.append("%s %s %s" %(u, v, p))
        # write to file
        with open(outfile, "w+") as g:
            g.write("\n".join(content))
    print 'Total time:', time.time() - time2write

def get_data_for_spine(sn_in, act_in, sn_out, act_out):
    '''
    :param sn_in: social network structure
    :param act_in: action log
    :param sn_out: tabbed social network
    :param act_out: action log with removed duplicated actions
    :return:
    '''
    content = [] # content to write to the file
    with open(sn_in) as f:
        old2new = dict()
        mapped = 0
        track = 1
        for i, line in enumerate(f):
            if i == track*100000:
                print 'Processed %s edges' %i
                track += 1
            try:
                u, v = map(int, line.split())
            except:
                continue
            if u not in old2new:
                old2new[u] = mapped
                mapped += 1
            if v not in old2new:
                old2new[v] = mapped
                mapped += 1
            content.append("%s %s" %(old2new[u], old2new[v]))

    with open(act_in) as f, open("./tmp2.txt", "w+") as g:
        for line in f:
            d = map(int, line.split())
            if len(d) == 2:
                if d[0] in old2new:
                    g.write("\t%s\t%s\n" %(old2new[d[0]], d[1]))
            else:
                if d[0] in old2new and d[1] in old2new:
                    g.write("%s\t%s\t%s\n" %(old2new[d[0]], old2new[d[1]], d[2]))


    with open("./tmp.txt", "w+") as g:
        g.write("\n".join(content))
    print '-----------------------------'

    with open("./tmp2.txt") as f, open(act_out, "w+") as g:
        for line in f:
            d = line.split()
            if d[0] == d[1]:
                continue
            else:
                g.write(line)

    with open("./tmp.txt") as f, open(sn_out, "w+") as g:
        for line in f:
            d = line.split()
            g.write("%s\t%s\n" %(d[0], d[1]))

    os.remove("./tmp.txt")
    os.remove("./tmp2.txt")

def prune_data_for_spine(sn_in, sn_out, act_in, act_out, N):
    '''
    Keep only subgraph of N nodes of original network from sn_in.
    Keep actions only between remaining nodes
    :param sn_in:
    :param sn_out:
    :param act_in:
    :param act_out:
    :param N: number of nodes to keep
    :return:
    '''
    nodes = set()
    with open(sn_in) as f, open(sn_out, "w+") as g:
        for line in f:
            d = map(int, line.split())
            M = len(nodes)
            if M < N - 1:
                nodes.add(d[0])
                nodes.add(d[1])
            elif M == N - 1:
                if d[0] not in nodes:
                    nodes.add(d[0])
                else:
                    nodes.add(d[1])

            if d[0] in nodes and d[1] in nodes:
                g.write(line)

    with open(act_in) as f, open(act_out, "w+") as g:
        for line in f:
            d = map(int, line.split())
            if d[0] in nodes and d[1] in nodes:
                g.write(line)

# get_data_for_spine("Flixster/flixster.sn", "Flixster/flixster.out", "Flixster/flixster2.sn", "Flixster/flixster2.out")
prune_data_for_spine("Flixster/flixster3.sn", "Flixster/flixster3_reduced.sn", "Flixster/flixster3.out", "Flixster/flixster3_reduced.out",  5000)
#TODO find appropriate number of nodes after each spine doesn't break



# cleanGraph("Flixster/flixster.txt", "Flixster/flixster2.txt", "no_model")

# with open("Flixster/flixster2.txt") as f, open("Flixster/flixster3.txt", "w+") as g:
#     for line in f:
#         d = map(float, line.split())
#         if d[2] > 1e-10:
#             g.write(line)

# with open("Goodreads/goodreads_actions.out") as f, open("Goodreads/goodreads.out", "w+") as g:
#     line = next(f)
#     cur_node = int(line.split()[0])
#     g.write("\t%s\t0\n" %cur_node)
#     g.write(line)
#     for i, line in enumerate(f):
#         data = line.split()
#         if cur_node == int(data[0]):
#             g.write(line)
#         else:
#             cur_node = int(data[0])
#             g.write("\t%s\t0\n" %cur_node)
#             g.write(line)




# G = nx.Graph()
# from math import log
# edge_probabilities = []
# with open("hep15233_2.txt") as f:
#     for line in f:
#         u, v, p = map(float, line.split())
#         edge_probabilities.append(p)
#         if u != v:
#             G.add_edge(int(u), int(v), weight = -log(p))
# print "G: n %s m %s" %(len(G), len(G.edges()))
# print

#
# with open("hep15233_2_rel.txt") as f:
#     count = 0
#     for i, line in enumerate(f):
#         u,v,r = map(float, line.split())
#         u = int(u)
#         v = int(v)
#         if u not in G[v]:
#             print u,v,r
#             count += 1
#     print 'Count', count, "Total:", i


#
# with open("hep15233_2.txt", "w+") as f:
#     for (u,v) in G.edges():
#         p = random.triangular(0, .2, .14*3 - .2)
#         f.write("%s %s %s\n" %(u,v,p))

# pairs = [(11052, 11371), (10655, 9097)]
# node1, node2 = pairs[0]
#
# # triangle graph
# G = nx.Graph()
# G.add_weighted_edges_from([(0,1,-log(.5)), (1,2,-log(.1)), (2,0,-log(.8))])
# pairs = [(0,1)]
# node1, node2 = pairs[0]
#
# q_nodes = set([node1] + G[node1].keys() + G[node2].keys())
# q_edges = G.edges(q_nodes)
# Q = nx.Graph()
# Q.add_nodes_from(q_nodes)
#
# n1 = G[node1].keys()
# n1.remove(node2)
# n2 = G[node1].keys()
# n2.remove(node2)
# neighbors = n1 + n2
#
# neighbor_edges = []
# for (u,v) in q_edges:
#     if u in neighbors + [node1, node2] and v in neighbors + [node1, node2]:
#         neighbor_edges.append((u,v))
# Q.add_edges_from(q_edges[:2])
#
# extra_nodes = []
# for (u,v) in q_edges:
#     if u not in q_nodes:
#         extra_nodes.append(u)
#     if v not in q_nodes:
#         extra_nodes.append(v)
#
# import matplotlib.pyplot as plt
# import networkx as nx
#
# pos=nx.spring_layout(Q) # positions for all nodes
#
# # nodes
# # nx.draw_networkx_nodes(Q,pos,
# #                        nodelist=extra_nodes,
# #                        node_color='r',
# #                        node_size=50,
# #                    alpha=0.8)
# nx.draw_networkx_nodes(Q,pos,
#                        nodelist=neighbors,
#                        node_color='g',
#                        node_size=200,
#                    alpha=0.8)
# nx.draw_networkx_nodes(Q,pos,
#                        nodelist=[node1, node2],
#                        node_color='b',
#                        node_size=500,
#                    alpha=0.8)
# # edges
# nx.draw_networkx_edges(Q,pos,width=1.0,alpha=0.5)
# edge_labels = dict()
# for (u,v) in Q.edges():
#     edge_labels[(u,v)] = round(math.exp(1)**(-G[u][v]["weight"]), 2)
#
# nx.draw_networkx_edge_labels(Q, pos, edge_labels=edge_labels)
#
# plt.axis('off')
# plt.savefig("triangle.png", dpi=400) # save as png
# plt.show() # display

#
# for file_num in range(1,201):
#     adj_list = dict()
#     with open("./GAME_PWs/GAMEtr%s_Sergei.txt" %file_num) as f:
#         for line_number, line in enumerate(f):
#             s = " ".join(map(lambda v: str(pano2sergei[int(v)]), line.split())) + os.linesep
#             adj_list[pano2sergei[line_number]] = s
#     sorted_lines = sorted(adj_list.iteritems(), key = lambda (dk,dv): dk)
#     with open("./GAME_PWs/GAME%s_Sergei_new.txt" %file_num, "w+") as f:
#         for (_, line) in sorted_lines:
#             f.write(line)

# with open("./HarvesterPW_results/spreads.txt") as f:
#     start_spread = False
#     for line in f:
#         if not line.split():
#             start_spread = False
#         elif start_spread:
#             mc = int(line[:2])
#             spreads = json.loads(line[2:])
#             print '|PWs|: %s, mean: %.2f, std: %.2f, median: %.2f' %(mc, np.mean(spreads), np.std(spreads), np.median(spreads))
#         elif line.startswith("Harvester"):
#             print
#             print line.split()[-1]
#             start_spread = True

# print "Dataset:", dataset
# print "Number of nodes:", len(G)
# print "Number of edges:", len(G.edges())
# degree_seq =  sorted(nx.degree(G).values(), reverse = True)
# print "Average degree:", sum(degree_seq)/(len(G))
# print "Maximum degree:", degree_seq[0]
# CCs = nx.connected_components(G)
# print "# of CCs:", len(CCs)
# print "Maximum CC size: %s (%s %%)" %(len(CCs[0]), len(CCs[0])/len(G))
# print "Mean CC size:", len(CCs[len(CCs)//2])
# print "Average clustering coefficient:", nx.average_clustering(G)
# # print "Diameter:", max(find_diameter(G, CCs))

console = []