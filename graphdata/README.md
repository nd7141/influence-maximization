 
####This folder contains several different types of graphs.####

* hep.txt – A collaboration graph crawled from arXiv.org, High Energy Physics – Theory section, from year 1991 to year 2003.

* phy.txt – A collaboration graph crawled from arXiv.org, Physics section.

* graph30.txt - random graph with 30 vertices and 120 edges (created by networkx.py)

####This is a brief introduction to the data format of graph files.####

Line 1: Two integers – n and m. 
n is the number of nodes in the graph, all nodes are numbered from 0 to n-1. m is the number of edges, multiple edges between the same pair of nodes are counted separately.

Line 2 to m+1: There are m lines totally. Each line contains two integers, representing two vertices connected by an edge. As mentioned before, if there are two or more edges between the same pair of nodes, their ids will appear together in multiple lines.

Original source: [http://research.microsoft.com/](http://research.microsoft.com/en-us/people/weic/graphdata.zip)
