Influence maximization in Social Graphs
=======================================

Repository for research project that studies [influence maximization problem.](http://www-bcf.usc.edu/~dkempe/publications/spread.pdf)

Degree Discount Algorithm
-------------------------

Degree discount heuristic was proposed first by [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf) Its results are slightly inferior than greedy hill-climbing algorithms, however it runs several orders of magnitudes faster. It's fine-tuned for Independent Cascade model. Running time is O(k*log(n)+m), where k - number of initial targets, n - number of vertices, m - number of edges. 

Representative Nodes Algorithm
------------------------------
Representative nodes heuristic is searching for the set of k nodes such that its size is maximum. Two different definitions of set size are used. First is cumulative sum of all distances between each pair of vertices inside the set. Second is minimum distance between a pair of vertices in the set. Algorithm didn't show significant results, probably because small differences in edge weights don't make some vertices be "more influential" than others. 
