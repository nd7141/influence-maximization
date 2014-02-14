Influence maximization in Social Graphs
=======================================

Repository for research project that studies [influence maximization problem.](http://www-bcf.usc.edu/~dkempe/publications/spread.pdf)

Degree Discount Algorithm
-------------------------

Degree discount heuristic was proposed first by [Chen et al.](http://snap.stanford.edu/class/cs224w-readings/chen09influence.pdf) Its results are slightly inferior than greedy hill-climbing algorithms, however it runs several orders of magnitudes faster. It's fine-tuned for Independent Cascade model. Running time is O(k*log(n)+m), where k - number of initial targets, n - number of vertices, m - number if edges. 


TODO:
----
* Add Fibonacci heap to degreeDiscountIC.py to reduce running time
