# RandomCopying
Python implementation of random copying models
==============================================

Implementation of uniform random and correlated copying models. 

Max Falkenberg, mff113@ic.ac.uk

BSD 3 License. 

Please reference below publication if used for research purposes.

Reference: TBD

Environment Used
--------
python                    3.7.9
matplotlib                3.3.2
scipy                     1.5.2
numpy                     1.19.2

Instructions for use.
---------------------

Python3 implementation.

Import  and generate a graph instance and assign it to a free variable:

    import random_copy as rc
    G = rc.rc_graph(p=0,seed = None,statistics = False,cliques=False)
    
Graph instance must be initialised with variables m, n and seed:

 1. **p**: Copying mode. Either p is a float between 0 and 1 for uniform copying model, else p = 'CCM' for correlated copying model. 

Add N nodes to graph:

    G.add_nodes(N)
    #N can take any positive integer value.

Observed degree adjacency list stored in list of lists called as G.obs_adjlist

Observed degree adjacency list stored in list of lists called as G.hidden_adjlist

Export x and y values for degree distribution graph:

    G.degree_dist(mode = 'obs',plot)
    #mode: Export data for degree distribution of observed network degree if mode='obs', or hidden network if mode='inf'.
    #plot: If plot=True, data is plotted.


