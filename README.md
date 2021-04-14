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
    G = rc.rc_graph(p=0,q=0,r=0)
    
Graph instance must be initialised with variables m, n and seed:

 **p**: Inner circle copying probability. Equivalent to **p_H** in paper.

 **q**: Outer circle copying probability. Equivalent to **p_O** in paper.

 **r**: Probability that copied edges are added to hidden network. Equivalent to **r** in paper.
 
 p, q and r must be floats between 0 t 1.

 **p = q** for UCM.

 **p = 1, q = 0, r = 0** for CCM.



Add N nodes to graph:

    G.add_nodes(N)
    #N can take any positive integer value.

Observed degree adjacency list stored in list of lists called as G.obs_adjlist

Observed degree adjacency list stored in list of lists called as G.hidden_adjlist

Export x and y values for degree distribution graph:

    G.degree_dist(mode = 'obs',plot)
    #mode: Export data for degree distribution of observed network degree if mode='obs', or hidden network if mode='hidden'.
    #plot: If plot=True, data is plotted.

To export edgelist for importation into networkx or elsewhere:

    edgelist = G.gen_edgelist(mode = 'obs')
    #mode: Generate edgelist for observed network if mode = 'obs' or hidden network if mode = 'hidden'
    

