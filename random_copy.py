# -*- coding: utf-8 -*-
#    Copyright (C) 2021 by
#    Max Falkenberg <max.falkenberg13@imperial.ac.uk>
#    All rights reserved.
#    BSD license.
"""
General correlated copying graph generator.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import networkx as nx

class rc_graph:

    def __init__(self,p=0,q=0,r=0,seed = None):
        """
        Class for undirected correlated copying model.
        Parameters
        ----------
        p    :  float, default = 0. Equivalent to p_H in paper.
        q    :  float, default = 0. Equivalent to p_O in paper.
        r    :  float, default = 0. Equivalent to q in paper.
        seed : integer, random_state, or None (default)
               Indicator of random number generation state.
               See :ref:`Randomness<randomness>`.
        Returns
        -------
        G : Graph
        Raises
        ------
        Error
            If `p` not in range ``0 <= p <= 1``.
        Error
            If `q` not in range ``0 <= q <= 1``.
        """
        try:
            if p >= 0 and p <= 1:
                pass
            else:
                raise Exception("Probability `p` not in range `0 <= p <= 1`.")
        except:
            raise Exception("Probability `p` must be float in range `0 <= p <= 1`.")
        self.p = p #copying probability
        try:
            if q >= 0 and q <= 1:
                pass
            else:
                raise Exception("Probability `q` not in range `0 <= q <= 1`.")
        except:
            raise Exception("Probability `q` must be float in range `0 <= q <= 1`.")
        self.q = q
        try:
            if r >= 0 and r <= 1:
                pass
            else:
                raise Exception("Probability `r` not in range `0 <= r <= 1`.")
        except:
            raise Exception("Probability `r` must be float in range `0 <= r <= 1`.")
        self.r = r

        self.t = 3 #time step. Equals total number of nodes.
        self.seed = seed #Random seed
        random.seed(seed)
        self.T_hidden = 2 #Number of targets (including repeats)
        self.T_outer = 1
        self.T_obs = 3
        self.hidden_adjlist = [[1],[0,2],[1]] #Adjacency list for the hidden network
        self.outer_adjlist = [[2],[],[0]]
        self.obs_adjlist = [[1,2],[0,2],[0,1]] #Adjacency list for nodes in observed NOT IN hidden
        self.hidden_k = [1,2,1] #Degree of nodes in hidden network
        self.outer_k = [1,0,1]
        self.obs_k = [2,2,2] #Degree of nodes in obs network
        self.T_track_hidden = [] #Track number of edges in observed network over time
        self.T_track_outer = []
        self.T_track_obs = []
        self.eff_p = []

    def add_nodes(self,N):
        """
        Add N nodes to the network.
        Parameters
        ----------
        N: int
            Number of nodes to add to the network.
        """
        start_time = time.time()
        for i in range(N):
            target = int(self.t * random.random()) #Initial target
            new_outer = []
            new_inner = []

            hcount = 1
            ocount = 0
            for j in self.hidden_adjlist[target]:
                if random.random() < self.p:
                    if random.random() < self.r:
                        new_inner.append(j)
                        hcount+=1
                    else:
                        new_outer.append(j)
                        ocount+=1
            for j in self.outer_adjlist[target]:
                if random.random() < self.q:
                    if random.random() < self.r:
                        new_inner.append(j)
                        hcount +=1
                    else:
                        new_outer.append(j)
                        ocount +=1

            self.hidden_adjlist += [[target] + new_inner]
            self.outer_adjlist += [new_outer]
            self.obs_adjlist += [[target] + new_inner + new_outer]
            self.hidden_k += [hcount]
            self.outer_k += [ocount]
            self.obs_k += [hcount + ocount]
            self.hidden_adjlist[target] += [self.t]
            self.hidden_k[target] += 1
            self.obs_adjlist[target] += [self.t]
            self.obs_k[target] += 1

            for j in new_inner:
                self.obs_adjlist[j] += [self.t]
                self.obs_k[j] += 1
                self.hidden_adjlist[j] += [self.t]
                self.hidden_k[j] += 1
            for j in new_outer:
                self.obs_adjlist[j] += [self.t]
                self.obs_k[j] += 1
                self.outer_adjlist[j] += [self.t]
                self.outer_k[j] += 1

            self.t += 1
            self.T_obs += 2 * self.obs_k[-1]
            self.T_track_obs += [self.T_obs]  # Track number of edges
            self.T_hidden += 2 * self.hidden_k[-1]
            self.T_track_hidden += [self.T_hidden]  # Track number of edges
            self.T_outer += 2 * self.outer_k[-1]
            self.T_track_outer += [self.T_outer]
            self.eff_p.append((self.obs_k[-1]-1) / (self.obs_k[target]-1))

            # if time.time() - start_time > 120:
            #     break
        print(time.time() - start_time)


    def degree_dist(self,mode = 'hidden',plot=True,offset = 2):
        """
        Export degree distribution for the observed or hidden network.
        Parameters
        ----------
        mode: 'obs', 'hidden' or 'outer':
            Export degree distribution for observed network if mode == 'obs'.
            Export degree distribution for hidden network if mode == 'hidden'.
            Export degree distribution for outer network if mode == 'outer'.
            Plot degree distribution if plot == True.
        plot: boolean
            Plot degree distribution if True.
        Returns
        -------
        x: ndarray
            Degree of nodes for degree distribution.
        y: ndarray
            Probability that nodes in the network have a specific degree
            corresponding to equivalent index in x.
        """
        if mode == 'hidden':
            y,x = np.histogram(self.hidden_k,bins=np.arange(int(np.min(self.hidden_k)),int(np.max(self.hidden_k)+2)))
        elif mode == 'obs':
            y,x = np.histogram(self.obs_k,bins=np.arange(int(np.min(self.obs_k)),int(np.max(self.obs_k)+2)))
        else:
            y, x = np.histogram(self.outer_k, bins=np.arange(int(np.min(self.outer_k)), int(np.max(self.outer_k) + 2)))
        x = x[:-1]
        x = x[y != 0]
        y = y[y != 0]
        y = y.astype('float')
        y /= np.sum(y)
        if plot:
            if mode == 'obs':
                def itt(k, pk_l1,offset=offset):
                    num1 = pk_l1 * np.sqrt(2 * (k - 1 - (2-offset))) + 2 ** (1 - k)
                    den = 1 + np.sqrt(2 * (k- (2-offset)))
                    return num1 / den

                hold = [1. / 6]
                for i in range(3, np.max(self.obs_k)+1):
                    hold.append(itt(i, hold[-1]))

            if plt.fignum_exists(0):
                plt.plot(x, y, ls='', marker='.')
            else:
                plt.figure(num=0)
                plt.plot(x,y,ls='',marker='.')
                plt.xscale('log')
                plt.yscale('log')
                if mode == 'obs':
                    plt.plot(range(2, np.max(self.obs_k)+1),hold,ls='--',color = 'k')

                plt.xlabel(r'$k$', fontsize=21)
                plt.ylabel(r'$P(k)$', fontsize=21)
                plt.title(mode, fontsize=15)
                plt.tick_params(labelsize='large', direction='out', right=False, top=False)
                plt.tight_layout()
            plt.show()
        return x,y

    def plot_edge_growth(self,mode = 'hidden'):
        """
        Plot number of edges in the observed network over time.
        Can set mode to ``hidden'', ``obs'' or ``outer''.

        """
        if mode == 'hidden':
            T_track = self.T_track_hidden
        elif mode == 'obs':
            T_track = self.T_track_obs
        else:
            T_track = self.T_track_outer
        x = np.arange(2,len(T_track)+2,dtype='uint64') #time
        x_track = np.arange(3,len(T_track)+3,dtype='uint64') #time

        plt.plot(x,x-1,color='k',ls='--',label=r'$\propto t$') #linear scaling
        plt.plot(x,(x*(x-1))/2,color='k',ls='-.',label=r'$\propto t^{2}$') #complete graph
        T_track = np.array(T_track)/2
        plt.plot(x_track,T_track) #edge growth
        plt.xlabel(r'$t$',fontsize = 21)
        plt.ylabel(r'$E(t)$',fontsize = 21)
        plt.xlim((2,None))
        plt.ylim((1,None))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def gen_edgelist(self,mode='obs'):
        if mode == 'obs':
            adj = self.obs_adjlist
        elif mode == 'hidden':
            adj = self.hidden_adjlist
        elif mode == 'outer':
            adj = self.outer_adjlist
        else:
            raise Exception("Mode can only take values `obs` or `hidden`.")
        edgelist = []
        for i,j in enumerate(adj):
            for k in j:
                edgelist.append((i,k))
        return(edgelist)

    def gen_networkx(self,mode='obs',plot = False):
        G = nx.Graph()
        G.add_edges_from(self.gen_edgelist(mode))
        if plot:
            c = np.array(list(nx.node_clique_number(G).values()))
            c = np.log(c)
            nx.draw(G, pos=nx.kamada_kawai_layout(G),node_size=40,node_color=c,vmin=np.min(c),vmax=np.max(c))
        return G
