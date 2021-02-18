# -*- coding: utf-8 -*-
#    Copyright (C) 2020 by
#    Max Falkenberg <max.falkenberg13@imperial.ac.uk>
#    All rights reserved.
#    BSD license.
"""
Correlated copying graph generator.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time
from scipy.special import comb

class rc_graph:
    """
    Creates a graph by copying using the uniform copying model or the correlated copying model.
    Attributes
    ----------
    t: int
        Timestep. Equal to number of nodes in the network.
    p: float or 'CCM' or 'shuffle' or 'CCM_prob'
        Copying probability.
    seed: int
        Random number seed
    T:  int
        Number of edges in the current observed network.
    T_track: list
        Number of edges in the observed network for full evolution of
        network. Only accessible is statistics==True.
    obs_k: list
        Observed degree of node i at index i.
    hidden_k: list
        Observed degree of node i at index i.
    obs_adjlist: list of lists
        Adjacency list for influence network.
    hidden_adjlist: list of lists
        Adjacency list for observed network.
    second_moment: int
        Current average second moment of degree. Only recorded if
        statistics == True. Not normalised by N.
    second_moment_track: list
        Average second moment of degree over time. Only recorded if
        statistics == True. Not normalised by N.
    Methods
    -------
    add_nodes(N)
        Add N nodes to the network.
    degree_dist(mode='obs',plot=True)
        Export degree distribution for observed network if mode == 'obs'
        or hidden network if mode == 'hidden'. Plot if plot == True.
    plot_edge_growth(scaling=None)
        Plots edge growth if statistics have been recorded.
    """

    def __init__(self,p=0,seed = None,statistics = False,cliques=False):
        """
        Class for undirected correlated copying model.
        Parameters
        ----------
        p    :  float or 'CCM' or 'prob' or 'shuffle', optional, default = 0.
                copying mode
                If p == 'CCM', 'hidden' edges copied in observed with p=1. Otherwise p=0.
                If p == 'shuffle', copied edges are randomly shuffled.
                If p == 'CCM_prob', each observed edge independently copied with
                probability set as ratio of hidden to observed degree.
                If numerical value input, set as fixed copy probability. Observed network equivalent to UCM.
        seed : integer, random_state, or None (default)
               Indicator of random number generation state.
               See :ref:`Randomness<randomness>`.
        statistics : boolean, default = True.
               Boolean to indicate whether to store statistics or not.
        Returns
        -------
        G : Graph
        Raises
        ------
        Error
            If `p` not in range ``0 <= p <= 1``.
        """
        if p == 'CCM' or p == 'shuffle' or p == 'CCM_prob':
            pass
        elif p < 0 or p > 1:
            raise Exception("Probability `p` not in range `0 <= p <= 1`.")


        self.t = 2 #time step. Equals total number of nodes.
        self.p = p #copying probability
        self.seed = seed #Random seed
        random.seed(seed)
        self.__statistics = statistics #Track statistics?
        self.__targets = [0,1] #Target list
        self.T = 2 #Number of targets (including repeats)
        self.obs_adjlist = [[1],[0]] #Adjacency list where nth list is a list of observed neighbors of node n
        self.hidden_adjlist = [[1],[0]] #Adjacency list for the hidden network
        self.obs_k =[1,1] #Degree of nodes in observed network
        self.hidden_k = [1,1] #Degree of nodes in hidden network
        self.T_track = [] #Track number of edges in observed network over time
        self.second_moment = 2 #Second moment currently
        self.second_moment_track = [] #Second moment over time.
        self.cliques_track = cliques
        self.eff_p = []
        if self.cliques_track:
            self.cliques = [[1]]

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
            self.eff_p.append(self.hidden_k[target]/self.obs_k[target])
            if self.p == 'CCM':
                copy_nodes = [target] + self.hidden_adjlist[target] #Neighbors of target which may be copied
                if self.cliques_track:
                    if len(self.cliques)<self.hidden_k[target]+1:
                        to_add = self.hidden_k[target] + 1 - len(self.cliques)
                        self.cliques.append([0]*(self.t-1))
                    N = np.ones(len(self.cliques),dtype='int')*self.hidden_k[target]
                    R = np.arange(0,len(N))
                    com = comb(N,R)
                    for i in range(len(self.cliques)-1):
                        self.cliques[i] += [com[i]+com[i+1]]
                    self.cliques[-1] += [com[-1]]

            elif self.p == 'shuffle':
                copy_nodes = [target] + random.sample(self.obs_adjlist[target],self.hidden_k[target])
            elif self.p == 'CCM_prob':
                p = self.hidden_k[target]/self.obs_k[target]
                copy_nodes = [target]
                for j in self.obs_adjlist[target]:
                    if random.random()<p:
                        copy_nodes += [j]
            else:
                copy_nodes = [target]
                for j in self.obs_adjlist[target]:
                    if random.random() < self.p:
                        copy_nodes += [j]
            self.obs_adjlist += [copy_nodes]
            self.obs_k += [len(copy_nodes)]
            for j in copy_nodes: #Adjust adjacency lists
                self.obs_adjlist[j] += [self.t]
                self.obs_k[j] += 1

            self.hidden_adjlist[target] += [self.t] #Updates neighbors in observed network
            self.hidden_adjlist += [[target]]
            self.hidden_k += [1]
            self.hidden_k[target] += 1

            self.t += 1
            if self.__statistics:
                self.T += 2*len(copy_nodes)
                self.T_track += [self.T] #Track number of edges
                self.second_moment += len(copy_nodes)**2 #Change in sum from new node
                for j in copy_nodes: #Change in second moment from existing nodes
                    self.second_moment += (2*self.obs_k[j])-1 #+k**2 - (k-1)**2
                self.second_moment_track += [self.second_moment]
        print(time.time()-start_time)

    def degree_dist(self,mode = 'hidden',plot=True,offset = 2):
        """
        Export degree distribution for the observed or hidden network.
        Parameters
        ----------
        mode: 'obs' or 'hidden':
            Export degree distribution for observed network if mode == 'obs'.
            Export degree distribution for hidden network if mode == 'hidden'.
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
        else:
            y,x = np.histogram(self.obs_k,bins=np.arange(int(np.min(self.obs_k)),int(np.max(self.obs_k)+2)))
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
        return x,y

    def plot_edge_growth(self):
        """
        Plot number of edges in the observed network over time.

        """
        if self.__statistics != True:
            raise Exception('Statistics for edge growth not recorded.')
        x = np.arange(2,len(self.T_track)+2,dtype='uint64') #time
        x_track = np.arange(3,len(self.T_track)+3,dtype='uint64') #time

        plt.plot(x,x-1,color='k',ls='--',label=r'$\propto t$') #linear scaling
        plt.plot(x,(x*(x-1))/2,color='k',ls='-.',label=r'$\propto t^{2}$') #complete graph

        T_track = np.array(self.T_track)/2
        plt.plot(x_track,T_track) #edge growth
        if isinstance(self.p,str):
            pass
        else:
            k_mom = self.p * np.array(self.second_moment_track)/(2*T_track)
            #Ratio of second to first moment scaled by p
            crossover = np.argmin(k_mom<1) #Index where k_mom exceeds 1
            if k_mom[crossover]>1: #Only plot if crossover reached
                plt.plot([x_track[crossover],x_track[crossover]],[T_track[0]-1,T_track[crossover]],ls=':',color='k')
                plt.plot([x_track[0]-1,x_track[crossover]],[T_track[crossover],T_track[crossover]],ls=':',color='k')
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
        else:
            raise Exception("Mode can only take values `obs` or `hidden`.")
        edgelist = []
        for i,j in enumerate(adj):
            for k in j:
                edgelist.append((i,k))
        return(edgelist)
