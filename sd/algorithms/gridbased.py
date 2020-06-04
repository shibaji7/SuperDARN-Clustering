#!/usr/bin/env python

"""
gridbased.py: module is deddicated to run different grid based algorithms.
     
     Grid-based clustering algorithms are efficient in mining large multidimensional data sets. These
     algorithms partition the data space into a finite number of cells to forma grid structure and then form
     clusters from the cells in the grid structure. Clusters correspond to regions that are more dense in
     data points than their surroundings.
        - BANG
        - CLIQUE
"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import numpy as np

from pyclustering.cluster.bang import bang
from pyclustering.cluster.clique import clique

class GBased(object):
    """All grid based algorithms are implemened here."""
    
    def __init__(self, method, data, random_state=0):
        """
        Initialize all the parameters.
        method: Name of the algorithms (lower case joined by underscore)
        data: Data (2D Matrix)
        random_state: Random initial state
        """
        self.L = data.shape[0]
        self.method = method
        self.data = data
        self.data_list = [data[x,:].tolist() for x in range(data.shape[0])]
        np.random.seed(random_state)
        
        self.levels = 11
        self.ccore = True
        self.density_threshold = 0.01
        self.amount_threshold = 3
        self.amount_intervals = 1
        return

    def setup(self, keywords={}):
        """
        Setup the algorithms
        """
        for p in keywords.keys():
            setattr(self, p, keywords[p])
            
        if self.method == "bang": self.obj = bang(self.data_list, self.levels, ccore=self.ccore,
                density_threshold=self.density_threshold, amount_threshold=self.amount_threshold)
        if self.method == "clique": self.obj = clique(self.data_list, self.amount_threshold, 
                self.density_threshold, ccore=self.ccore)
        return

    def extract_lables(self):
        """
        Extract lables form the cluster and noise information
        """
        self.clusters = self.obj.get_clusters()
        self.noise = self.obj.get_noise()
        C = range(len(self.clusters))
        labels_ = np.zeros(self.L)
        noise_ = np.zeros(self.L)
        for _c in C:
            labels_[self.clusters[_c]] = _c
        noise_[self.noise] = 1
        setattr(self.obj, "labels_", labels_)
        setattr(self.obj, "noise_", noise_)
        return
    
    def run(self):
        """
        Run the models
        """
        if self.method == "bang": 
            self.obj = self.obj.process()
            self.extract_lables()
        if self.method == "clique":
            self.obj = self.obj.process()
            self.extract_lables()
        return
