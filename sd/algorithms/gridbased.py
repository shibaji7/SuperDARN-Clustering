#!/usr/bin/env python

"""
gridbased.py: module is deddicated to run different grid based algorithms.
     
     Grid-based clustering algorithms are efficient in mining large multidimensional data sets. These
     algorithms partition the data space into a finite number of cells to forma grid structure and then form
     clusters from the cells in the grid structure. Clusters correspond to regions that are more dense in
     data points than their surroundings.
        - BANG
        - BIRCH
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


class GBased(object):
    """All grid based algorithms are implemened here."""
    
    def __init__(self, method, data, n_clusters=2, random_state=0):
        """
        Initialize all the parameters.
        method: Name of the algorithms (lower case joined by underscore)
        data: Data (2D Matrix)
        n_clusters: Number of clusters
        random_state: Random initial state
        """
        self.method = method
        self.data = data
        self.n_clusters = n_clusters
        np.random.seed(random_state)
        
        self.affinity = "euclidean"
        self.linkage = "ward"
        self.distance_threshold = None
        return


     def setup(self, keywords={}):
        """
        Setup the algorithms
        """
        for p in keywords.keys():
            setattr(self, p, keywords[p])
            
        if self.method == "agglomerative": self.obj = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage,
                affinity=self.affinity)
        if self.method == "feature": self.obj = FeatureAgglomeration(n_clusters=self.n_clusters, linkage=self.linkage,
                affinity=self.affinity, distance_threshold=self.distance_threshold)
        return
    
    def run(self):
        """
        Run the models
        """
        if self.method == "agglomerative": self.obj.fit(self.data)
        if self.method == "feature": self.obj.fit(self.data)
        return
