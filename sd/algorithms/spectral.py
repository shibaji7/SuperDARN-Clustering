#!/usr/bin/env python

"""
spectral.py: module is deddicated to run different spectral algorithms.
     
     As opposed to “traditional clustering algorithms” such as k-means and generative mixture 
     models which always result in clusters with convex geometric shape, spectral clustering 
     can solve problems in much more complex scenarios, such as intertwined spirals, or other 
     arbitrary nonlinear shapes, because it does not make assumptions on the shapes of clusters.
        - Spectral Clustering
        - Spectral Bi Clustering
        - Spectral Co Clustering
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

from sklearn.cluster import SpectralClustering, SpectralBiclustering, SpectralCoclustering

class Spectral(object):
    """All spectral algorithms are implemened here."""
    
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
        
        self.cmethod = "bistochastic"
        self.n_components = 25
        self.n_best = 3
        self.svd_method = "randomized"
        self.n_svd_vecs = None
        self.mini_batch = False
        self.init = "k-means++"
        self.n_init = 10
        self.n_jobs = 5
        self.random_state = random_state
        self.gamma = 1.0
        self.affinity = "nearest_neighbors"
        self.eigen_solver = "arpack"
        self.n_neighbors = 10
        return
    
    
    def setup(self, keywords={}):
        """
        Setup the algorithms
        """
        for p in keywords.keys():
            setattr(self, p, keywords[p])
            
        if self.method == "spc": self.obj = SpectralClustering(n_clusters=self.n_clusters, n_components=self.n_components,
                random_state=self.random_state, n_init=self.n_init, gamma=self.gamma, affinity=self.affinity, 
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, eigen_solver=self.eigen_solver)
        if self.method == "spcb": self.obj = SpectralBiclustering(n_clusters=self.n_clusters, method=self.cmethod, 
                n_components=self.n_components, n_best=self.n_best, svd_method=self.svd_method, n_svd_vecs=self.n_svd_vecs,
                mini_batch=self.mini_batch, init=self.init, n_init=self.n_init, n_jobs=self.n_jobs, random_state=self.random_state)
        if self.method == "spcc": self.obj = SpectralCoclustering(n_clusters=self.n_clusters, svd_method=self.svd_method,
                n_svd_vecs=self.n_svd_vecs, mini_batch=self.mini_batch, init=self.init, n_init=self.n_init, n_jobs=self.n_jobs,
                random_state=self.random_state)
        return
    
    def run(self):
        """
        Run the models
        """
        if self.method == "spc": self.obj.fit(self.data)
        if self.method == "spcb": self.obj.fit(self.data)
        if self.method == "spcc": self.obj.fit(self.data)
        return
