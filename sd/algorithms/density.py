#!/usr/bin/env python

"""
density.py: module is deddicated to run different density based algorithms.

    Density-based clusters are connected, dense areas in the data space separated from each other
    by sparser areas. Furthermore, the density within the areas of noise is assumed to be lower than the
    density in any of the clusters. Due to their local nature, dense connected areas in the data space can
    have arbitrary shape. Given an index structure that supports region queries, density-based clusters
    can be efficiently computed by performing at most one region query per database object. Sparse
    areas in the data space are treated as noise and are not assigned to any cluster.
        - dbscan
        - optics
        - hdbscan
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

from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN

class DBased(object):
    """All density based algorithms are implemened here."""

    def __init__(self, method, data, random_state=0):
        """
        Initialize all the parameters.
        method: Name of the algorithms (lower case joined by underscore)
        data: Data (2D Matrix)
        n_clusters: Number of clusters
        random_state: Random initial state
        """
        self.method = method
        self.data = data
        np.random.seed(random_state)

        self.eps = 0.5
        self.min_samples = 5
        self.metric = "euclidean"
        self.algorithm = "auto"
        self.n_jobs = 5
        self.leaf_size = 30
        self.max_eps = np.inf
        self.p = 2
        self.cluster_method = "xi"
        self.xi = 0.05
        self.min_cluster_size = 10
        self.cluster_selection_method = "eom"
        self.cluster_selection_epsilon = 0.0
        self.alpha = 1.0
        return

    def setup(self, keywords={}):
        """
        Setup the algorithms
        """
        for p in keywords.keys():
            setattr(self, p, keywords[p])

        if self.method == "dbscan": self.obj = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric,
                algorithm=self.algorithm, n_jobs=self.n_jobs, leaf_size=self.leaf_size, p=self.p)
        if self.method == "optics": self.obj = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps, metric=self.metric,
                p=self.p, cluster_method=self.cluster_method, eps=self.eps, xi=self.xi, algorithm=self.algorithm,
                leaf_size=self.leaf_size, n_jobs=self.n_jobs)
        if self.method == "hdbscan": self.obj = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
                alpha=self.alpha, cluster_selection_epsilon=self.cluster_selection_epsilon, metric=self.metric, p=self.p, 
                leaf_size=self.leaf_size, algorithm=self.algorithm, core_dist_n_jobs=self.n_jobs, 
                cluster_selection_method=self.cluster_selection_method)
        return

    def run(self):
        """
        Run the models
        """
        if self.method == "dbscan": self.obj.fit(self.data)
        if self.method == "optics": self.obj.fit(self.data)
        if self.method == "hdbscan": self.obj.fit(self.data)
        return
