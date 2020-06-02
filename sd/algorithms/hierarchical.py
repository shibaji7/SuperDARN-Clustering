#!/usr/bin/env python

"""
hierarchical.py: module is deddicated to run different hierarchical algorithms.

    Hierarchical clustering algorithms [23] were developed to overcome some of the disadvantages
    associated with flat or partitional-based clustering methods. Hierarchical algorithms were 
    developed to build a more deterministic and flexible mechanism for clustering the data objects. 
    Hierarchical methods can be categorized into agglomerative and divisive clustering methods.
    Agglomerativemethods start by taking singleton clusters (that contain only one data object per 
    cluster) at the bottom level and continue merging two clusters at a time to build a bottom-up
    hierarchy of the clusters. Divisive methods, on the other hand, start with all the data objects
    in a huge macro-cluster and split it continuously into two groups generating a top-down
    hierarchy of clusters.
        - agglomerative (here only this algoritm with different linkage function is in operation)
"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import sys
sys.path.append("extra/")
import numpy as np

from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration

class Hierarchi(object):
    """All hierarchical algorithms are implemened here."""

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

    def setup(self, **keywords):
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
