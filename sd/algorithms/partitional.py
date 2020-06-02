#!/usr/bin/env python

"""
partitional.py: module is deddicated to run different partitional algorithms.

    Partitional clustering algorithms aim to discover the groupings present in the data by optimizing
    a specific objective function and iteratively improving the quality of the partitions. These algorithms 
    generally require certain user parameters to choose the prototype points that represent each cluster.
    For this reason they are also called prototype-based clustering algorithms.
        - kMeans
        - kMediods
        - kMedians
        - kModes
        - fuzzykMeans
        - Mean Shift
        - Kernel kMeans
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

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from kmodes.kmodes import KModes
from kmedians import KMedians
from kmedoids import KMedoids
from fuzzykmeans import FuzzyKMeans
from kernelkmeans import KernelKMeans

class Partition(object):
    """All partitoned algorithms are implemened here."""

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
        return

    def setup(self, **keywords):
        """
        Setup the algorithms
        """
        for key in keywords.keys():
            setattr(self, key, keywords[key])
        if self.method == "kmeans": self.obj = KMeans(n_clusters=self.n_clusters, **keywords)
        if self.method == "kmedoids": self.obj = KMedoids(n_clusters=self.n_clusters, **keywords)
        if self.method == "kmodes": self.obj = KModes(n_clusters=self.n_clusters, init="Huang", **keywords)
        if self.method == "kmedians": self.obj = KMedians(n_clusters=self.n_clusters, **keywords)
        if self.method == "fuzzykmeans": self.obj = FuzzyKMeans(n_clusters=self.n_clusters, **keywords)
        if self.method == "meanshift": self.obj = MeanShift(n_jobs=10, **keywords)
        if self.method == "kernelkmeans": self.obj = KernelKMeans(n_clusters=self.n_clusters, **keywords)
        return

    def run(self):
        """
        Run the models
        """
        if self.method == "kmeans": self.obj.fit(self.data)
        if self.method == "kmedoids": self.obj.fit(self.data)
        if self.method == "kmodes": self.obj.fit(self.data)
        if self.method == "kmedians": self.obj.fit(self.data)
        if self.method == "fuzzykmeans": self.obj.fit(self.data)
        if self.method == "meanshift": self.obj.fit(self.data)
        if self.method == "kernelkmeans": self.obj.fit(self.data)
        return
