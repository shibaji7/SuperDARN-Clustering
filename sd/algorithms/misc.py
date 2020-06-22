#!/usr/bin/env python

"""
misc.py: module is deddicated to run miscellaneous algorithms, implemented by https://pypi.org/project/pyclustering/.
    - BSAS
    - MBSAS
    - TTSAS
    - CLARANS
    - CURE
    - Elbow
    - EMA
    - Fuzzy C-Means
    - GA
    - G-Means
    - HSyncNet
    - K-Means++
    - ROCK
    - Silhouette
    - SOM-SC
    - SyncNet
    - Sync-SOM
    - X-Means
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

from pyclustering.cluster.bsas import bsas
from pyclustering.cluster.mbsas import mbsas
from pyclustering.cluster.ttsas import ttsas
from pyclustering.cluster.clarans import clarans

class Misc(object):
    """All miscellaneous model algorithms are implemened here."""

    def __init__(self, method, data, n_clusters=2, random_state=0):
        """
        Initialize all the parameters.
        method: Name of the algorithms (lower case joined by underscore)
        data: Data (2D Matrix)
        n_clusters: Number of clusters
        random_state: Random initial state
        """

        self.L = data.shape[0]
        self.data = data
        self.method = method
        self.data_list = [data[x,:].tolist() for x in range(data.shape[0])]
        np.random.seed(random_state)
        self.random_state = random_state
        
        self.n_clusters = n_clusters
        self.ccore = True
        return

    def extract_lables(self):
        """
        Extract lables form the cluster
        """
        self.clusters = self.obj.get_clusters()
        C = range(len(self.clusters))
        lables_ = np.zeros(self.L)
        for _c in C:
            lables_[self.clusters[_c]] = _c
        setattr(self.obj, "labels_", lables_)
        return

    def extract_noise(self):
        """
        Extract noise information from outputs
        """
        self.noise = self.obj.get_noise()
        noise_ = np.zeros(self.L)
        setattr(self.obj, "noise_", noise_)
        return

    def _init_(self):
        """
        Initialize model parameters
        """
        if self.method in ["bsas", "mbsas", "ttsas"]:
            self.threshold1 = 1.
            self.maximum_clusters = 3
            self.threshold2 = 2.
        if self.method == "clarans":
            self.numlocal = 1
            self.maxneighbor = 10
        return

    def setup(self, **keywords):
        """
        Setup the algorithms
        """
        self._init_()
        for key in keywords.keys():
            setattr(self, key, keywords[key])

        if self.method == "bsas": self.obj = bsas(self.data_list, self.maximum_clusters, self.threshold1, ccore=self.ccore)
        if self.method == "mbsas": self.obj = mbsas(self.data_list, self.maximum_clusters, self.threshold1, ccore=self.ccore)
        if self.method == "ttsas": self.obj = ttsas(self.data_list, self.threshold1, self.threshold2, ccore=self.ccore)
        if self.method == "clarans": self.obj = clarans(self.data_list, self.n_clusters, self.numlocal, self.maxneighbor)
        return

    def run(self):
        """
        Run the models
        """
        self.obj = self.obj.process()
        self.extract_lables()
        return
