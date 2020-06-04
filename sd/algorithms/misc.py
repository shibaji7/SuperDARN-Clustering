#!/usr/bin/env python

"""
misc.py: module is deddicated to run miscellaneous algorithms, implemented by https://pypi.org/project/pyclustering/.
    - BIRCH
    - BSAS
    - CLARANS
    - CURE
    - Elbow
    - EMA
    - Fuzzy C-Means
    - GA
    - G-Means
    - HSyncNet
    - K-Means++
    - MBSAS
    - ROCK
    - Silhouette
    - SOM-SC
    - SyncNet
    - Sync-SOM
    - TTSAS
    - X-Means
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

from pyclustering.cluster.brich import brich

class Mixtures(object):
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
        self.method = method
        self.data = [data[x,:].tolist() for x in range(data.shape[0])]
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
        setattr(self.obj, "lables_", lables_)
        return

    def extract_noise(self):
        """
        Extract noise information from outputs
        """
        self.noise = self.obj.get_noise()
        noise_ = np.zeros(self.L)
        setattr(self.obj, "noise_", noise_)
        return

    def setup(self, **keywords):
        """
        Setup the algorithms
        """
        for key in keywords.keys():
            setattr(self, key, keywords[key])

        if self.method == "brich": self.obj = brich(self.data, self.n_clusters, branching_factor=50, max_node_entries=200, diameter=0.5, type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE, entry_size_limit=500, diameter_multiplier=1.5 )
        return

    def run(self):
        """
        Run the models
        """
        return
