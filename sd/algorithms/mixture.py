#!/usr/bin/env python

"""
mixture.py: module is deddicated to run different mixture model algorithms.

    Mixture models for cluster analysis  have been addressed in a number of ways. The 
    underlying assumption is that the observations to be clustered are drawn from one of
    several components, and the problem is to estimate the parameters of each component 
    so as to best fit the data. Inferring the parameters of these components and identifying 
    which component produced each observation leads to a clustering of the set of observations.
        - Gaussian Mixture Model
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

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

class Mixtures(object):
    """All mixture model algorithms are implemened here."""

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
        self.random_state = random_state
        self.init_params = "kmeans"
        self.cov = "full"
        self.max_iter = 500
        self.n_init = 5
        self.weight_concentration_prior_type = "dirichlet_process"
        return

    def setup(self, **keywords):
        """
        Setup the algorithms
        """
        for key in keywords.keys():
            setattr(self, key, keywords[key])
        if self.method == "gmm": self.obj = GaussianMixture(n_components=self.n_clusters,
                covariance_type=self.cov, max_iter=self.max_iter, random_state=self.random_state,
                n_init=self.n_init, init_params=self.init_params)
        if self.method == "bgmm": self.obj = BayesianGaussianMixture(n_components=self.n_clusters,
                covariance_type=self.cov, max_iter=self.max_iter, random_state=self.random_state,
                n_init=self.n_init, init_params=self.init_params, 
                weight_concentration_prior_type=self.weight_concentration_prior_type)
        return

    def run(self):
        """
        Run the models
        """
        if self.method == "gmm": 
            self.obj.fit(self.data)
            setattr(self.obj, "labels_", self.obj.predict(self.data))
        if self.method == "bgmm":
            self.obj.fit(self.data)
            setattr(self.obj, "labels_", self.obj.predict(self.data))
        return
