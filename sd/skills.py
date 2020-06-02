#!/usr/bin/env python

"""
skills.py: module is dedicated to skills of a clustering algoritm.
    
    Internal Validation class (Skill)
    External Validation 
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
sys.path.append("skillset/")
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from internal_validation import InternalIndex

class Skills(object):
    """
    Internal Validation class that computes model skills.

    Internal validation methods make it possible to establish
    the quality of the clustering structure without having access
    to external information (i.e. they are based on the information
    provided by data used as input to the clustering algorithm).
    """

    def __init__(self, X, labels, verbose=True):
        """
        Initialize the parameters.
        X: List of n_features-dimensional data points. Each row corresponds to a single data point.
        lables: Predicted labels for each sample.
        """
        self.X = X
        self.labels = labels
        self.clusters = set(self.labels)
        self.n_clusters = len(self.clusters)
        self.n_samples, _ = X.shape
        self.verbose = verbose
        self.iv = InternalIndex(self.n_clusters)
        self._compute()
        return

    def _compute(self):
        """
        Compute all different skills scores
            - Davies Bouldin Score (dbscore)
            - Calinski Harabasz Score (chscore)
            - Silhouette Score (siscore)
            - Ball Hall Score (bhscore)
            - Hartigan Score (hscore)
            - Xu Score (xuscore)
            - Dunn Index (di)
        """
        self._errors()
        self.dbscore = davies_bouldin_score(self.X, self.labels)
        self.chscore = calinski_harabasz_score(self.X, self.labels)
        self.siscore = silhouette_score(self.X, self.labels)
        self.bhscore = self._ball_hall_score()
        self.hscore = self._hartigan_score()
        self.xuscore = self._xu_score()
        if self.verbose:
            print("\n Estimated Skills.")
            print(" Davies Bouldin Score - ",self.dbscore)
            print(" Calinski Harabasz Score - ",self.chscore)
            print(" Silhouette Score - ",self.siscore)
            print(" Ball-Hall Score - ",self.bhscore)
            print(" Hartigan Score - ",self.hscore)
            print(" Xu Score - ",self.xuscore)
            print(" Estimation done.")
        return

    def _xie_beni_score(self):
        return

    def _errors(self):
        """
        Estimate SSE and SSB of the model.
        """
        sse, ssb = 0., 0.
        mean = np.mean(self.X, axis=0)
        for k in self.clusters:
            _x = self.X[self.labels == k]
            mean_k = np.mean(_x, axis=0)
            ssb += len(_x) * np.sum((mean_k - mean) ** 2)
            sse += np.sum((_x - mean_k) ** 2)
        self.sse, self.ssb = sse, ssb
        return

    def _ball_hall_score(self):
        """
        The Ball-Hall index is a dispersion measure based on the quadratic 
        distances of the cluster points with respect to their centroid.
        """
        n_clusters = len(set(self.labels))
        return self.sse / n_clusters

    def _hartigan_score(self):
        """
        The Hartigan index is based on the logarithmic relationship between
        the sum of squares within the cluster and the sum of squares between clusters.
        """
        return np.log(self.ssb/self.sse)

    def _xu_score(self):
        """
        The Xu coefficient takes into account the dimensionality D of the data, 
        the number N of data examples, and the sum of squared errors SSEM form M clusters.
        """
        n_clusters = len(set(self.labels))
        return np.log(n_clusters) + self.X.shape[1] * np.log2(np.sqrt(self.sse/(self.X.shape[1]*self.X.shape[0]**2)))



