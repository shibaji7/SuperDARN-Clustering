import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class KMeans(BaseEstimator):
    
    def __init__(self, n_clusters, max_iter=100, random_state=0, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        return
        
    def _e_step(self, X):
        self.labels_ = euclidean_distances(X, self.cluster_centers_, squared=True).argmin(axis=1)
        return
        
    def _average(self, X):
        return X.mean(axis=0)
    
    def _m_step(self, X):
        X_center = None
        for center_id in range(self.n_clusters):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers_[center_id] = X_center
            else:
                self.cluster_centers_[center_id] = \
                        self._average(X[center_mask])
        return
    
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        vdata = np.mean(np.var(X, 0))
        
        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.n_clusters]
        self.cluster_centers_ = X[self.labels_]
        
        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()
            
            self._e_step(X)
            self._m_step(X)
            
            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break
            
        return self


class KMedians(KMeans):
    
    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)
        return

    def _average(self, X):
        return np.median(X, axis=0)
