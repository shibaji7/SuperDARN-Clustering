#!/usr/bin/env python

"""sdalgo.py: module is dedicated to SuperDARN custom algorithms."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

from sklearn.mixture import GaussianMixture
from scipy.stats import boxcox
import time
import os
from matplotlib.dates import date2num
from scipy import sparse
import numpy as np
from sklearn.cluster import DBSCAN

class Algorithm(object):
    """
    Superclass for algorithms.
    Contains data processing and plotting functions.
    """
    def __init__(self, start_time, end_time, rad, params):
        self.start_time = start_time
        self.end_time = end_time
        self.rad = rad
        self.params = params
        return

    def _filter_by_time(self, start_time, end_time, data_dict):
        time = data_dict['time']
        start_i, end_i = None, None
        start_time, end_time = date2num(start_time), date2num(end_time)
        if start_time < time[0][0]: # Sometimes start time is a few seconds before the first scan
            start_time = time[0][0]
        for i, t in enumerate(time):
            if np.sum(start_time >= t) > 0 and start_i == None: start_i = i
            if np.sum(end_time >= t) > 0 and start_i != None: end_i = i+1
        data_dict['gate'] = data_dict['gate'][start_i:end_i]
        data_dict['time'] = data_dict['time'][start_i:end_i]
        data_dict['beam'] = data_dict['beam'][start_i:end_i]
        data_dict['vel'] = data_dict['vel'][start_i:end_i]
        data_dict['wid'] = data_dict['wid'][start_i:end_i]
        data_dict['elv'] = data_dict['elv'][start_i:end_i]
        data_dict['trad_gsflg'] = data_dict['trad_gsflg'][start_i:end_i]
        return data_dict

    def _1D_to_scanxscan(self, array):
        """
        Convert a 1-dimensional array to a list of data from each scan
        :param array: a 1D array, length = <total # points in data_dict>
        :return: a list of arrays, shape: <number of scans> x <data points for each scan>
        """
        scans = []
        i = 0
        for s in self.data_dict['gate']:
            scans.append(array[i:i+len(s)])
            i += len(s)
        return scans

class DBSCAN_GMM(GMMAlgorithm):
    """ Run DBSCAN on space/time features, then GMM on space/time/vel/wid """

    def __init__(self, start_time, end_time, rad,
            beam_eps=3, gate_eps=1, scan_eps=1,  # DBSCAN
            minPts=5, eps=1,  # DBSCAN
            n_clusters=5, cov="full",  # GMM
            features=["beam", "gate", "time", "vel", "wid"],  # GMM
            BoxCox=False):
        self.params = {"scan_eps" : scan_eps, "beam_eps": beam_eps, "gate_eps": gate_eps, "eps": eps,
                "min_pts": minPts, "n_clusters" : n_clusters, "cov": cov, "features": features, "BoxCox": BoxCox}
        clust_flg, self.runtime = self._dbscan_gmm()
        self.clust_flg = self._1D_to_scanxscan(clust_flg)
        return

    def _dbscan_gmm(self):
        # Run DBSCAN on space/time features
        X = self._get_dbscan_data_array()
        t0 = time.time()
        db = DBSCAN(eps=self.params["eps"],
                min_samples=self.params["min_pts"]
                ).fit(X)
        db_runtime = time.time() - t0
        # Print # of clusters created by DBSCAN
        db_flg = db.labels_
        gmm_data = self._get_gmm_data_array()
        clust_flg, gmm_runtime = self._gmm_on_existing_clusters(gmm_data, db_flg)
        return clust_flg, db_runtime + gmm_runtime

    def _get_dbscan_data_array(self):
        beam = np.hstack(self.data_dict["beam"])
        gate = np.hstack(self.data_dict["gate"])
        # Get the scan # for each data point
        scan_num = []
        for i, scan in enumerate(self.data_dict["time"]):
            scan_num.extend([i]*len(scan))
            scan_num = np.array(scan_num)
            # Divide each feature by its "epsilon" to create the illusion of DBSCAN having multiple epsilons
            data = np.column_stack((beam / self.params["beam_eps"],
                gate / self.params["gate_eps"],
                scan_num / self.params["scan_eps"]))
                                                                                                                                                            return data

class GridBasedDBAlgorithm():
    """
    Grid-based DBSCAN
    Based on Kellner et al. 2012
    
    This is the fast implementation of Grid-based DBSCAN, with a timefilter added in.
    If you don"t want the timefilter, run it on just 1 scan.
    
    GBDB params dict keys:
    f, g, pts_ratio
    Other class variables unique to GBDB:
    self.C
    self.r_init
    self.dr
    self.dtheta
    """
    
    UNCLASSIFIED = 0
    NOISE = -1
    
    def __init__(self, start_time, end_time, rad, params):
        """
        Initialze the parameters.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.rad = rad
        self.params = params
        # Create the C matrix - ratio of radial / angular distance for each point
        dtheta = self.params["dtheta"] * np.pi / 180.0
        nrang, nbeam = int(self.data_dict["nrang"]), int(self.data_dict["nbeam"])
        self.C = np.zeros((nrang, nbeam))
        for gate in range(nrang):
            for beam in range(nbeam):
                self.C[gate, beam] = self._calculate_ratio(self.params["dr"], dtheta, gate, beam,
                        r_init=self.params["r_init"])
        return

    def _gbdb(self, data, data_i):
        t0 = time.time()
        cluster_id = 1
        clust_flgs = []
        nscans = len(data)
        grid_labels = [np.zeros(data[0].shape).astype(int) for i in range(nscans)]
        for scan_i in range(nscans):
            m_i = data_i[scan_i]
            for grid_id in m_i:
                if grid_labels[scan_i][grid_id] == self.UNCLASSIFIED:
                    if self._expand_cluster(data, grid_labels, scan_i, grid_id, cluster_id):
                        cluster_id = cluster_id + 1
            scan_pt_labels = [grid_labels[scan_i][grid_id] for grid_id in m_i]
            clust_flgs.extend(scan_pt_labels)
        runtime = time.time() - t0
        return np.array(clust_flgs), runtime

    def _in_ellipse(self, p, q, hgt, wid):
        return ((q[0] - p[0])**2.0 / hgt**2.0 + (q[1] - p[1])**2.0 / wid**2.0) <= 1.0

    
    def _calculate_ratio(self, dr, dt, i, j, r_init=0):
        """
        This is the ratio between radial and angular distance for some point on the grid.
        There is very little variance from beam to beam for our radars - down to the 1e-16 level.
        So the rows of this have minimal effect.
        """
        r_init, dr, dt, i, j = float(r_init), float(dr), float(dt), float(i), float(j)
        cij = (r_init + dr * i) / (2.0 * dr) * (np.sin(dt * (j + 1.0) - dt * j) + np.sin(dt * j - dt * (j - 1.0)))
        return cij

    def _get_gbdb_data_matrix(self, data_dict):
        gate = data_dict["gate"]
        beam = data_dict["beam"]
        values = [[True]*len(s) for s in beam]
        ngate = int(data_dict["nrang"])
        nbeam = int(data_dict["nbeam"])
        nscan = len(beam)
        data = []
        data_i = []
        for i in range(nscan):
            m = sparse.csr_matrix((values[i], (gate[i], beam[i])), shape=(ngate, nbeam))
            m_i = list(zip(np.array(gate[i]).astype(int), np.array(beam[i]).astype(int)))
            data.append(m)
            data_i.append(m_i)
        return data, data_i

    def _expand_cluster(self, data, grid_labels, scan_i, grid_id, cluster_id):
        seeds, possible_pts = self._region_query(data, scan_i, grid_id)
        k = possible_pts * self.params["pts_ratio"]
        if len(seeds) < k:
            grid_labels[scan_i][grid_id] = self.NOISE
            return False
        else:
            grid_labels[scan_i][grid_id] = cluster_id
            for seed_id in seeds:
                grid_labels[seed_id[0]][seed_id[1]] = cluster_id
            while len(seeds) > 0:
                current_scan, current_grid = seeds[0][0], seeds[0][1]
                results, possible_pts = self._region_query(data, current_scan, current_grid)
                k = possible_pts * self.params["pts_ratio"]
                if len(results) >= k:
                    for i in range(0, len(results)):
                        result_scan, result_point = results[i][0], results[i][1]
                        if grid_labels[result_scan][result_point] == self.UNCLASSIFIED \
                                or grid_labels[result_scan][result_point] == self.NOISE:
                            if grid_labels[result_scan][result_point] == self.UNCLASSIFIED:
                                seeds.append((result_scan, result_point))
                            grid_labels[result_scan][result_point] = cluster_id
                seeds = seeds[1:]
            return True

    def _region_query(self, data, scan_i, grid_id):
        seeds = []
        hgt = self.params["g"]
        wid = self.params["g"] / (self.params["f"] * self.C[grid_id[0], grid_id[1]])
        ciel_hgt = int(np.ceil(hgt))
        ciel_wid = int(np.ceil(wid))
        
        # Check for neighbors in a box of shape ciel(2*wid), ciel(2*hgt) around the point
        g_min = max(0, grid_id[0] - ciel_hgt)     # gate box
        g_max = min(int(self.data_dict["nrang"]), grid_id[0] + ciel_hgt + 1)
        b_min = max(0, grid_id[1] - ciel_wid)     # beam box
        b_max = min(int(self.data_dict["nbeam"]), grid_id[1] + ciel_wid + 1)
        s_min = max(0, scan_i - self.params["scan_eps"])  # scan box
        s_max = min(len(data), scan_i + self.params["scan_eps"]+1)
        possible_pts = 0
        for g in range(g_min, g_max):
            for b in range(b_min, b_max):
                new_id = (g, b)
                # Add the new point only if it falls within the ellipse defined by wid, hgt
                for s in range(s_min, s_max):   # time filter
                    if self._in_ellipse(new_id, grid_id, hgt, wid):
                        possible_pts += 1
                        if data[s][new_id]:   # Add the point to seeds only if there is a 1 in the sparse matrix there
                            seeds.append((s, new_id))
        return seeds, possible_pts

