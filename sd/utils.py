#!/usr/bin/env python

"""utils.py: module is dedicated to utility methods to simulate clustering algoritm."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import numpy as np
from scipy.stats import boxcox
from scipy import stats
from scipy.stats import beta
from pysolar.solar import get_altitude
from netCDF4 import Dataset
from sklearn.preprocessing import MinMaxScaler

class SDScatter(object):
    """ SuperDARN scatter detection and identification module """
    
    IS = 0 # Ionospheric scatter
    GS = 1 # Ground scatter
    SS = 2 # SAIS: Sub-auroral ionospheric scatter
    US = -1 # Unknown scatter
    
    def __init__(self, method=0):
        """ 
            Initialze scatters using different methods [0, 1, 2, 3, 4]
            0: Sundeen et al. |v| + w/3 < 30 m/s
            1: Blanchard et al. |v| + 0.4w < 60 m/s
            2: Blanchard et al. [2009] |v| - 0.139w + 0.00113w^2 < 33.1 m/s
            3: Proposed for SAIS w-[50-{0.7*(v+5)^2}] = 0 m/s
        """
        self.method = method
        return
    
    def get_name(self):
        """ Get the name of the method """
        if self.method == 0: u = "Sundeen"
        if self.method == 1: u = "Blanchard"
        if self.method == 2: u = "Blanchard.2009"
        if self.method == 3: u = "Proposed.SAIS"
        return u
    
    def classify(self, w, v, p):
        """ Classify based on velocity, spectral width, and power """
        if method == 0: self.gs = (np.abs(v) + w/3 < 30).astype(int)
        if method == 1: self.gs = (np.abs(v) + 0.4*w < 60).astype(int)
        if method == 2: self.gs = (np.abs(v) - 0.139*w + 0.00113*w**2 < 33.1).astype(int)
        if method == 3: self.gs = (w - 50-(0.7*(v+5)**2) < 0).astype(int)
        return

def time_days_to_index(time_days):
    """
    Method implemented by Esther Robb to convert datetime to index
    time_days: List of datetime
    """
    return time_sec_to_index(time_days_to_sec(time_days))

def time_sec_to_index(time_sec):
    """
    Method implemented by Esther Robb to convert seconds of a day to index
    time_sec: List of seconds
    """
    uniq_time = np.sort(np.unique(time_sec))
    shifted_time = np.roll(uniq_time, -1)       # circular left shift to compute all the time deltas
    dt = np.min((shifted_time - uniq_time)[:-1])
    # dt = np.median((shifted_time - uniq_time)[:-1])
    index_time = time_sec / dt
    return index_time

def time_days_to_sec(time):
    """
    Method implemented by Esther Robb to convert datetime to seconds
    time: List of times
    """
    time_sec = np.round([(t - time[0]).seconds for t in time])
    #np.round(np.unique(time_sec).reshape((-1, 1)))
    return time_sec

def boxcox_tx(data, features=["v", "w_l", "p_l"]):
    """
    Implement a boxcox transformation
    data: Pandas dataframe
    features: all features need a transformations
    """
    _tx = data.copy()
    for f in features:
        if f == "v" or f == "w_l": _tx[f] = boxcox(np.abs(_tx[f]))[0] * np.sign(_tx[f])
        else: _tx[f] = boxcox(np.abs(_tx[f]))[0] * np.sign(_tx[f])
    return _tx

def get_altitude_azimuth(rad, times, beams, gates):
    """
    Altitude and azimuth angles of scatter location
    rad: Radar code
    times: List of times
    beams: List of beams
    gates: List of gates
    """
    alt, azm = [], []
    for t, b, g in zip(times, beams, gates):
        lat, lon = fetch_scatter_latlon(rad, b, g)
        alt.append(get_altitude(lat, lon, t))
        azm.append(get_azimuth(lat, lon, t))
    return np.array(alt), np.array(azm)

def normalize(data, features, a=0, b=1):
    """
    Normalize the data between a and b
    data: pandas dataframe
    features: all features need a transformations
    """
    scaler = MinMaxScaler(feature_range=(a, b))
    for f in features:
        data[f] = scaler.fit_transform(np.array(data[f]).reshape(-1, 1)).ravel()
    return data

def kde(probs, labels, pth=0.5, pbnd=[.2,.8]):
    """
    Estimate overall confidance of each clusters estimated using KDE
    probs: Probabilitis of IS/GS
    labels: cluster labels
    pth: probability of KDE thresolding
    pbnd: bounding of unknown scatters
    """
    clusters = {}
    for _c in range(len(set(labels))):
        ps = probs[labels == _c]
        a, b, loc, scale = beta.fit(ps, floc=0., fscale=1.)
        gp = 1 - beta.cdf(pth, a, b, loc=0., scale=1.)
        ci = None
        if gp <= pbnd[0]: gflg=0.
        elif gp >= pbnd[1]: gflg=1.
        clusters[_c] = {"gs_prob": gp, "CI": ci}
    return clusters

def get_sza(times, rad, mask=None):
    """
    Fetch sza at all range cell in radar FoV
    rad: Radar code
    mask: mask metrix
    """
    fname = "data/sim/{rad}.geolocate.data.nc.gz".format(rad=rad)
    os.system("gzip -d " + fname)
    fname = fname.replace(".gz","")
    data = Dataset(fname)
    lat, lon = data["geo_lat"], data["geo_lon"]
    sza = []
    for d in times:
        sza.append(get_altitude(lat, lon, d))
    sza = np.array(sza)
    os.system("gzip " + fname)
    return sza

