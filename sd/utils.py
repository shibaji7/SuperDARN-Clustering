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


import numpy as np
from scipy.stats import boxcox
from pysolar.solar import *

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

def boxcox_tx(data, fetures=["v", "w_l", "p_l"]):
    """
    Implement a boxcox transformation
    data: Pandas dataframe
    features: all features need a transformations
    """
    _tx = data.copy()
    for f in fetures:
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
