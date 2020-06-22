#!/usr/bin/env python

"""geoloc.py: module is dedicated to run davitpy (python 2.7) and store glat-glon into files."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import datetime as dt
import argparse
from netCDF4 import Dataset
import time
import os

from davitpy.pydarn.radar import radar
from davitpy.pydarn.radar.radFov import fov


def geolocate_radar_fov(rad):
    """ Geolocate each range cell """
    r = radar(code=rad)
    s = r.sites[0]
    f = fov(site=s)
    blen, glen = len(f.beams), len(f.gates)
    glat, glon = np.zeros((blen, glen)), np.zeros((blen, glen))
    for i,b in enumerate(f.beams):
        for j,g in enumerate(f.gates):
            glat[i,j], glon[i,j] = f.latCenter[b,g], f.lonCenter[b,g]
    fname = "data/sim/{rad}.geolocate.data.nc".format(rad=rad)
    rootgrp = Dataset(fname, "w")
    rootgrp.description = """ SuperDARN Clustering - Geolocated points for each range cells. """
    rootgrp.history = "Created " + time.ctime(time.time())
    rootgrp.source = "SuperDARN Clustering"
    rootgrp.createDimension("nbeam", blen)
    rootgrp.createDimension("ngate", glen)
    _glat = rootgrp.createVariable("geo_lat","f8",("nbeam","ngate"))
    _glon = rootgrp.createVariable("geo_lon","f8",("nbeam","ngate"))
    _glat[:], _glon[:] = glat, glon
    rootgrp.close()
    os.system("gzip "+fname)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    args = parser.parse_args()
    geolocate_radar_fov(args.rad)
