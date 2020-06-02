#!/usr/bin/env python

"""simulate.py: module is dedicated to simulate the clustering algoritm."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.extend(["algorithms/", "algorithms/extra/"])
import datetime as dt
import os
import numpy as np

from get_sd_data import FetchData
import utils
from skills import Skills

from partitional import Partition
from hierarchical import Hierarchi


def run_all_partion_clustering(rad, date_range, boxcox=True, 
        params=["bmnum", "noise.sky", "tfreq", "v", "p_l", "w_l", "slist", "elv", "time_index"], 
        methods = ["kmeans", "kmodes", "kmedians", "fuzzykmeans", "meanshift", "kernelkmeans"],
        n_clusters=20):
    """
    Invoke all partitioned clustering algorithm
    rad: Radar code
    date_range: Date range
    """
    fd = FetchData(rad, date_range)
    beams, _ = fd.fetch_data(v_params=["elv", "v", "w_l", "gflg", "p_l", "slist", "v_e"])
    rec = fd.convert_to_pandas(beams)
    rec["time_index"] = utils.time_days_to_index([x.to_pydatetime() for x in rec["time"].tolist()])
    if boxcox: rec = utils.boxcox_tx(rec)
    print("\n",rec.head())
    for method in methods:
        print("\n >> Running {c} clustering".format(c=method))
        model = Partition(method, rec[params].values, n_clusters=n_clusters)
        model.setup()
        model.run()

        print("\n Estimating model skills.")
        skill = Skills(model.data, model.obj.labels_)
    return

def run_all_hierarchi_clustering(rad, date_range, boxcox=True,
        params=["bmnum", "noise.sky", "tfreq", "v", "p_l", "w_l", "slist", "elv", "time_index"],
        methods = ["agglomerative", "feature"],
        n_clusters=[20,5]):
    """
    Invoke all partitioned clustering algorithm
    rad: Radar code
    date_range: Date range
    """
    fd = FetchData(rad, date_range)
    beams, _ = fd.fetch_data(v_params=["elv", "v", "w_l", "gflg", "p_l", "slist", "v_e"])
    rec = fd.convert_to_pandas(beams)
    rec["time_index"] = utils.time_days_to_index([x.to_pydatetime() for x in rec["time"].tolist()])
    if boxcox: rec = utils.boxcox_tx(rec)
    print("\n",rec.head())
    for _i, method in enumerate(methods):
        print("\n >> Running {c} clustering".format(c=method))
        model = Hierarchi(method, rec[params].values, n_clusters=n_clusters[_i])
        model.setup()
        model.run()
        
        print("\n Estimating model skills.")
        skill = Skills(model.data, model.obj.labels_)
    return



if __name__ == "__main__":
    run_all_partion_clustering("sas", [dt.datetime(2018, 4, 5), dt.datetime(2018, 4, 5, 1)], methods=["kmeans", "kmodes", "kmedians"])
    run_all_hierarchi_clustering("sas", [dt.datetime(2018, 4, 5), dt.datetime(2018, 4, 5, 1)], methods=["agglomerative"])
    os.system("rm *.log")
    os.system("rm -rf __pycache__/")
    os.system("rm -rf algorithms/__pycache__/")
    os.system("rm -rf algorithms/extra/__pycache__/")
    os.system("rm -rf skillset/__pycache__/")

