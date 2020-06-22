#!/usr/bin/env python

"""optimize.py: module is dedicated to optimize the parameters of the clustering algoritms."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import argparse
import sys
sys.path.extend(["algorithms/", "algorithms/extra/"])
import datetime as dt
import os
from dateutil import parser as dparser
import numpy as np

from get_sd_data import FetchData
import utils
from skills import Skills
from plot_utils import 

from partitional import Partition
from hierarchical import Hierarchi
from mixture import Mixtures
from density import DBased
from spectral import Spectral
from gridbased import GBased


class Model(object):
    """ Class is dedicated to run a model and optimize the parameters """

    def __init__(self, rad, stime, etime, args):
        """
        Initialize all the parameters needed to run the model
        """
        self.rad = rad
        self.stime = stime
        self.etime = etime
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        self._ini_()
        self._run()
        if hasattr(self, "skills") and self.skills: self._est_skill_()
        if hasattr(self, "plot") and self.plot: self._plot_estimates_()
        return

    def _plot_estimates(self):
        """
        Plot estimations
        """
        rtp = RangeTimePlot(nrang, unique_times, fig_name, num_subplots=3)
        return

    def _ini_(self, params=["bmnum", "noise.sky", "tfreq", "v", "p_l", "w_l", "slist", "elv", "time_index"], 
            v_params=["elv", "v", "w_l", "gflg", "p_l", "slist", "v_e"]):
        """
        Model initialize
        """
        fd = FetchData(self.rad, [self.stime, self.etime])
        beams, _ = fd.fetch_data(v_params=v_params)
        self.rec = fd.convert_to_pandas(beams)
        self.rec["time_index"] = utils.time_days_to_index([x.to_pydatetime() for x in self.rec["time"].tolist()])
        if hasattr(self, "boxcox") and self.boxcox: self.rec = utils.boxcox_tx(self.rec)
        if hasattr(self, "norm") and self.norm: self.rec = utils.normalize(self.rec, params)
        if self.verbose: print("\n",self.rec.head())
        return

    def _est_skill_(self):
        """
        Estimate model skills
        """
        print("\n Estimating model skills.")
        skill = Skills(self.m.data, self.m.obj.labels_, verbose=self.verbose)
        return


    def _run(self, params=["bmnum", "noise.sky", "tfreq", "v", "p_l", "w_l", "slist", "elv", "time_index"]):
        """
        Run the model
        """
        if self.category == "partition": self.m = Partition(self.model, self.rec[params].values, n_clusters=self.n_clusters)
        self.m.setup()
        self.m.run()
        return

def _del_():
    """
    Delete all generated cache
    """
    os.system("rm *.log")
    os.system("rm -rf __pycache__/")
    os.system("rm -rf algorithms/__pycache__/")
    os.system("rm -rf algorithms/extra/__pycache__/")
    os.system("rm -rf algorithms/sdalgo/__pycache__/")
    os.system("rm -rf skillset/__pycache__/")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", default="partition", help="Algorithm category (default 'partition')")
    parser.add_argument("-m", "--model", default="kmeans", help="Algorithm name (default 'kmeans')")
    parser.add_argument("-nc", "--n_clusters", type=int, default=20, help="Number of clusters (default 20)")
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    parser.add_argument("-s", "--start", default=dt.datetime(2018, 4, 5), help="Start date (default 2018-04-05)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2018, 4, 5, 1), help="End date (default 2018-04-05T01)",
            type=dparser.isoparse)
    parser.add_argument("-cl", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-sk", "--skills", action="store_false", help="Run skill estimate (default True)")
    parser.add_argument("-pl", "--plot", action="store_true", help="Plot estimations (default False)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    Model(args.rad, args.start, args.end, args)
    print("")
    _del_()
