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
from plot_utils import RangeTimePlot, getD2N

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
        self.create_folder()
        for k in vars(args).keys():
            setattr(self, k, vars(args)[k])
        self._ini_()
        self._run_()
        if hasattr(self, "skills") and self.skills: self._est_skill_()
        if hasattr(self, "plot") and self.plot: self._plot_estimates_()
        if hasattr(self, "save") and self.save: self._save_estimates_()
        return
    
    def create_folder(self):
        """ Create folder structure to save outputs """
        self._dir_ = "data/op/{dn}/{rad}/".format(dn=self.stime.strftime("%Y.%m.%d"), rad=self.rad)
        os.system("mkdir -p " + self._dir_)
        return

    def _save_estimates_(self):
        """
        Save estimates into hdf5 files
        """
        fname = "data/op/{rad}.{model}.{dn}.nc".format(rad=self.rad, model=self.model, dn=self.stime.strftime("%Y-%m-%d"))
        dat = self.rec.to_xarray()
        #dat.attrs = {"dbscore": self.skill.dbscore, "chscore": self.skill.chscore, "siscore": self.skill.siscore,
        #        "bhscore": self.skill.bhscore, "hscore": self.skill.hscore, "xuscore": self.skill.xuscore,
        #        "xiebenie": self.skill.xiebenie}
        dat.to_netcdf(fname)
        return

    def _plot_estimates_(self):
        """
        Plot estimations
        """
        beams = set(self.rec["bmnum"])
        for b in beams:
            print("Beam = ", b)
            data_dict = (self.rec[self.rec.bmnum==b]).to_dict(orient="list")
            # Create and show subplots
            fig_name = ("%s\t\t\t\tBeam %d\t\t\t\t%s" % (self.rad.upper(), b, self.start.strftime("%Y%m%d"))).expandtabs()
            #num_subplots = 3 if alg != "Traditional" else 2
            rtp = RangeTimePlot(75, data_dict["time"], fig_name, num_subplots=3)
            #rtp = RangeTimePlot(nrang, unique_times, fig_name, num_subplots=num_subplots)
            #if alg != "Traditional":
            clust_name = ("%s : %d clusters"
                        % (self.model, len(np.unique(np.hstack(data_dict["labels"]))))
                        )
            rtp.addClusterPlot(data_dict, np.array(data_dict["labels"]), b, clust_name)
                
                #isgs_name = ("%s : %s threshold" % (alg, threshold))
                #rtp.addGSISPlot(self.data_dict, gs_flg, b, isgs_name)
            vel_name = "v"
            rtp.addVelPlot(data_dict, b, vel_name, vel_max=250, vel_step=25)
            #    if save_fig:
            plot_date = self.start.strftime("%Y%m%d")
            filename = self._dir_ + "%s_%s_%02d_%s_%s.png" % (self.rad, plot_date, b, self.category, self.model)
            rtp.save(filename)
            rtp.close()
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
        self.rec["time"] = getD2N(self.rec["time"].tolist())
        if hasattr(self, "boxcox") and self.boxcox: self.rec = utils.boxcox_tx(self.rec)
        if hasattr(self, "norm") and self.norm: self.rec = utils.normalize(self.rec, params)
        if self.verbose: print("\n",self.rec.head())
        return

    def _est_skill_(self, params=["p_l"]):
        """
        Estimate model skills
        """
        print("\n Estimating model skills.")
        self.skill = Skills(self.rec[params].values, self.m.obj.labels_, verbose=self.verbose)
        return


    def _run_(self, params=["bmnum", "noise.sky", "tfreq", "v", "w_l", "slist", "elv", "time_index"]):
        """
        Run the model
        """
        if self.category == "partition":
            self.m = Partition(self.model, self.rec[params].values, n_clusters=self.n_clusters)
            self.m.setup()
        if self.category == "density": 
            params = ["bmnum","slist","v","w_l"]
            self.m = DBased(self.model, self.rec[params].values)
            m_params={"dbscan":{"eps":5.}, "optics":{"max_eps":7.,"metric":"minkowski"},
                    "hdbscan":{"metric":"minkowski", "algorithm":"best"}}
            self.m.setup(m_params[self.model])
        self.m.run()
        self.rec["labels"] = self.m.obj.labels_
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
    parser.add_argument("-c", "--category", default="density", help="Algorithm category")
    parser.add_argument("-m", "--model", default="dbscan", help="Algorithm name")
    parser.add_argument("-nc", "--n_clusters", type=int, default=4, help="Number of clusters (default 8)")
    parser.add_argument("-r", "--rad", default="sas", help="SuperDARN radar code (default sas)")
    parser.add_argument("-s", "--start", default=dt.datetime(2018, 4, 5), help="Start date (default 2018-04-05)",
            type=dparser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2018, 4, 6), help="End date (default 2018-04-05T01)",
            type=dparser.isoparse)
    parser.add_argument("-cl", "--clear", action="store_true", help="Clear pervious stored files (default False)")
    parser.add_argument("-sk", "--skills", action="store_true", help="Run skill estimate (default False)")
    parser.add_argument("-pl", "--plot", action="store_false", help="Plot estimations (default True)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity (default False)")
    parser.add_argument("-sv", "--save", action="store_false", help="Increase output verbosity (default True)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    dates = pd.read_csv("data/sim/dates.csv", parse_dates=["dn"])
    rads = pd.read_csv("data/sim/rads.csv")
    for start in dates.dn.tolist():
        args.start = start
        args.end = start + dt.timedelta(days=1)
        for rad in rads.rad.tolist():
            args.rad = rad
            Model(args.rad, args.start, args.end, args)
            break
        break
    print("")
    _del_()
