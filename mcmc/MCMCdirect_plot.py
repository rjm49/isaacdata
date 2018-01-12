import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ggplot import *

mcmcdf = pd.read_csv("dir_mcmc_results.csv", index_col=1, header=0)
mcmcdf = mcmcdf[mcmcdf["RUNS"]>=100]
# mcmcdf.plot(kind="scatter", x="LV", y="muRTS")
# mcmcdf.plot(kind="scatter", x="LV", y="muRTF")
# plt.show()

rtfs = mcmcdf.loc[:,["LV","mdRTF","muRTF"]]
rtfs["LV"] = rtfs["LV"]+0.1
gg = ggplot(aes(), data=mcmcdf) +\
     geom_point(mapping=aes(x="LV", y="muRTS"), color="blue", size=2, lab="muRTS") + \
     geom_point(data=rtfs, mapping=aes(x="LV", y="muRTF"), color="red", size=2, lab="muRTF") +\
     labs(x="Expert level, $\Lambda_q$", y="Mean pathlength to (pass=blue) and (fail=red)", title="Qn outcome path-lengths (MCMC)")

print(gg)

mcmcdf_sf = pd.read_csv("sf_mcmc_results.csv", index_col=0, header=0)
total_runs = mcmcdf_sf["TX"].sum()
mcmcdf_sf["SXP"]=mcmcdf_sf["SX"]/total_runs
mcmcdf_sf["FXP"]=mcmcdf_sf["FX"]/total_runs
mcmcdf_sf["LV_F"]=mcmcdf_sf["LV"]+0.1

gg = ggplot(aes(), data=mcmcdf_sf) + \
     labs(x="Expert level, $\Lambda_q$", y="Pr(reach and pass(red=fail))", title="Qn outcome prob's (MCMC)") + \
     geom_point(mapping=aes("LV", "SXP"), size=2, color="blue") + \
     geom_point(mapping=aes("LV_F", "FXP"), size=2, color="red")

print(gg)
