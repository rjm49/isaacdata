import numpy
import pandas
from scipy import stats
from matplotlib import pyplot as plt
from scipy.interpolate import spline, UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs

n_users = 10000
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users)
passdiffs, stretches, passquals, all_qids = load_new_diffs()

# diff_df = pandas.read_csv("../pass_diffs.csv", header=None)
# mcmc_df = pandas.DataFrame.from_csv("mcmc_results.csv", header=None)
#
# combined = open("combined_df.csv","w")
# for ix,row in mcmc_df.iterrows():
#     s = ",".join([ix, str(levels[ix]-1), str(row[1]), str(passdiffs[ix]), str(stretches[ix]), str(passquals[ix]) ])
#     combined.write(s+"\n")
# combined.close()
# exit()

LEVEL_IX = 1
MCMC_IX = 2
PASSRATE_IX = 3
STRETCH_IX = 4
WILSON_IX = 5
GEND_XS_FOR_CURVES = numpy.linspace(1,6,20)

mcmc_mns = numpy.array([])
mnprs = numpy.array([])
mnstrs = numpy.array([])
stds = numpy.array([])
qls = numpy.array([])

stretch_sds = numpy.array([])
passrate_sds = numpy.array([])

df = pandas.DataFrame.from_csv("mcmc_results.csv", header=None, index_col=None)
df = df[df[LEVEL_IX]>0] # get rid of levels of 0 or less
print(df.shape)

qmode_df = pandas.DataFrame.from_csv("../atypes.csv", header=None, index_col=None)

print(qmode_df.shape)

filters = pandas.unique(qmode_df[7])

# print(qmode_df.shape[0])
for filter in filters: # ["ALL", "choice", "quantity", "symbol"]:
    non_mcs = qmode_df[qmode_df[7]==filter][0]
    #print(non_mcs)
    #print(pandas.unique(non_mcs))
    #print(non_mcs.size)
    nw = df[ df[0].isin(non_mcs) ]
    print(filter,"-> nw shape", nw.shape[0])
    # print(nw)
exit()

levels = numpy.unique(df[LEVEL_IX])

for L in levels:
    rows = df[df[LEVEL_IX]==L]
    #invals = numpy.max(df[2]) - rows[2]
    invals = rows[2]
    m = numpy.mean(invals)

    mstr = numpy.mean(rows[STRETCH_IX])
    stretch_sd = numpy.std(rows[STRETCH_IX])

    mpr = numpy.mean(rows[PASSRATE_IX])
    passrate_sd = numpy.std(rows[PASSRATE_IX])

    mpq = numpy.mean(rows[5])
    std = numpy.std(invals)

    mcmc_mns = numpy.append(mcmc_mns, m)

    mnprs = numpy.append(mnprs, mpr)
    mnstrs = numpy.append(mnstrs, mstr)
    stretch_sds = numpy.append(stretch_sds, stretch_sd)
    passrate_sds = numpy.append(passrate_sds, passrate_sd)

    stds = numpy.append(stds, std)
    qls = numpy.append(qls, mpq)

q_levels = df[1]
mcmcs = df[MCMC_IX]
stretches = df[STRETCH_IX]
passrates = df[PASSRATE_IX]
wilsons = df[WILSON_IX]
col = ["#000000","#00ff00","#ff8800","#880000","#008888","#ff8888"]

maxv = numpy.max(mcmcs)
#plt.scatter(q_levels, maxv*allinvals/numpy.max(mcmcs), s=0.1, c="#ff8800", alpha=0.3)
#plt.scatter(q_levels, maxv*stretches/numpy.max(stretches), s=0.1, c="#6666cc", alpha=0.3)
#plt.scatter(q_levels, maxv*passrates/numpy.max(passrates), s=0.1, c="#66cc66", alpha=0.3)
# plt.plot(lvlt, spl(lvlt), c="#ff8800")

def func(x, aa, a, b, c):
    return aa*(x**3) + a*(x*x) + b*x + c


fig, axs = plt.subplots(nrows=1, ncols=3)

popt, pcov = curve_fit(func, q_levels, mcmcs)
axs[0].scatter(q_levels, mcmcs, s=5.0, c=col[MCMC_IX], alpha=0.05, label=None)
axs[0].errorbar(levels, mcmc_mns, stds, linestyle="None", c=col[MCMC_IX], fmt="none", capsize=4)
axs[0].plot(GEND_XS_FOR_CURVES, func(GEND_XS_FOR_CURVES, *popt), label="mcmc", c=col[MCMC_IX])

popt, pcov = curve_fit(func, q_levels, passrates)
axs[1].scatter(q_levels, passrates, s=5.0, c=col[PASSRATE_IX], alpha=0.05, label=None)
axs[1].errorbar(levels, mnprs, passrate_sds, linestyle="None", c=col[PASSRATE_IX], fmt="none", capsize=4)
axs[1].plot(GEND_XS_FOR_CURVES, func(GEND_XS_FOR_CURVES, *popt), label="passrate", c=col[PASSRATE_IX])

popt, pcov = curve_fit(func, q_levels, wilsons)
axs[1].plot(GEND_XS_FOR_CURVES, func(GEND_XS_FOR_CURVES, *popt), label="p/r uncertainty", c=col[WILSON_IX])


popt, pcov = curve_fit(func, q_levels, stretches)
axs[2].scatter(q_levels, stretches, s=5.0, c=col[STRETCH_IX], alpha=0.05, label=None)
axs[2].errorbar(levels, mnstrs, stretch_sds, linestyle="None", c=col[STRETCH_IX], fmt="none", capsize=4)
axs[2].plot(GEND_XS_FOR_CURVES, func(GEND_XS_FOR_CURVES, *popt), label="stretch", c=col[STRETCH_IX])


#print(corr)
fig.suptitle("Data-driven weightings vs expert-suggested levels")
fig.legend()
plt.show()
exit()

ax = plt.gca()
# plt.xticks(ix, xtix, rotation='vertical')
# plt.plot(ix, sorted(vals), label="mcmc prob")
# plt.scatter(ix, slvls, s=0.8, c="#ff880088", label="levels")
# plt.scatter(ix, sstrxs, s=0.4, c="#55880055", label="n_atts")
# plt.scatter(ix, sprxs, s=0.4, c="#88555588", label="passrate")
# plt.scatter(ix, spqxs, s=0.4, c="#55888888", label="wilson")

plt.xlabel("Qn Index")
plt.ylabel("MCMC Stationary Prob")
plt.legend()
plt.show()
