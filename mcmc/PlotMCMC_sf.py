import matplotlib
import numpy
import pandas
import scipy
from scipy import stats
from matplotlib import pyplot as plt
from scipy.interpolate import spline, UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs


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

n_users = -1

def print_stats(lv, qtype, ttype, series):
    k2 = scipy.stats.normaltest(series)
    print( "\t".join(map(str,[qtype, lv, ttype, numpy.min(series), numpy.max(series), numpy.mean(series), numpy.std(series), numpy.median(series), scipy.stats.skew(series), k2[0], k2[1]])))

def plot_it(df, ax, filter_name, offs):
    mcmc_mns = numpy.array([])
    mcmc_stds = numpy.array([])
    levels = numpy.unique(df["LV"])

    for L in levels:
        rows = df[df["LV"]==L]
        #invals = numpy.max(df[2]) - rows[2]
        invals = rows["LV"]
        sx_vals = rows["SX"]
        fx_vals = rows["FX"]
        m = numpy.mean(invals)
        std = numpy.std(invals)
        mcmc_mns = numpy.append(mcmc_mns, m)
        mcmc_stds = numpy.append(mcmc_stds, std)

    q_levels = df["LV"]+offs
    l_offset = levels+offs
    mcmc_txs = df["TX"]
    mcmc_sxs = df["SX"]
    mcmc_fxs = df["FX"]
    col = ["#000000","#448844","#ff8800","#880000","#00ffff","#0000ff","#ff00ff","#0000ff"]

    # maxv = numpy.max(mcmcs)
    #plt.scatter(q_levels, maxv*allinvals/numpy.max(mcmcs), s=0.1, c="#ff8800", alpha=0.3)
    #plt.scatter(q_levels, maxv*stretches/numpy.max(stretches), s=0.1, c="#6666cc", alpha=0.3)
    #plt.scatter(q_levels, maxv*passrates/numpy.max(passrates), s=0.1, c="#66cc66", alpha=0.3)
    # plt.plot(lvlt, spl(lvlt), c="#ff8800")

    dotalpha=0.05

    for lv in levels:
        print_stats(lv, filter_name, "TX", mcmc_txs[df["LV"]==lv])
        print_stats(lv, filter_name, "SX", mcmc_sxs[df["LV"]==lv])
        print_stats(lv, filter_name, "FX", mcmc_fxs[df["LV"]==lv])

    ax.scatter(q_levels-0.02, mcmc_txs, s=10.0, alpha=dotalpha, label=None, c="#000000")

    ax.scatter(q_levels, mcmc_sxs, s=10.0, alpha=dotalpha, label=None, c=col[1+int(offs*10)])
    ax.scatter(q_levels+0.01, mcmc_fxs, s=10.0, alpha=dotalpha, label=None, c=col[4+int(offs*10)])
    # ax.plot(l_offset, mcmc_mns, alpha=0.5, c=col[1+int(offs*10)], label=filter_name)
    # ax.errorbar(l_offset, mcmc_mns, mcmc_stds, c=col[1+int(offs*10)], fmt="none", capsize=4)
    # ax.scatter(l_offset, mcmc_mns, s=10.0, alpha=1, c="black")

# print(qmode_df.shape[0])
cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users)
passdiffs, stretches, passquals, all_qids = load_new_diffs()
df = pandas.read_csv("sf_mcmc_results.csv", header=0, index_col=None)
print(df.shape)
df = df[df["LV"]>0] # get rid of levels of 0 or less
#df.iloc[:,1] -= 1.0
print(df.shape)
qmode_df = pandas.read_csv("../atypes.csv", header=None, index_col=None)
print(qmode_df.shape)
filters = pandas.unique(qmode_df[7])
matplotlib.rcParams.update({'font.size': 10})
# fig, axs = plt.subplots(6,1)


# fig, ax = plt.subplots(2)
levels = numpy.unique(df["LV"])
tot = numpy.sum(df["TX"])
# for lv in list(levels):
#     tp = df[df["LV"]==lv]
#     #ax[0].hist(tp["TX"]/tot, cumulative=True, normed=0, alpha=0.5, stacked=True)
#     ax[0].hist(tp["SX"], histtype="step", bins=100, cumulative=False, normed=0, alpha=0.5, stacked=False)
#     ax[1].hist(tp["FX"], histtype="step", bins=100, cumulative=False, normed=0, alpha=0.5, stacked=False)

nbins = lambda arr : ((numpy.max(arr)-numpy.min(arr))//1000)
plt.hist(df["TX"], histtype="step", cumulative=False, bins=nbins(df["TX"]), label="attempts", alpha=0.5)
plt.hist(df["SX"], histtype="step", cumulative=False, bins=nbins(df["SX"]), label="success", alpha=0.5)
plt.hist(df["FX"], histtype="step", cumulative=False, bins=nbins(df["FX"]), label="failed", alpha=0.5)
plt.legend()
plt.show()
#exit()

plt.hist(stretches.values(), bins=500)

#plt.hist(df["PR"], bins=100)
plt.show()

fig, ax = plt.subplots(1,1)
offsets = [0.0, 0.1, 0.2, 0.3]
for ix, filter in enumerate(filters): # ["ALL", "choice", "quantity", "symbol"]:
    non_mcs = qmode_df[qmode_df[7]==filter][0]
    nw = df[ df["QID"].isin(non_mcs) ]
    if nw.shape[0]==0:
        continue
    print(filter,"-> nw shape", nw.shape[0])
    # print(nw)

    plot_it(nw, ax, filter, offsets[ix])

fig.subplots_adjust(hspace=.5)
fig.suptitle("MCMC weightings vs expert-suggested levels across qn types")
ax.legend()
ax.set_xlabel("Qn Level")
ax.set_ylabel("MCMC Dwelltime")
# plt.tight_layout()
plt.show()
