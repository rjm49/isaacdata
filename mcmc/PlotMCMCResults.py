import numpy
import pandas
from scipy import stats
from matplotlib import pyplot as plt
from scipy.interpolate import spline, UnivariateSpline
from scipy.optimize import curve_fit

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


STRETCH_IX = 4
PASSRATE_IX = 3
WILSON_IX = 5


df = pandas.DataFrame.from_csv("combined_df.csv", header=None, index_col=None)

invals = numpy.max(df[2]) - df[2]
#invals = numpy.clip(-numpy.log(vals), a_min=0, a_max=1000000) #-vals

mode_df = pandas.DataFrame.from_csv("../atypes.csv", header=None, index_col=None)

mns = numpy.array([])
mnprs = numpy.array([])
mnstrs = numpy.array([])
stds = numpy.array([])
qls = numpy.array([])

for filter in ["ALL", "choice", "quantity", "symbol"]:
    if filter!="ALL":
            non_mcs = mode_df[mode_df[7]!="choice"][0]
print(non_mcs.shape[0],"non mc qs")



print(non_mcs)

df = df[df[0].isin(non_mcs)]
print(df.shape[0])

levels = numpy.unique(df[1])
allinvals = df[2] # numpy.max(df[2]) - df[2]
for L in levels:
    rows = df[df[1]==L]
    #invals = numpy.max(df[2]) - rows[2]
    invals = rows[2]
    m = numpy.mean(invals)
    mstr = numpy.mean(rows[STRETCH_IX])
    mpr = numpy.mean(rows[3])
    mpq = numpy.mean(rows[5])
    std = numpy.std(invals)

    mns = numpy.append(mns, m)
    mnprs = numpy.append(mnprs, mpr)
    mnstrs = numpy.append(mnstrs, mstr)
    stds = numpy.append(stds, std)
    qls = numpy.append(qls, mpq)

q_levels = df[1]
mcmcs = df[2]
stretches = df[STRETCH_IX]
passrates = df[PASSRATE_IX]
wilsons = df[WILSON_IX]

maxv = numpy.max(mcmcs)
#plt.scatter(q_levels, maxv*allinvals/numpy.max(mcmcs), s=0.1, c="#ff8800", alpha=0.3)
#plt.scatter(q_levels, maxv*stretches/numpy.max(stretches), s=0.1, c="#6666cc", alpha=0.3)
#plt.scatter(q_levels, maxv*passrates/numpy.max(passrates), s=0.1, c="#66cc66", alpha=0.3)
plt.errorbar(levels, list(mns), list(stds), linestyle="None", c="#ff8800", fmt="none", capsize=2)
# plt.plot(lvlt, spl(lvlt), c="#ff8800")

def func(x, aa, a, b, c):
    return aa*(x**3) + a*(x*x) + b*x + c

popt, pcov = curve_fit(func, q_levels, allinvals)
plt.plot(levels, func(levels, *popt), c="#ff8800")
#plt.scatter(levels, mns, s=5.0, c="#000000", zorder=6)

#plt.scatter(levels, maxv*mnprs/numpy.max(passrates), s=5.0, c="#00cc00", zorder=5)
popt, pcov = curve_fit(func, q_levels, maxv*passrates/numpy.max(passrates))
#plt.plot(levels, func(levels, *popt))

#plt.scatter(levels, maxv*mnstrs/numpy.max(stretches), s=5.0, c="#0000cc", zorder=4)
popt, pcov = curve_fit(func, q_levels, maxv*stretches/numpy.max(stretches))
#plt.plot(levels, func(levels, *popt))

#plt.scatter(q_levels, maxv*wilsons/numpy.max(wilsons), s=0.1, c="#ee00ee", alpha=0.3)
#plt.scatter(levels, maxv*qls/numpy.max(wilsons), s=5.0, c="#cc00cc", zorder=4)
popt, pcov = curve_fit(func, q_levels, maxv*wilsons/numpy.max(wilsons))
#plt.plot(levels, func(levels, *popt))

#print(corr)
plt.legend()
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
