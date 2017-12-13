from collections import Counter

import numpy as np
import pandas as pd
from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs, load_mcmc_diffs
from utils.utils import extract_runs_w_timestamp

if __name__ == '__main__':
    n_users = -1
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)
    passdiffs, stretches, passquals, all_qids = load_new_diffs()
    all_qids = list(all_qids)

    users = np.unique(users)
    n_users = len(users)
    print("kickoff for n_users?",n_users)

    n_qids = len(all_qids)
    width = len(cats) #+1 for counter

    qmx = np.zeros(shape=(n_qids,width))
    sqmx = np.zeros(shape=(n_qids,width))
    fqmx = np.zeros(shape=(n_qids,width))

    cqmx = np.zeros(shape=n_qids)
    csqmx = np.zeros(shape=n_qids)
    cfqmx = np.zeros(shape=n_qids)

    usersf = open("direct_mcmc_users.txt","w")

    eps_cnt=0
    qenc = np.zeros(shape=len(cats))
    for u in users:
        qenc[:,:] = 0.0
        xp=0
        eps_cnt +=1
        print("user = ", u)
        usersf.write(u+"\n")
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            q = q.replace("|","~")
            qix = all_qids.index(q)
            xp+=1
            qmx[qix] += qenc
            cqmx[qix] += 1
            if n_pass>0:
                sqmx[qix] += qenc
                csqmx[qix] += 1
            else:
                fqmx[qix] += qenc
                cfqmx[qix] += 1
            catix = cat_ixs[cat_lookup[q]]
            qenc[catix] += 1

    eps_cnt += 1
    print("#episodes=", eps_cnt)
    print("avg ep len=", (xp/float(eps_cnt)))

    qmx = np.divide(qmx,cqmx[:,None])
    fqmx = np.divide(fqmx,cfqmx[:,None])
    sqmx = np.divide(sqmx,csqmx[:,None])

    np.nan_to_num(qmx, copy=False)
    np.nan_to_num(fqmx, copy=False)
    np.nan_to_num(sqmx, copy=False)

    # qmx=np.hstack((qmx,cqmx))
    # fqmx=np.hstack((fqmx,cfqmx))
    # sqmx=np.hstack((sqmx,csqmx))

    pqmx = pd.DataFrame(qmx, index=all_qids, columns=cats)#+["run_count"])
    pfqmx = pd.DataFrame(fqmx, index=all_qids, columns=cats)#+["run_count"])
    psqmx = pd.DataFrame(sqmx, index=all_qids, columns=cats)#+["run_count"])

    # pqmx = pqmx[np.sum(pqmx, axis=1)>0]
    # psqmx = psqmx[np.sum(psqmx, axis=1)>0]
    # pfqmx = pfqmx[np.sum(pfqmx, axis=1)>0]

    pqmx.to_csv("qmx.csv")
    pfqmx.to_csv("fqmx.csv")
    psqmx.to_csv("sqmx.csv")

    toplot = pd.DataFrame(index=all_qids, columns=["XP","S_XP","F_XP"])
    toplot["XP"]=np.sum(qmx, axis=1)
    toplot["S_XP"]=np.sum(sqmx, axis=1)
    toplot["F_XP"]=np.sum(fqmx, axis=1)
    toplot.to_csv("toplot.csv")
