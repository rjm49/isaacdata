from collections import Counter, defaultdict

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

    n_qids = 1+len(all_qids) #+1 to represent ROOT

    usersf = open("direct_mcmc_users.txt","w")

    # S = np.zeros(shape=(n_qids, n_qids)) # success matrix
    # F = np.zeros(shape=(n_qids, n_qids))  # failure matrix
    s_cnt = np.zeros(shape=n_qids)
    f_cnt = np.zeros(shape=n_qids)
    sa_cnt = np.zeros(shape=n_qids)
    fa_cnt = np.zeros(shape=n_qids)
    # runs_to_reach = np.zeros(shape=n_qids)
    # runs_to_pass = np.zeros(shape=n_qids)
    # runs_to_fail = np.zeros(shape=n_qids)

    runs_to_reach = defaultdict(list)
    runs_to_pass = defaultdict(list)
    runs_to_fail = defaultdict(list)

    eps_cnt=0
    for u in users:
        xp=0
        eps_cnt +=1
        curr_qix=0 #u @ ROOT
        print("user = ", u)
        usersf.write(u+"\n")
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            q = q.replace("|","~")
            L = levels[q]+1
            qix = all_qids.index(q)
            xp+=1
            runs_to_reach[qix].append(xp)
            if n_pass>0:
                # S[curr_qix, qix] +=1
                runs_to_pass[qix].append(xp)
                s_cnt[qix] +=1
                sa_cnt[qix] += n_atts
            else:
                # F[curr_qix, qix] +=1
                runs_to_fail[qix].append(xp)
                f_cnt[qix] +=1
                fa_cnt[qix] += n_atts

    eps_cnt += 1
    print("#episodes=", eps_cnt)
    print("avg ep len=", (xp/float(eps_cnt)))

    # q_cnt = q_cnt / n_steps # convert to probabilties
    dfout = pd.DataFrame(index = all_qids, columns = "QID,LV,RUNS,SX,FX,PRATE,FRATE,SSTR,FSTR,wilson,muRTA,muRTS,muRTF,mdRTA,mdRTS,mdRTF".split(",") )

    for qix, qid in enumerate(all_qids):
        # fout.write(str(qid) +","+ str(q_cnt[qix])+"\n")
        runs = s_cnt[qix] + f_cnt[qix]
        xp_per_run = runs_to_reach[qix]/runs
        xp_per_s = runs_to_pass[qix]/runs
        xp_per_f = runs_to_fail[qix]/runs
        dfout.loc[qid,:] = (qid, levels[qid], runs, s_cnt[qix], f_cnt[qix], s_cnt[qix]/runs, f_cnt[qix]/runs, sa_cnt[qix]/runs, fa_cnt[qix]/runs, passquals[qid], np.mean(runs_to_reach[qix]), np.mean(runs_to_pass[qix]), np.mean(runs_to_fail[qix]), np.median(runs_to_reach[qix]), np.median(runs_to_pass[qix]), np.median(runs_to_fail[qix]) )
        print("set",qid)
    # dfout.dropna(inplace=True)
    print("NAs dropped")
    dfout.to_csv("dir_mcmc_results.csv")
    print("file writ")