from collections import Counter

import numpy
import pandas as pd
from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs, load_mcmc_diffs
from utils.utils import extract_runs_w_timestamp

if __name__ == '__main__':
    n_users = -1
    n_steps = 100e6
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)
    passdiffs, stretches, passquals, all_qids = load_new_diffs()

    users = numpy.unique(users)
    n_users = len(users)
    print("kickoff for n_users?",n_users)

    observed_qids = []
    failed_qids = []

    usersf = open("mcmc_users.txt","w")

    for u in users:
        print("user = ", u)
        usersf.write(u+"\n")
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            q = q.replace("|","~")
            if n_pass>0:
                if q not in observed_qids:
                    observed_qids.append(q)
            else:
                if q not in failed_qids:
                    failed_qids.append(q)


    usersf.close()
    òbserved_qids = ["ROOT"] + observed_qids
    with open("obsqs.txt","w") as qf:
        qf.write("\n".join(observed_qids))

    with open("failqs.txt","w") as ff:
        ff.write("\n".join(failed_qids))

    n_qids = len(observed_qids)
    n_fqids = len(failed_qids)
    X = numpy.zeros(shape=(n_qids, n_qids))  # init'se a new feature vector w same width as all_X

    #F=None
    F = numpy.zeros(shape=(n_qids, n_fqids)) #failure matrix

    for u in users:
        print("user = ", u)
        curr_qix = 0
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            q = q.replace("|","~")
            if n_pass>0:
                qix = observed_qids.index(q)
                X[curr_qix, qix] = X[curr_qix, qix]+1
                curr_qix = qix
            else: #register failed attempts here
                if F is not None:
                    qix = failed_qids.index(q)
                    F[curr_qix, qix] = F[curr_qix, qix]+1


    #Numpy divide each row in X by its sum to normalise to probabilties
    # print(X.shape)
    # X = X[numpy.any(X!=0, axis=0)]
    print(X.shape)
    X = X / X.sum(axis=1, keepdims=True)
    X = numpy.nan_to_num(X)
    numpy.savetxt("X.csv", X, delimiter=",")

    if F is not None:
        #F = F/F.sum(axis=1, keepdims=True)
        F = numpy.nan_to_num(F)
        numpy.savetxt("F.csv", F, delimiter=",")

    q_cnt = numpy.zeros(shape=n_qids)
    i=0
    qix = 0
    while i < n_steps:
        q_cnt[qix] += 1
        rowsum = numpy.sum(X[qix])
        if(rowsum==0): #if we've hit a dead end...
            print("dead end")
            qix = 0 #reset the random walk
            i+=1
            continue
        #else choose from possible next nodes weighted by probability
        ixs = []
        probs = []
        for ix, prob in enumerate(X[qix]):
            ixs.append(ix)
            probs.append(prob)
        # print(qix)
        next_qix = numpy.random.choice( ixs, p = probs )
        qix = next_qix
        i+=1

    # q_cnt = q_cnt / n_steps # convert to probabilties

    fout = open("mcmc_results.csv", "w")
    for qix, qid in enumerate(observed_qids):
        # fout.write(str(qid) +","+ str(q_cnt[qix])+"\n")
        fout.write(",".join(map(str,( qid, levels[qid], q_cnt[qix], passdiffs[qid], stretches[qid], passquals[qid] )))+"\n")
    fout.close()