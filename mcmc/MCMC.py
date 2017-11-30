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
    success_qids = []
    failed_qids = []

    usersf = open("sf_mcmc_users.txt","w")

    for u in users:
        print("user = ", u)
        usersf.write(u+"\n")
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            q = q.replace("|","~")
            if q not in observed_qids:
                observed_qids.append(q)
            # now do pass/fail specific records
            if n_pass>0:
                if q not in success_qids:
                    success_qids.append(q)
            else:
                if q not in failed_qids:
                    failed_qids.append(q)


    usersf.close()
    òbserved_qids = ["ROOT"] + observed_qids

    with open("obsqs.txt","w") as qf:
        qf.write("\n".join(observed_qids))
    with open("failqs.txt","w") as ff:
        ff.write("\n".join(failed_qids))
    with open("succqs.txt","w") as sf:
        sf.write("\n".join(success_qids))

    n_qids = len(observed_qids)
    n_sqids = len(success_qids)
    n_fqids = len(failed_qids)

    X = numpy.zeros(shape=(n_qids, n_qids)) # transition matrix (joint)
    S = numpy.zeros(shape=(n_qids, n_qids)) # success matrix
    F = numpy.zeros(shape=(n_qids, n_qids))  # failure matrix

    for u in users:
        print("user = ", u)
        curr_qix = 0
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            q = q.replace("|","~")
            qix = observed_qids.index(q)
            X[curr_qix, qix] = X[curr_qix, qix] + 1
            if n_pass>0: #register successful attempt
                S[curr_qix, qix] = S[curr_qix, qix]+1
            else: #register failed attempt
                F[curr_qix, qix] = F[curr_qix, qix]+1
            curr_qix = qix

    # #Numpy divide each row in X by its sum to normalise to probabilties
    # # print(X.shape)
    # # X = X[numpy.any(X!=0, axis=0)]
    # print(X.shape)
    # X = X / X.sum(axis=1, keepdims=True)
    # X = numpy.nan_to_num(X)
    # numpy.savetxt("X.csv", X, delimiter=",")
    #
    # if F is not None:
    #     #F = F/F.sum(axis=1, keepdims=True)
    #     F = numpy.nan_to_num(F)
    numpy.savetxt("S.csv", S, delimiter=",")
    numpy.savetxt("F.csv", F, delimiter=",")

    q_cnt = numpy.zeros(shape=n_qids)
    s_cnt = numpy.zeros(shape=n_qids)
    f_cnt = numpy.zeros(shape=n_qids)
    i=0
    qix = 0
    seen=set()
    eps_cnt=0
    loop_cnt=0
    dead_cnt=0
    while i < n_steps:
        if (i % 1000)==0:
            print(i,"eps")

        rowsum = numpy.sum(S[qix]) + numpy.sum(F[qix])
        if(rowsum==0): #if we've hit a dead end...
            # print("dead end")
            qix = 0 #reset the random walk
            i+=1
            eps_cnt+=1
            dead_cnt+=1
            seen = set()
            continue
        #else choose from possible next nodes weighted by probability
        # ixs = []
        # probs = []
        # tagd_S = [ (nx,pr/rowsum,"S") for nx,pr in enumerate(S[qix]) ]
        # tagd_F = [ (nx,pr/rowsum,"F") for nx,pr in enumerate(F[qix]) ]
        # all_mvs = tagd_S+tagd_F
        # all_mv_ixs = [ ix for ix,_ in enumerate(all_mvs) ]

        Slen = S[qix].shape[0]
        #print("Slen",Slen)
        all_mvs = numpy.append(S[qix],F[qix]) / rowsum
        #print(all_mvs)
        all_mv_ixs = numpy.arange(len(all_mvs))
        next_mv_ix = numpy.random.choice( all_mv_ixs, p = all_mvs )

        if next_mv_ix in seen:
            # print("seen")
            qix = 0
            i += 1
            seen = set()
            loop_cnt += 1
            eps_cnt += 1
            continue
        seen.add(next_mv_ix)

        if next_mv_ix < Slen: #success!
            qix = next_mv_ix
            #ForS = "S"
            s_cnt[qix] += 1
        else: #failure :(
            qix = next_mv_ix - Slen # failure transition indices are offset by Slen...
            #ForS = "F"
            f_cnt[qix] += 1
        i+=1
    eps_cnt += 1
    print("#episodes=", eps_cnt)
    print("avg ep len=", (i/float(eps_cnt)))
    print("truncated loops=", loop_cnt)
    print("dead ends=", dead_cnt)

    # q_cnt = q_cnt / n_steps # convert to probabilties



    fout = open("sf_mcmc_results.csv", "w")
    for qix, qid in enumerate(observed_qids):
        # fout.write(str(qid) +","+ str(q_cnt[qix])+"\n")
        fout.write(",".join(map(str,( qid, levels[qid], q_cnt[qix],s_cnt[qix],f_cnt[qix], passdiffs[qid], stretches[qid], passquals[qid] )))+"\n")
    fout.close()