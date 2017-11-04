from collections import Counter

import numpy
import random
import pandas as pd
import string
from FileLoader import FileLoader
from backfit.BackfitUtils import init_objects
from backfit.utils.utils import load_new_diffs, load_mcmc_diffs
from utils.utils import extract_runs_w_timestamp

if __name__ == '__main__':
    n_users = -1
    n_steps = 100000000
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users)
    passdiffs, stretches, passquals, all_qids = load_new_diffs()
    n_users = len(users)

    observed_qids = []
    for u in users:
        # print("user = ", u)
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            if n_pass>0 and  q not in observed_qids:
                observed_qids.append(q)

    òbserved_qids = ["ROOT"] + observed_qids
    n_qids = len(observed_qids)
    X = numpy.zeros(shape=(n_qids, n_qids))  # init'se a new feature vector w same width as all_X

    for u in users:
        print("user = ", u)
        curr_qix = 0
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            ts, q, n_atts, n_pass = run
            if n_pass>0:
                qix = observed_qids.index(q)
                X[curr_qix, qix] = X[curr_qix, qix]+1
                curr_qix = qix

    #Numpy divide each row in X by its sum to normalise to probabilties
    # print(X.shape)
    # X = X[numpy.any(X!=0, axis=0)]
    print(X.shape)
    X = X / X.sum(axis=1, keepdims=True)
    X = numpy.nan_to_num(X)

    numpy.savetxt("X.csv", X, delimiter=",")

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
        fout.write(str(qid) +","+ str(q_cnt[qix])+"\n")
    fout.close()