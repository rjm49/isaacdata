'''
Created on 7 Sep 2017

@author:  Russell Moore
'''
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from numpy import sqrt

SCORE_MODE_AVG = "SC_AVG"
SCORE_MODE_REPLACE = "SC_REPLACE"
SCORE_MODE_ACCUM = "SC_ACCUM"
SCORE_MODE_DECAY = "SC_DECAY"
DW_STRETCH = "STRETCH"
DW_LEVEL = "LEVEL"
DW_NO_WEIGHT = "NO_WGT"
DW_BINARY = "BINARY"
DW_NATTS = "N_ATTS"
DW_PASSRATE = "PASSRATE"
DW_MCMC = "MCMC"

ATT_TIM=0
ATT_COR=1
ATT_QID=2

def extract_runs_w_timestamp(attempts):
    times = []
    qids = []
    cors = []
    lens = []
    num_correct = 0
    num_attempts = 0
    c = Counter()
    run_qid=None
    tm=None

    for ix,att_series in attempts.iterrows():
        #we want to isolate contiguous atts against a question into "qids" and determine whether each run was successful or not
        #since students can repeatedly run against a question, it is not sufficient just to filter on the question ID
        new_run_qid = att_series[ATT_QID]
        is_corr = att_series[ATT_COR]
        tm = att_series[ATT_TIM]
        
        if(new_run_qid != run_qid):
#             print("new run")
            qids.append(run_qid)
            lens.append(num_attempts)
            cors.append(num_correct)
            times.append(tm)
            run_qid = new_run_qid
            num_correct = 1 if is_corr else 0
            num_attempts =1
        else:
            num_attempts += 1
            num_correct += (1 if is_corr else 0)

    qids.append(run_qid)
    lens.append(num_attempts)
    cors.append(num_correct)
    times.append(tm)

    uni = list( zip(times, qids,lens,cors) )
        
#     print(uni)
    return uni[1:] #lose the first entry, which is a dummy


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def generate_pass_diff():
    users = open("../users.csv").read().splitlines()
    diffile = open("../pass_diffs.csv", "w")
    print("files opened")
    runcount = Counter()
    passcount = Counter()
    attscount = Counter()
    diffs={}
    stretch={}
    print(len(users))
    print("starting user loop...")
    for i,u in enumerate(users):
            if i%100==0:
                print(i)
#            print("user = ", u)
            #load user episode file
            attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)    
            runs = extract_runs_w_timestamp(attempts)
            for run in runs:
                ts, q, n_atts, n_pass = run
                qt = q.replace("|","~")
#                 qt =q
                runcount[qt]+=1
                if(n_pass>0):
                    passcount[qt]+=1
                    attscount[qt]+=n_atts

    print("starting k loop")    
    for k in runcount.keys():
        diffs[k] = passcount[k]/runcount[k]
        stretch[k] = 0 if attscount[k]==0 else ( float("inf") if passcount[k]==0 else attscount[k]/passcount[k] )
        confidence = wilson_score(passcount[k], runcount[k])
        diffile.write(str(k)+","+str(diffs[k])+","+str(stretch[k])+","+str(confidence)+","+str(attscount[k])+"\n")
    diffile.close()
    return diffs

def wilson_score(ns,n):
    chksum = ns+n
    if chksum==0:
        return 0
    
    z = 1.0
    phat = float(ns)/n
    wilsonU = ((phat + z*z/(2*n) + z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
    wilsonL = ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
    print(wilsonU,wilsonL)
    return (wilsonU - wilsonL)

def load_new_diffs(_file="../pass_diffs.csv"):
    passdiffs = {}
    passquals = {}
    stretches = {}
    diff_df = pd.read_csv(_file, header=None)
    for dr in diff_df.iterrows():
        data = dr[1]
        qt = data[0].replace("|","~")
        passdiffs[qt] = data[1]
        stretches[qt] = data[2]
        passquals[qt] = data[3]
    all_qids = passdiffs.keys()
    return passdiffs, stretches, passquals, all_qids

def load_mcmc_diffs(_file="../mcmc/mcmc_results.csv"):
    diff_df = pd.read_csv(_file, header=None)
    mcmcdiffs = {}
    for dr in diff_df.iterrows():
        data = dr[1]
        qt = data[0].replace("|", "~")
        mcmcdiffs[qt] = data[1]
    return mcmcdiffs

def calc_qdiff(qt, passrates, stretches, levels, mcmcdiffs, mode=None):
    if(mode == DW_STRETCH or mode==DW_NATTS):
        return stretches[qt]
    elif(mode == DW_LEVEL):
#         assert not min(levels.values())<0
#         assert not max(levels.values())>7
        return 1+levels[qt] #plus-one to convert [0..6] to [1..7]
    elif(mode == DW_NO_WEIGHT):
        return 1.0
    elif(mode == DW_BINARY):
        return 1.0
    elif(mode == DW_PASSRATE):
        return passrates[qt]
    elif(mode == DW_MCMC):
        if qt not in mcmcdiffs:
            return 0
        else:
            return mcmcdiffs[qt]
    return None

if __name__=="__main__":
    generate_pass_diff()