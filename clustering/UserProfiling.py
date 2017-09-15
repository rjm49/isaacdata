import sklearn as sk

'''
Created on 24 Aug 2017

@author: Russell
'''
import os
import sys
from random import shuffle
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

import pandas
import numpy
import matplotlib.pyplot as plt
from math import isnan
from collections import Counter
from utils.utils import extract_runs_w_timestamp, ATT_QID, ATT_COR, ATT_TIM

decay = 0.999

def user_is_interesting(u):
    uqatts = pandas.read_csv("../by_user/{}.txt".format(u), header=None)
    print("uquatts\n",uqatts)

    uqatts[ATT_TIM] = pandas.to_datetime(uqatts[ATT_TIM], format='%Y-%m-%d %H:%M:%S.%f')

    hasPasses =  True in uqatts[ATT_COR]
    uniqQids = len(uqatts[ATT_QID].value_counts())
    timeFrame = max(uqatts[0]) - min(uqatts[0])    
    oneDay = datetime.timedelta(days=1)
    
    if(hasPasses and uniqQids>1 and timeFrame>oneDay):
        return True
    else:
        return False

def remove_zero_rc(arr, indx):
    zrows = numpy.nonzero(arr.sum(axis=1)==0)
    zcols = numpy.nonzero(arr.sum(axis=0)==0)
    arr = numpy.delete(arr, zrows, axis=0)
    arr = numpy.delete(arr, zcols, axis=1)
    return arr

create_xm = True
plot = True
if __name__ == '__main__':

    #build user experience matrix here....
    qmeta = pandas.read_csv("../qmeta.csv", header=None)
    users = open("../users.csv").read().splitlines()
    shuffle(users)#, lambda: 0.666)

    n_users = 30
    print("Determining interesting users...")
    interesting_users=[]
    for v in users:
        if user_is_interesting(v):
            interesting_users.append(v)
            if len(interesting_users) == n_users:
                break
    print("got users:",interesting_users)

    users  = interesting_users
#     users = users[0:1000]
    
    levels = set()
    lev_lookup = {}
    
    SUB=3
    FLD=4
    TOP=5

    cats = set()
    cat_lookup = {}    
    cat_ixs = {}
    combo_ix=set()
    
    all_qids = set()
    for line in qmeta.itertuples():
        qid = line._1
        all_qids.add(qid)
        
        cat = str(line[SUB])+"/"+str(line[FLD])+"/"+str(line[TOP])
        cats.add(cat)
        cat_lookup[qid]= cat
        
        lev= -1 if isnan(line._2) else int(line._2)
        levels.add(lev)
        lev_lookup[qid]=lev
        
#         c_L = str(lev) # + "_" + str(lev)
        combo_ix.add(cat)
        
    for it in lev_lookup.items():   
        print(it[0],"\t\t\t",it[1])

    q_ixs={}
    #all_qids = unique(all_qids)
    for ix, q in enumerate(all_qids):
        q_ixs[q]=ix

    for ix, c in enumerate(cats):
        cat_ixs[c]=ix
        
    if create_xm:
        #What stuff do we want to record about the user
        #x axis = time scale in real time
        #series for:
        #num of questions at each level
        
#         active_user_lookup = OrderedDict()
#         exp_mx_lookup = OrderedDict()

        exp_mx = None
        w=5
        h=len(users)//w
        fig, axes = plt.subplots(h,w)
        for uix,u in enumerate(users):
            lev_cnt = Counter()
            uqatts = pandas.read_csv("../by_user/{}.txt".format(u), header=None)
            print("uquatts\n",uqatts)

            exp_mx= pandas.DataFrame(columns=["timestamp"]+list(levels))
            exp_mx.fillna(0, inplace=True)
            runs= extract_runs_w_timestamp(uqatts)
            print(u,"runs\n",runs)
            for rix, run in enumerate(runs):
                ts, q, n_atts, n_pass = run
                q = q.replace("|","~")
#                 L = lev_lookup[q]
                c = cat_lookup[q]
                #decay the user's previous career
                #exp_mx[uix] = decay * exp_mx[uix]
                lev = lev_lookup[q]
                if(n_pass > 0):
                    lev_cnt[lev] += 1.0
                exp_mx.loc[rix,"timestamp"]=ts
                for L in levels:
                    exp_mx.loc[rix,L] = lev_cnt[L]
    
#         exp_mx, all_qids = remove_zero_rc(exp_mx, all_qids)
#             if(u.startswith("causnjv")):
            print("data for", u)
            print(exp_mx)
#                 exit()
            exp_mx['timestamp'] = pandas.to_datetime(exp_mx['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            this_plot_row = uix//w
            this_plot_col = uix%w
            print(uix,"plot to:", this_plot_row, this_plot_col)
            exp_mx.plot(ax=axes[this_plot_row, this_plot_col], title=u , x="timestamp", alpha=0.5, kind='area', stacked=True, linewidth=0)
            axes[this_plot_row, this_plot_col].legend(fontsize=4)
        plt.show()

