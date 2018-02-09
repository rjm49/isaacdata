'''
Created on 1 Oct 2017

@author:  Russell Moore
'''
from random import shuffle
import pandas as pd
import numpy
from numpy import isnan
from collections import OrderedDict
import random
import os

def init_objects(n_users, path="../../../isaac_data_files/", seed=None):
    qmeta = pd.read_csv(path+"qmeta.csv", header=None)
    #print(qmeta)
    users = open(path+"users.csv").read().splitlines()
    users = [u for u in users if u.isnumeric()]
    
    random.seed(seed)
    shuffle(users)
    if n_users>0:
        users = users[0: n_users]
    print("BackfitUtils, no users =", len(users))

    cats= []
    diffs = OrderedDict()

    #fields for qmeta
    QID=1
    LEV=2
    SUB=3
    FLD=4
    TOP=5
    DIF=8
    
    
    levels = OrderedDict()
    cat_lookup = OrderedDict()
    cat_ixs = OrderedDict()
    all_qids = set()
    for r in qmeta.itertuples():
        q_id = r[QID]
        all_qids.add(q_id)
        cat = str(r[SUB])+"/"+str(r[FLD])+"/"+str(r[TOP])
        cats.append(cat)
        cat_lookup[q_id]= cat
        diff_raw = r[DIF]
        diffs[q_id] = -1 if (diff_raw == float("inf")) else diff_raw

        lv = numpy.float(r[LEV])
        levels[q_id]= 0 if isnan(lv) else lv


    #replace any negative (formerlly inf) values with the max system difficulty
    max_diff = max(diffs.values())
    for k, v in diffs.items():
            if v < 0:
                diffs[k] = max_diff
   
    cats = list(set(cats))
    for ix, c in enumerate(cats):
        cat_ixs[c]=ix

    return cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs

