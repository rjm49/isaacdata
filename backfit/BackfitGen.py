import os, sys

from sklearn.naive_bayes import BernoulliNB

from backfit.utils.utils import calc_qdiff, DW_STRETCH, DW_NATTS, DW_PASSRATE, DW_LEVEL, DW_MCMC, DW_BINARY, \
    DW_NO_WEIGHT

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print(sys.path)

import pandas as pd
import numpy
from utils.utils import extract_runs_w_timestamp


# def get_qenc(catix, passrate, stretch, lev, mcmc, mode=None):
#     qenc = numpy.zeros(shape=qenc_width)
#     #qenc[:] = 0.0 #reset question encoding
#     weight = 1.0
#     if mode== DW_NATTS or mode==DW_STRETCH:
#         weight = stretch
#     elif mode==DW_PASSRATE:
#         weight = passrate
#     elif mode==DW_LEVEL:
#         weight = lev
#     elif mode==DW_MCMC:
#         weight = mcmc
#     qenc[catix]=weight # set the next q category and diff
#     return qenc

def generate_run_files(alpha, _featureset_to_use, _w, fade, cats, cat_lookup, all_qids, users, stretches, passrates, passquals, levels, mcmcdiffs, cat_ixs, n_classes):
    qenc_width = len(cats)
    stem = _featureset_to_use+"_"+str(alpha) + "_" + str(fade) + "_" + _w
    x_filename= stem+"_X.csv"
    y_filename= stem+"_y.csv"

    X_file = open(stem+"_X.csv","w")
    y_file = open(stem+"_y.csv","w")

    n_features = len(cats)
    #     all_X = numpy.zeros(shape=(0,n_features))

    print("using n_features=", n_features)

    # tmx = numpy.loadtxt("../mcmc/X.csv", delimiter=",") # load the prob transition mx
    # qf = open("../mcmc/obsqs.txt")
    # qindex = [rec.split(",")[0] for rec in qf.read().splitlines()]
    # qf.close()
    #
    # print(tmx.shape[0], len(qindex))
    # assert tmx.shape[0] == len(qindex)
    # print("loaded transition data")

    run_ct= 0
    X = numpy.zeros(shape=n_features) #init'se a new feature vector w same width as all_X
    print("Generating files for {} users...".format(len(users)))
    for u in users:
        print("user = ", u)
        X[:]= 0.0

        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)

        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            run_ct+=1
            ts, q, n_atts, n_pass = run
            qt = q.replace("|","~")
            lev = levels[qt]
            if lev<2:
                continue

            qdiff = calc_qdiff(qt, passrates, stretches, levels, mcmcdiffs, mode=_w)

            catix = cat_ixs[ cat_lookup[qt] ]

            passrate = passrates[qt]
            qpassqual = passquals[qt]
            stretch = stretches[qt]
            mcmc = mcmcdiffs[qt] if qt in mcmcdiffs else 0
            # mcmc = 0
            # if(n_pass > 0):
            #     tailix = qindex.index(qt)
            #     headix = qindex.index(qt)
            #     mcmc = tmx[headix, tailix]
            #     print ("mcmc = ",mcmc)
            #print(qindex)

            qenc = numpy.zeros(shape=qenc_width)
            # qenc[:] = 0.0 #reset question encoding
            q_weight = 1.0
            if _w == DW_NATTS or _w == DW_STRETCH:
                q_weight = stretch
            elif _w == DW_PASSRATE:
                q_weight = passrate
            elif _w == DW_LEVEL:
                q_weight = lev
            elif _w == DW_MCMC:
                q_weight = mcmc
            qenc[catix] = q_weight  # set the next q category and diff

            X_file.write(",".join([str(x) for x in X])+","+",".join([str(e) for e in qenc])+"\n")
            X = X * fade

            a_weight = 1.0
            if _w == DW_BINARY:
                a_weight = 1.0
            elif _w == DW_NATTS:
                a_weight = n_atts
            elif _w == DW_NO_WEIGHT:
                a_weight = 1.0 / n_atts
            elif _w == DW_PASSRATE:
                a_weight = passrate / n_atts
            elif _w == DW_STRETCH:
                a_weight = stretch / n_atts
            elif _w == DW_MCMC:
                a_weight = mcmc / n_atts
            elif _w == DW_LEVEL:
                a_weight = lev / n_atts

            if (n_pass>0):
                if n_classes == 2:
                    y = 0
                else:
                    y = (-1 if n_atts==1 else 0)
                X[catix] = 1 # (1.0-alpha)*X[catix] + alpha*a_weight
            else:
                y = 1
            y_file.write(str(y)+"\n")

        X_file.flush()
        y_file.flush()
    X_file.close()
    y_file.close()
    print(n_users, "users", run_ct,"runs", run_ct/float(n_users), "rpu")
    return x_filename,y_filename
