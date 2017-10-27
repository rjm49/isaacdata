import os, sys
import scipy.stats
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm.classes import OneClassSVM, SVC
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from _collections import defaultdict
from math import pi
import random
from random import randint
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

from sys import stderr
from backfit.BackfitUtils import init_objects
from backfit.BackfitTest import train_and_test

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

import pandas as pd
import numpy
import copy
from utils.utils import ATT_QID, ATT_COR, extract_runs_w_timestamp, balanced_subsample

from matplotlib import pyplot as plt
#DEFINE SOME CONSTANTS

SCORE_MODE_AVG = "SC_AVG"
SCORE_MODE_REPLACE = "SC_REPLACE"
SCORE_MODE_ACCUM = "SC_ACCUM"
SCORE_MODE_DECAY = "SC_DECAY"
DW_STRETCH = "STRETCH"
DW_LEVEL = "LEVEL"
DW_NO_WEIGHT = "NO_WGT"

FEAT_F33 = "F33"
FEAT_XNP = "FXNP"
FEAT_6060 = "F6060"

featureset_to_use = FEAT_F33
#force_balanced_classes = True
n_users = 2000
max_runs = None #10000
percTest = 0.1

do_gen_runs = True
train_clf = True
do_test = True
optimise_predictors = True

predictors = [
            DummyClassifier(strategy="stratified"),
#               DummyClassifier(strategy="uniform"),
              #SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
            #SVC(max_iter=-1, class_weight='balanced'),
            #SVC(max_iter=-1, kernel="linear", class_weight='balanced'),
              #PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, tol=1e-3, n_jobs=-1),
#             MLPClassifier(max_iter=1000, nesterovs_momentum=True, early_stopping=True),
              #Perceptron(max_iter=1000),
            #GaussianNB(),
            LogisticRegression(class_weight='balanced'),
#             OneClassSVM()
]

predictor_params = [
                    None,
#                     None,
#                             {'alpha': numpy.logspace(-3, 2) },
                    #{'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
                    #{'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
#                             None,
                    #{'n_iter':200,'activation':['relu'], 'hidden_layer_sizes':[(100,),(100,50),(50,),(20,),(10,)], 'alpha': numpy.logspace(-6,-1) },
#                     {'n_iter':150,'hidden_layer_sizes':[(100,), (66,10)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
#                       None
                        #None,
                        {'n_iter':50,'C': numpy.logspace(-3, 2)},
#                         None,
                    ]


def generate_pass_diff():
    users = open("../users.csv").read().splitlines()
    diffile = open("../pass_diffs.csv", "w")
    runcount = Counter()
    passcount = Counter()
    attscount = Counter()
    diffs={}
    stretch={}
    for u in users:
            print("user = ", u)
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
    for k in runcount.keys():
        diffs[k] = passcount[k]/runcount[k]
        stretch[k] = 0 if attscount[k]==0 else ( float("inf") if passcount[k]==0 else attscount[k]/passcount[k] )
        diffile.write(str(k)+","+str(diffs[k])+","+str(stretch[k])+"\n")
    diffile.close()
    return diffs
    
def generate_run_files(_gen_runs, _featureset_to_use, _score_mode, _w, cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs):
    stem = _featureset_to_use+"_"+_score_mode+"_"+_w
    x_filename= stem+"_X.csv"
    y_filename= stem+"_y.csv"
    if _gen_runs:
        gen_runs(x_filename, y_filename,  _featureset_to_use, _score_mode, _w,  cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs)
    return x_filename,y_filename

def gen_runs(x_filename, y_filename, featureset_to_use, score_mode, diff_weighting, cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs):
    runs_file = open("runs.csv", "w")
    X_file = open(x_filename,"w")
    y_file = open(y_filename,"w")

    #sample arrays must be wide enough to hold all features AND a one-hot encoding of the question)
    n_features = None
    if(featureset_to_use==FEAT_F33):
        n_features = len(cats)
    elif featureset_to_use==FEAT_XNP:
        n_features = 3*len(cats)
    elif featureset_to_use==FEAT_6060:
        n_features = len(all_qids)
    all_X = numpy.zeros(shape=(0,n_features+len(cats)))

    print("using n_features=", n_features)
    
    run_ct= 0
    for u in users:
        print("user = ", u)
        X = numpy.zeros(shape=all_X.shape[1]) #init'se a new feature vector w same width as all_X
        cnts = copy.copy(X)
        X[0:n_features]=-1.0  # set first half of array -1
#             cnts = numpy.zeros(shape=len(cats))
        #load user episode file
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)

        runs = extract_runs_w_timestamp(attempts)
        
        print("run_ct",run_ct)
        for run in runs:
            run_ct+=1
            ts, q, n_atts, n_pass = run
            qt = q.replace("|","~")
            lev = levels[qt]
            if lev<1:
                continue
            
            if(DW_STRETCH==diff_weighting):
                qdiff= diffs[qt]
            elif(DW_LEVEL==diff_weighting):
                qdiff= 1.0 + levels[qt] #plus-one to convert [0..6] to [1..7]
            elif(DW_NO_WEIGHT==diff_weighting):
                qdiff=1.0 #i.e. no weighting
            else:
                print("diff weighting not set to a valid value .. exiting")
                exit(1)

            qcat = cat_lookup[qt]
            catix = cat_ixs[qcat]
            
#                 X= numpy.hstack((X, qdiff))
#                 X= numpy.hstack((X, catix))
            X[n_features-1:]=0.0 #reset latter part of array to zero
            X[n_features-1+catix]=qdiff # set the next q category and diff
            
            
            sc = (qdiff / n_atts) if (n_pass>0) else 0.0
                        
            y = 1.0 if sc>0 else -1.0
            
            #all_y.append(y)
            #all_X = vstack((all_X, X.reshape(1,-1)))
            y_file.write(str(y)+"\n")
            #y = numpy.array([y])
            #print("partfit - - - -", X)
            #if(not dum):
            #    predictor.partial_fit(X.reshape(1,-1),y, classes=numpy.array([-1,1]))
            #predictor.fit(X.reshape(1,-1),y)
            X_file.write(",".join([str(x) for x in X])+"\n")
#             X = update_student_state(featureset_to_use, score_mode, X, sc, n_atts, n_pass, q, catix, cnts)

#             if(SCORE_MODE_DECAY==score_mode):
#                 X = X * 0.9            
#             elif(SCORE_MODE_AVG):
#                 if X[catix]==0:
#                     X[catix]=y * qdiff/n_atts
#                 else:
#                     X[catix] = (X[catix] + y * qdiff/n_atts)/2
#             else:
            if X[catix]==-1:
                X[catix]=0
                
            retain = 0.0
            if(sc>0):
                X[catix] = (retain*X[catix] + sc)/ (1+retain)
            
#             runs_file.write(",".join([str(s) for s in ep])+"\n")

#             if max_runs and run_ct>=max_runs:
#                 break                
    
    runs_file.close()            
    X_file.close()
    y_file.close()

def update_student_state(f_set, mode, state, score, n_atts, n_pass, q_ix, catix, cnts):
    if f_set == FEAT_XNP:
        offset = state.shape[1]/3 #should be an integer!
        xix = catix
        nix = (offset)+catix
        pix = (2*offset)+catix
        state[xix] += 1
        state[nix] += n_atts
        state[pix] += n_pass
    return state
    #NON XNP score modes!    
    if f_set == FEAT_6060:
        if mode==SCORE_MODE_REPLACE:
            state[q_ix] = score
        elif mode==SCORE_MODE_DECAY:
            state *= 0.999
            state[q_ix] += score
        elif mode==SCORE_MODE_ACCUM:        
            state[q_ix] += score
        else:
            stderr.write("Unsupported score mode for F6060, mode code=", mode)
            exit(1)
        return state
    
    if f_set == FEAT_F33:
        ix=catix
        if state[ix]<0: # clear out any -1s
            state[ix]=0.0

        if mode==SCORE_MODE_AVG and cnts!=None:
            state[ix] = (state[ix]*cnts[ix] + score) / (cnts[ix]+1)
            cnts[ix] += 1.0
        elif mode==SCORE_MODE_REPLACE:
            state[ix] = score
        elif mode==SCORE_MODE_DECAY:
#             state *= 0.999
            state[ix] *= 0.9
            state[ix] = state[ix] + score
        elif mode==SCORE_MODE_ACCUM:
            state[ix] += score
        else:
            print("Score mode not set to a valid value, exiting")
            exit(1)
        print(ix,state[ix])
        return state

    stderr.write("NO FEATURE SET CHOSEN, exiting")
    exit(1)
    

if __name__ == '__main__':
    do_scaling = True
    X_train = X_test = X_test_raw = None
    force_balanced_classes = True
    cats, cat_lookup, all_qids, users, stretchdiffs, levels, cat_ixs = init_objects(n_users)
    
    passdiffs = {}
    diff_df = pd.read_csv("../pass_diffs.csv", header=None)
    print(diff_df)
    for dr in diff_df.iterrows():
        data = dr[1]
        passdiffs[ data[0].replace("|","~") ] = data[1]
    all_qids = passdiffs.keys()
    
    #IMPORTANT DIFF SETTING HERE
    diffs = passdiffs
        
    n_users = len(users)
    report_name = "report_{}_fb{}_opt{}_scale{}_{}.txt".format(n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), featureset_to_use)
    report = open(report_name,"w")
    for w in [DW_STRETCH]: #, DW_STRETCH, DW_LEVEL]:
        for sm in [SCORE_MODE_REPLACE]: #, SCORE_MODE_ACCUM]:
            xfn, yfn = generate_run_files(do_gen_runs, featureset_to_use, sm, w, cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs)
            print(cat_ixs)
            X_train, X_test, scaler = train_and_test(predictors, predictor_params, xfn, yfn, n_users, percTest, featureset_to_use, sm, w, force_balanced_classes, do_scaling, optimise_predictors, report=report)
    report.close()

    print("complete, report file is:", report_name)    
    exit()
    
    n = X_test.shape[0]
    random.seed(666)
    rix = randint(0,(n-1))
    rnd_K = X_test[rix]
    
    rnd_K = rnd_K[0:33]
#     K_pcs = pca.transform(rnd_K)

    #predict limit of ZPD for each Qn on Isaac
    maxcats = numpy.zeros((len(cats),))
    allstuff = numpy.zeros((len(cats),))
    cnts = numpy.zeros((len(cats),))
    p = predictors[1] # use the LogReg classifier
    for q in all_qids:
        print(q)
        catix = cat_ixs[cat_lookup[q]]
        diff = diffs[q]
        ohe = numpy.zeros((len(cats),))
        ohe[catix] = diff
#         print(ohe.shape)
#         print(ohe)
#         print(rnd_K.shape)
        test_case = numpy.append(rnd_K, ohe)
#         print(test_case.shape)
#         print(test_case)
        yprd = p.predict(test_case.reshape(1,-1))
        print(catix, diff, "-> yprd=",yprd)
        if yprd==1:
            allstuff[catix]+=diff
            cnts[catix]+=1
        if yprd==1 and diff>maxcats[catix]:
            maxcats[catix] = diff
        print(maxcats)

    allstuff = allstuff / cnts

    rnd_K = numpy.abs(rnd_K)
    N = rnd_K.shape[0]
    polar_ticks = [n / float(N) * 2 * pi for n in range(N)]

# Because our chart will be circular we need to append a copy of the first 
# value of each list at the end of each list with data

    maxcats = numpy.append(maxcats, numpy.zeros((33,)))
    maxcats = scaler.transform(maxcats.reshape(1,-1))
    maxcats = maxcats.reshape(66,)    
    maxcats = maxcats[0:33,]
    maxcats = numpy.abs(maxcats)
    
        
    rnd_K = numpy.append(rnd_K, rnd_K[0])
    maxcats = numpy.append(maxcats, maxcats[0])
    polar_ticks.append(polar_ticks[0])

    fig = plt.figure(figsize=(8,8))
    plt.title("Sample K-zone", size=18)
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")
    ax= fig.add_subplot(111, polar=True)
    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    print(rnd_K)
    print(maxcats)
    
# Plot data
    ax.plot(polar_ticks, rnd_K, linewidth=0, linestyle='solid', zorder=3)
    ax.fill(polar_ticks, rnd_K, 'b', alpha=0.3)
    ax.plot(polar_ticks, maxcats, linewidth=0, linestyle='solid', zorder=2)
    ax.fill(polar_ticks, maxcats, 'r', alpha=0.2)
    print(rix)
    plt.show()

    print("complete, report file is:", report_name)