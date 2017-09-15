import os
import sys
from sklearn.dummy import DummyClassifier
from _random import Random
from sklearn.metrics.classification import classification_report,\
    precision_recall_fscore_support
from clf_tuning import run_random_search
from sklearn.model_selection._split import train_test_split
import pandas
from sklearn.preprocessing.data import StandardScaler
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

import pandas as pd
from pandas.core.algorithms import unique
from sklearn.svm.classes import LinearSVC, SVC
import sklearn
import numpy
import random
from numpy import nditer, ndarray, vstack
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import pickle
import math
from pandas.core.series import Series
import glob
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from pandas.core.frame import DataFrame
from sklearn.linear_model.perceptron import Perceptron
from _collections import defaultdict
from collections import Counter
import copy
from random import shuffle, randint
from utils.utils import ATT_QID, ATT_COR, extract_runs_w_timestamp, balanced_subsample

#DEFINE SOME CONSTANTS

SCORE_MODE_AVG = "SC_AVG"
SCORE_MODE_REPLACE = "SC_REPLACE"
SCORE_MODE_ACCUM = "SC_ACCUM"
SCORE_MODE_DECAY = "SC_DECAY"
DW_STRETCH = "STRETCH"
DW_LEVEL = "LEVEL"
DW_NONE = "NO_WGT"

score_mode = SCORE_MODE_AVG
diff_weighting = DW_NONE
do_scaling = True

user_n = 1000 # use -1 for all users
#test_n = 100


gen_runs = True
train_clf = True
do_test = True
force_balanced_classes = True
optimise_predictors = True
report_name = "report_{}_fb{}_opt{}_w{}_{}_scale{}.txt".format(user_n, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), diff_weighting, score_mode, ("1" if do_scaling else "0"))

def score_qattempt(u,q, category_lookup, cat_ixs):
    qatts = attempts[attempts.iloc[:,ATT_QID]==q]
    #print(">>>qatts",qatts)
    st = qatts.iloc[0,0]
    et = qatts.iloc[-1,0]
    
    q = q.replace("|","~")
    cat = category_lookup[q]
    feature_ix = cat_ixs[cat]

    attempt_outcomes = qatts[ATT_COR]
    #print(type(attempt_outcomes))
    if( attempt_outcomes.isnull().values.any() ):
        return (None, None, None) #skip onto next iteration if NaN present in this column - not a real q attempt
    
    #sc_raw = numpy.sum(attempt_outcomes)
    sc_raw=0
    for el in attempt_outcomes:
        if el:
            sc_raw += 1
    #print(">>>> >> >>> sc_raw", sc_raw)
    #print(attempt_outcomes)
    #y = numpy.array([ (1 if sc_raw>0 else -1) ])
    
    summary = [st,et,u,q,cat,len(attempt_outcomes),sc_raw]
#     print("SUMMARY ___",summary)
    return (sc_raw, feature_ix, summary)

def update_student_state(mode, state, score, ix, cnts):
    if mode==SCORE_MODE_AVG and cnts!=None:
        state[ix] = (state[ix]*cnts[ix] + score) / (cnts[ix]+1)
        cnts[ix] += 1.0
    elif mode==SCORE_MODE_REPLACE:
        state[ix] = score
    elif mode==SCORE_MODE_DECAY:
        state *= 0.999
        state[ix] += score
    elif mode==SCORE_MODE_ACCUM:
        state[ix] += score
    else:
        print("Score mode not set to a valid value, exiting")
        exit(1)
    return state


    

if __name__ == '__main__':
    qmeta = pd.read_csv("../qmeta.csv", header=None)
    #print(qmeta)
    users = open("../users.csv").read().splitlines()
    shuffle(users, lambda: 0.666)
    if user_n>0:
        users = users[0:user_n]
    
#    predictors = {}
    cats= []
    diffs = {}

    #fields for qmeta
    QID=1
    LEV=2
    SUB=3
    FLD=4
    TOP=5
    DIF=8
    
    
    levels = {}
    cat_lookup = {}
    cat_ixs = {}
    for r in qmeta.itertuples():
        q_id = r[QID]
        cat = str(r[SUB])+"/"+str(r[FLD])+"/"+str(r[TOP])
        cats.append(cat)
        cat_lookup[q_id]= cat
        diff_raw = r[DIF]
        diffs[q_id] = -1 if (diff_raw == float("inf")) else diff_raw

        levels[q_id]= numpy.float(r[LEV])
#         print(r[DIF])
        clsweight = {1:1, -1:6}
        #predictors[q_id] = SGDClassifier()
        predictors = [
                      DummyClassifier(),
                      DummyClassifier(strategy="constant", constant=1),
                      SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
                      SVC(class_weight="balanced"),
                      SVC(kernel="linear", class_weight="balanced"),
                      PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, tol=1e-3, n_jobs=-1),
                      MLPClassifier(max_iter=1000),
                      Perceptron(max_iter=1000)
        ]
        
        predictor_params = [
                            None,
                            None,
                            {'alpha': numpy.logspace(-3, 2) },
                            {'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
                            {'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
                            None,
                            {'alpha': numpy.logspace(-4,2) },
                            None
                            ]
        

    #replace any negative (formerlly inf) values with the max system difficulty
    max_diff = max(diffs.values())
    for k, v in diffs.items():
            if v < 0:
                diffs[k] = max_diff
   
    cats = unique(cats)
    for ix, c in enumerate(cats):
        cat_ixs[c]=ix
    #print(df)
    
    #print(users)
    #random.shuffle(users)
    #print(users)
    #df = pd.DataFrame(index=users_train, columns=cats)
    #counts = pd.DataFrame(index=users_train, columns=cats)

    #2016-06-01 17:08:22.599,False,a_diode_divider|75b49dfb-5934-496f-bed6-d2a2d665485e
    
    
    if gen_runs:
        runs_file = open("runs.csv", "w")
        X_file = open("all_X.csv","w")
        y_file = open("all_y.csv","w")
        tr_pass=tr_fail=0

        all_X = numpy.zeros(shape=(0,2*len(cats)))
        all_y = []
        run_ct= 1
        for u in users:
            print("user = ", u)
            X = numpy.zeros(shape=2*len(cats)) #init'se a new feature vector
            cnts = copy.copy(X)
            X[0:len(cats)-1]=-1.0  # set first half of array -1
#             cnts = numpy.zeros(shape=len(cats))
            #load user episode file
            attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)

            runs = extract_runs_w_timestamp(attempts)
            
            print(run_ct)
            for run in runs:
                run_ct+=1
                ts, q, n_atts, n_pas = run
                qt = q.replace("|","~")
                lev = levels[qt]
                if lev<1:
                    continue
                
                if(DW_STRETCH==diff_weighting):
                    qdiff= diffs[qt]
                elif(DW_LEVEL==diff_weighting):
                    qdiff= levels[qt]
                elif(DW_NONE==diff_weighting):
                    qdiff=1.0 #i.e. no weighting
                else:
                    print("diff weighting not set to a valid value .. exiting")
                    exit(1)
                    
                qcat = cat_lookup[qt]
                catix = cat_ixs[qcat]
                
#                 X= numpy.hstack((X, qdiff))
#                 X= numpy.hstack((X, catix))
                X[len(cats)-1:]=0.0 #reset latter part of array to zero
                X[len(cats)-1+catix]=qdiff # set the next q category and diff
                
                n_atts = run[1]
                n_pass = run[2]
                
                sc = qdiff* n_pass / n_atts
                ep = [u,q,qdiff,cat,n_atts,n_pass,sc]
                
#                 y = 1.0 if sc>0 else -1.0
                y = 1 if sc>0 else 0
#                     print("\n")
#                     print("*** |``   | | |   ***")
#                     print("*** |-   /| | |   ***")
#                     print("*** |   /'| | |__ ***")
                
                #all_y.append(y)
                #all_X = vstack((all_X, X.reshape(1,-1)))
                y_file.write(str(y)+"\n")
                y = numpy.array([y])
                #print("partfit - - - -", X)
                #if(not dum):
                #    predictor.partial_fit(X.reshape(1,-1),y, classes=numpy.array([-1,1]))
                #predictor.fit(X.reshape(1,-1),y)
                X_file.write(",".join([str(x) for x in X])+"\n")
                X = update_student_state(score_mode, X, sc, catix, cnts)
                runs_file.write(",".join([str(s) for s in ep])+"\n")
        
        runs_file.close()            
        X_file.close()
        y_file.close()
#         numpy.savetxt("all_X.csv", all_X, delimiter=',')
#         numpy.savetxt("all_y.csv", all_y, delimiter=',')

    all_X = numpy.genfromtxt("all_X.csv", delimiter=",")
    all_y = numpy.genfromtxt("all_y.csv", delimiter=",")

    for i in range(all_X.shape[0]):
        rowsum = numpy.sum(all_X[i,])
        print("X {} {}".format(i, rowsum))
        if numpy.isnan(rowsum):
            exit()
        
    for i in range(all_y.shape[0]):
        rowsum = numpy.sum(all_y[i,])
        print("y {} {}".format(i, rowsum))
        if numpy.isnan(rowsum):
            exit()
            


    #print("selecting balanced subsample")
    print("t t split")
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.10, random_state=666, stratify=all_y)
    
    scaler = StandardScaler()
    if do_scaling:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    if(force_balanced_classes):
        X_train, y_train = balanced_subsample(X_train, y_train, 1.0) #0.118)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    if train_clf:
        #predictor.fit(all_X, all_y)
        print("tuning classifier ...")
        for ix,p in enumerate(predictors):
            print(type(p))
            print(p.get_params().keys())
#             input("prompt")
            if optimise_predictors==True and predictor_params[ix]!=None:
                pbest = run_random_search(p, X_train, y_train, predictor_params[ix])
            else:
                pbest = p.fit(X_train, y_train)
            predictors[ix] = pbest
        
        print("pickling classifier ...")
        for ix,p in enumerate(predictors): #in predictors.items():
            with open('./pred{}.pkl'.format(ix), 'wb') as output:
                pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
        print("done!")

    if do_test:
        predictors=[] #seems a bit crappy to do this here
        for ix,fname in enumerate(glob.glob("./pred*.pkl")):
            pname = os.path.splitext(fname)[0]
            f = open(fname,"rb")
            predictors.append( pickle.load(f) )
            f.close()
        print("..done.")

        print("y_train:", y_train[y_train == 1].shape, y_train[y_train != 1].shape )
        print("y_test:", y_test[y_test == 1].shape, y_test[y_test != 1].shape)
        
        report = open(report_name,"w")
        report.write("Forced balance="+str(force_balanced_classes)+", F33, WGT="+diff_weighting+", SCMODE="+score_mode+", SCALE="+str(do_scaling)+"\n")
        for p in predictors:
            report.write(str(p)+"\n")
            report.write("TRAIN\n")
            y_pred = p.predict(X_train)
            report.write(classification_report(y_train, y_pred)+"\n")
            
            report.write("TEST\n")
            y_pred = p.predict(X_test)
            report.write(classification_report(y_test, y_pred)+"\n")
        report.close()
    exit()
    
#     if do_test:
#         if not train_clf: # i.e. if we want to load from an earlier training session
#             print("loading classifiers...")
#             for fname in glob.glob("./pred.pkl"):
#                 pname = os.path.splitext(fname)[0]
#                 f = open(fname,"rb")
#                 p = pickle.load(f)
#                 #predictors[pname]=p
#                 predictor = p
#                 f.close()
#             print("..done.")
#         
#         pas=fail = 0
#         true_ys = []
#         pred_ys = []
# 
#         all_X = numpy.zeros(shape=(0,2*len(cats)))
#         all_y = []
#         for u in users_test:
#             X = numpy.zeros(shape=2*len(cats)) #init'se a new feature vector
#             cnts = copy.copy(X)
#             X[0:len(cats)-1]=-1.0
#             #cnts = numpy.zeros(shape=len(cats))
#             attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
#             runs = extract_runs(attempts)
#             for r in runs:
#                 q = r[0]
#                 qtilda = q.replace("|","~")
#                 lev = levels[qtilda]
#                 if lev<1:
#                     continue
#                 qdiff= diffs[qtilda]
#                 sc =  qdiff* r[2] / r[1]
#                 qcat = cat_lookup[qtilda]
#                 catix = cat_ixs[qcat]
# #                 X= numpy.hstack((X, qdiff))
# #                 X= numpy.hstack((X, catix))
#                 X[len(cats)-1:]=0.0 #reset latter part of array to zero
#                 X[len(cats)-1+catix]=qdiff
#                 print("pred --- -- -- - -", X)
#                 
#                 if(dum):
#                     pred_y = random.choice([-1,1])
#                 else:
#                     pred_y = predictor.predict(X.reshape(1,-1))
#                 
#                 
#                 true_y = 1.0 if sc>0 else -1.0
#                 
#                 true_ys.append(int(true_y))
#                 pred_ys.append(int(pred_y))
#                 
#                 if true_y == 1:
#                     pas += 1
#                 else:
#                     fail += 1
# 
#                 print(int(pred_y), true_y)
#                 if int(pred_y) == true_y:
# #                     print("RIGHT!")
#                     print("*** |\ |\ |` |\ | /` ```   /\ |/  ***")
#                     print("*** |/ |/ |- |: | :   |    :: |\  ***")
#                     print("*** |  |\ |_ |/ | \_  |    \/ | \ ***")
# 
#                 else:
# #                     print("WRONG!")
#                     print("*** |) |  /| |\ ``` ***")
#                     print("*** |\ |  |+ |/  |  ***")
#                     print("*** |/ |_ || |\  |  ***")
# 
#                 #Now update the user's mental model        
#                 #sc = sc_raw * diffs[q]
#                 #cnt = cnts[catix]
#                 X = update_student_state(score_mode, X, sc, catix, cnts)
#                 #X[catix] = sc_raw # (X[catix]*cnt + sc) / (cnt+1)
#                 #cnts[catix]=cnt+1
#                 
#         cr = classification_report(true_ys, pred_ys)
#         print(cr)
#       
#         print("#+:", pas, "#-:", fail)