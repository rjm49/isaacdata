import pandas as pd
from pandas.core.algorithms import unique
from sklearn.svm.classes import LinearSVC
import sklearn
import numpy
import random
from numpy import nditer, ndarray
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import pickle
import math
from pandas.core.series import Series

if __name__ == '__main__':
    qmeta = pd.read_csv("../qmeta.csv")
    #print(qmeta)
    users = open("../users.csv").read().splitlines()
#     print(users)
#     exit()
    
    predictors = {}
    cats= []
    diffs = {}

    #fields for qmeta
    QID=1
    LEV=1
    SUB=3
    FLD=4
    TOP=5
    DIF=8
    
    cat_lookup = {}
    cat_ixs = {}
    for r in qmeta.itertuples():
        print("R--",r[3],r[4],r[5])
        q_id = r[QID]
        cat = str(r[SUB])+"/"+str(r[FLD])+"/"+str(r[TOP])
        cats.append(cat)
        cat_lookup[q_id]= cat
        diff_raw = r[DIF]
        diffs[q_id] = -1 if (diff_raw == float("inf")) else diff_raw
        print(r[DIF])
        predictors[q_id] = SGDClassifier()
   
    #replace any negative (formerlly inf) values with the max system difficulty
    max_diff = max(diffs.values())
    for k, v in diffs.items():
            if v < 0:
                diffs[k] = max_diff
   
    cats = unique(cats)
    for ix, c in enumerate(cats):
        cat_ixs[c]=ix
    df = pd.DataFrame(index=users, columns=cats)
    counts = pd.DataFrame(index=users, columns=cats)
    #print(df)
    
    #print(users)
    #random.shuffle(users)
    #print(users)
    ix = int( 0.1*len(users))
    users_test = users[0:ix]
    users_train = users[ix:]

    #2016-06-01 17:08:22.599,False,a_diode_divider|75b49dfb-5934-496f-bed6-d2a2d665485e
    ATT_TIM=0
    ATT_COR=1
    ATT_QID=2
    for u in users_train:
        print("user = ", u)
        X = numpy.zeros(shape=len(cats)) #init'se a new feature vector
        cnts = numpy.zeros(shape=len(cats))
        #load user episode file
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)
#         print(u)
#         print(attempts)
        qids = attempts.iloc[:,ATT_QID].unique() #TODO check that this preserves order
        
        print("qids=\n", qids)
        print(type(qids), qids.shape)
        
        
        for q in list(qids):
            print(X)
            print("q=",q)
            
            qatts = attempts[attempts.iloc[:,ATT_QID]==q]
            print(">>>qatts",qatts)
            
            q = q.replace("|","~")
            cat = cat_lookup[q]
            catix= cat_ixs[cat]

            attempt_outcomes = qatts[ATT_COR]
            if( attempt_outcomes.isnull().values.any() ):
                continue #skip onto next iteration if NaN present in this column - not a real q attempt
            
            if(True not in attempt_outcomes):
                y= numpy.array([-1])
            else:
                y= numpy.array([1])
                #student did not pass Question
            predictors[q].partial_fit(X.reshape(1,-1),y, classes=numpy.array([-1,1]))



            print("- - - qatts:",attempt_outcomes, attempt_outcomes.size)
            #Now update the user's mental model        
            
            cor_atts = numpy.sum(attempt_outcomes)
            if(cor_atts==0):
                sc_raw = 0
            else:
                print("CORR!",cor_atts)
                sc_raw = cor_atts/numpy.float64(attempt_outcomes.size)
            
#             print(cor_atts, sc_raw)
                    
            sc = sc_raw * diffs[q]
            cnt = cnts[catix]
            newval = (X[catix]*cnt + sc) / (cnt+1)
            print("## # #",sc_raw, diffs[q],sc,cnt,newval)
            X[catix] = newval
            cnts[catix]=cnt+1
    
    for p,clf in predictors.items():
        with open('./{}.pkl'.format(p), 'wb') as output:
            pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
    
    exit()
    
    fp=tp=fn=tn =0
    for u in users_test:
        X = numpy.zeros(shape=len(cats)) #init'se a new feature vector
        cnts = numpy.zeros(shape=len(cats))
        attempts = pd.read_csv("../by_user/{}.txt".format(u))
        qids = attempts.iloc[:,ATT_QID].unique() #TODO check that this preserves order
        for q in qids:
            qatts = attempts[attempts.iloc[:,ATT_QID]==q]
            print(">>>qatts",qatts)
            
            q = q.replace("|","~")
            cat = cat_lookup[q]
            catix= cat_ixs[cat]

            pred_y = predictors[q].predict(X.reshape(1,-1))
            true_y = 1 if True in qatts[ATT_COR] else -1
            if pred_y == true_y:
                if true_y == 1:
                    tp+=1
                else:
                    tn+=1
            else:
                if true_y == 1:
                    fn+=1
                else:
                    fp+=1
            #Now update the user's mental model        
            cor_atts = numpy.sum(qatts[ATT_COR])
            sc_raw = cor_atts/len(qatts[ATT_COR])
            sc = sc_raw * diffs[q]
            cnt = cnts[catix]
            X[catix] = (X[catix]*cnt + sc) / (cnt+1)
            cnts[catix]=cnt+1
            
    print(tn,tp,fn,fp)