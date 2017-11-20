import os
import pickle
from sys import stderr
import sys

import numpy
from sklearn.feature_selection.univariate_selection import SelectKBest

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backfit.clf_tuning import run_random_search

from sklearn.metrics.classification import classification_report, accuracy_score, f1_score, recall_score, \
    precision_score, precision_recall_fscore_support
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing.data import StandardScaler

from utils.utils import balanced_subsample


def train_and_test(alpha, predictors, predictor_params, x_filename, y_filename, n_users, percTest, featureset_to_use, diff_weighting, fade, force_balanced_classes, do_scaling, optimise_predictors, report):
    all_X = numpy.loadtxt(x_filename, delimiter=",")
    all_y = numpy.loadtxt(y_filename, delimiter=",")

    print("loaded X and y files")
    
    if numpy.isnan(all_X.any()):
        print("nan in", x_filename)
        exit()

    if numpy.isnan(all_y.any()):
        print("nan in", y_filename)
        exit()

    #print("selecting balanced subsample")
    print("t t split")
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=percTest, random_state=666)
    
    
    scaler = StandardScaler()
    if do_scaling:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    classes = numpy.unique(y_train)
    if(force_balanced_classes):
        X_train, y_train = balanced_subsample(X_train, y_train, 1.0) #0.118)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("tuning classifier ...")
    for ix,p in enumerate(predictors):
        print(type(p))
        print(p.get_params().keys())
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


    report.write("* ** *** |\| \` | |  |) /; `|` / |_| *** ** *\n")
    report.write("* ** *** | | /_ |^|  |) ||  |  \ | | *** ** *\n")
    report.write("runs="+str(all_X.shape[0])+"\n")

    for ix,p in enumerate(predictors):
        report.write(str(p).replace(",",";")+"\n")

        report.write("FB,WGT,ALPHA,PHI,SCL\n")
        report.write(",".join(map(str, (force_balanced_classes, diff_weighting, alpha, fade, do_scaling))) + "\n")

        y_pred_tr = p.predict(X_train)
        y_pred = p.predict(X_test)

        # p = precision_score(y_test, y_pred, average=None, labels=classes)
        # r = recall_score(y_test, y_pred, average=None, labels=classes)
        # F = f1_score(y_test, y_pred, average=None, labels=classes)
        p,r,F,s = precision_recall_fscore_support(y_test, y_pred, labels=classes, average=None, warn_for=('precision', 'recall', 'f-score'))
        avp, avr, avF, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, average='weighted',
                                                     warn_for=('precision', 'recall', 'f-score'))
        for ix,c in enumerate(classes):
            report.write("{},{},{},{},{}\n".format(c,p[ix],r[ix],F[ix],s[ix]))
        report.write("avg,{},{},{},{}\n".format(avp, avr, avF, numpy.sum(s)))

        # report.write(classification_report(y_test, y_pred)+"\n")
        # report.write("------END OF CLASSIFIER------\n")
        report.flush()

    return X_train, X_test, y_pred_tr, y_pred, y_test, scaler
