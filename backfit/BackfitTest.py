import os
import pickle
from sys import stderr
import sys

import numpy
from sklearn.feature_selection.univariate_selection import SelectKBest

from backfit.clf_tuning import run_random_search

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.metrics.classification import classification_report, accuracy_score, f1_score, recall_score, \
    precision_score
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing.data import StandardScaler

from utils.utils import balanced_subsample


def train_and_test(retain, predictors, predictor_params, x_filename, y_filename, n_users, percTest, featureset_to_use, diff_weighting, force_balanced_classes, do_scaling, optimise_predictors, report):
    
    all_X = numpy.genfromtxt(x_filename, delimiter=",")
#     yf = open("all_y.csv", 'r').readlines()
#     all_y = numpy.asarray([numpy.array(s.split(",")).astype('float64') for s in yf], dtype=numpy.float64)
    all_y = numpy.genfromtxt(y_filename, delimiter=",")

    
    for i in range(all_X.shape[0]):
        rowsum = numpy.sum(all_X[i,])
#         print("X {} {}".format(i, rowsum))
        if numpy.isnan(rowsum):
            print(x_filename)
            exit()
        
    for i in range(all_y.shape[0]):
        rowsum = numpy.sum(all_y[i,])
#         print("y {} {}".format(i, rowsum))
        if numpy.isnan(rowsum):
            print(y_filename)
            exit()

    #print("selecting balanced subsample")
    print("t t split")
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=percTest, random_state=666)
    
    
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    if do_scaling:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

#     kbest = SelectKBest()
#     X_train = kbest.fit_transform(X_train, y_train)
#     X_test = kbest.transform(X_test)
#     print(kbest.get_support(indices=True))    
#     print(X_train.shape)
    
    if(force_balanced_classes):
        X_train, y_train = balanced_subsample(X_train, y_train, 1.0) #0.118)

#     X_train = X_train[y_train==1] #only keep in-ZPD classes
#     y_train = y_train[y_train==1]
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

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


    report.write("* ** *** |\| \` | |  |) /; `|` / |_| *** ** *\n")
    report.write("* ** *** | | /_ |^|  |) ||  |  \ | | *** ** *\n")
    report.write("runs="+str(all_X.shape[0])+"\n")

#     pca = PCA(n_components=3)
#     pca.fit(X_train)

    for p in predictors:
        report.write("------Forced balance="+str(force_balanced_classes)+", F33, WGT="+diff_weighting+", RETAIN="+str(retain)+" SCALE="+str(do_scaling)+"\n")
        report.write(str(p)+"\n")
#         report.write("TRAIN\n")
        y_pred_tr = p.predict(X_train)
#         report.write(classification_report(y_train, y_pred_tr)+"\n")
        
        report.write("TEST\n")
        y_pred = p.predict(X_test)
        report.write(classification_report(y_true=y_test, y_pred=y_pred)+"\n")

        report.write( str(precision_score(y_test, y_pred, average=None))+"\n")
        report.write( str(recall_score(y_test, y_pred, average=None))+"\n")
        report.write( str(f1_score(y_test, y_pred, average=None))+"\n")

        # for c in set(y_pred): #test accy for each class
        #     acc = accuracy_score(y_test[y_test==c], y_pred[y_test==c])
        #     report.write("{} : {}\n".format(c,acc))

        overall = accuracy_score(y_test, y_pred)
        report.write("Acc {}\n".format(overall))

        report.write("------END OF CLASSIFIER------\n")
        report.flush()

    return X_train, X_test, y_pred_tr, y_pred, y_test, scaler
