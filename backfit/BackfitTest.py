import os
import pickle
from sys import stderr
import sys

import numpy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.utils import resample

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backfit.clf_tuning import run_random_search

from sklearn.metrics.classification import classification_report, accuracy_score, f1_score, recall_score, \
    precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing.data import StandardScaler

from utils.utils import balanced_subsample, balanced_supersample


def train_and_test(alpha, predictors, predictor_params, x_filename, y_filename, n_users, percTest, featureset_to_use, diff_weighting, phi, force_balanced_classes, do_scaling, optimise_predictors, report, conf_report=None):
    all_X = numpy.loadtxt(x_filename, delimiter=",")
    all_y = numpy.loadtxt(y_filename, delimiter=",")

    print("loaded X and y files", x_filename, y_filename)

    if numpy.isnan(all_X.any()):
        print("nan in", x_filename)
        exit()

    if numpy.isnan(all_y.any()):
        print("nan in", y_filename)
        exit()

    #print("selecting balanced subsample")
    print("t t split")
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=percTest, random_state=666)

    # feature extraction
    # test = SelectKBest(score_func=chi2, k=100)
    # kb = test.fit(X_train, y_train)
    # # summarize scores
    # numpy.set_printoptions(precision=3)
    # print(kb.scores_)
    # features = kb.transform(X_train)
    # mask = kb.get_support()
    # # summarize selected features
    # print(features.shape)
    # X_train = X_train[:,mask]
    # X_test = X_test[:,mask]


    scaler = StandardScaler()
    if do_scaling:
        # input(X_train.shape)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        with open('../../../isaac_data_files/qutor_scaler.pkl', 'wb') as output:
            pickle.dump(scaler, output, pickle.HIGHEST_PROTOCOL)

    # print("feature reduction...")
    # pc = PCA(n_components=100)
    # X_train = pc.fit_transform(X_train)
    # X_test = pc.transform(X_test)

    classes = numpy.unique(y_train)
    sample_weights = None
    if(force_balanced_classes):
        X_train, y_train = balanced_subsample(X_train, y_train, 1.0) #0.118)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("tuning classifier ...")
    for ix,p in enumerate(predictors):
        print(type(p))
        print(p.get_params().keys())

        if optimise_predictors==True and len(predictor_params[ix])>1:
            pbest = run_random_search(p, X_train, y_train, predictor_params[ix])
        else:
            pbest = p.fit(X_train, y_train)
        predictors[ix] = pbest

    print("pickling classifier ...")
    for ix,p in enumerate(predictors):
        p_name = predictor_params[ix]['name']
        with open('../../../isaac_data_files/p_{}_{}_{}.pkl'.format(p_name, alpha, phi), 'wb') as output:
            pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
    print("done!")

    # report.write("* ** *** |\| \` | |  |) /; `|` / |_| *** ** *\n")
    # report.write("* ** *** | | /_ |^|  |) ||  |  \ | | *** ** *\n")
    #report.write("RUNS,P,FB,WGT,ALPHA,PHI,SCL,0p,0r,0F,0supp,1p,1r,1F,1supp,avg_p,avg_r,avg_F,#samples\n")
    for ix,p in enumerate(predictors):

        report.write(",".join(map(str, (all_X.shape[0], str(p).replace(",",";").replace("\n",""), force_balanced_classes, diff_weighting, alpha, phi, do_scaling))))

        y_pred_tr = p.predict(X_train)
        y_pred = p.predict(X_test)

        # for x,y,yp in zip(X_train, y_test, y_pred):

        if conf_report:
            conf_report.write(str(p).replace(",",";").replace("\n","")+"\n")
            conf_report.write(str(alpha) +","+ str(phi)+"\n")
            conf_report.write(str(confusion_matrix(y_test,y_pred))+"\n")
            conf_report.write("\n")
        # p = precision_score(y_test, y_pred, average=None, labels=classes)
        # r = recall_score(y_test, y_pred, average=None, labels=classes)
        # F = f1_score(y_test, y_pred, average=None, labels=classes)
        p,r,F,s = precision_recall_fscore_support(y_test, y_pred, labels=classes, average=None, warn_for=('precision', 'recall', 'f-score'))
        avp, avr, avF, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, average='weighted',
                                                           warn_for=('precision', 'recall', 'f-score'))
        for ix,c in enumerate(classes):
            report.write(",{},{},{},{},{},".format(c,p[ix],r[ix],F[ix],s[ix]))
        report.write("{},{},{},{}\n".format(avp, avr, avF, numpy.sum(s)))

        # report.write(classification_report(y_test, y_pred)+"\n")
        # report.write("------END OF CLASSIFIER------\n")
        report.flush()
    return X_train, X_test, y_pred_tr, y_pred, y_test, scaler
