import os, sys
from sklearn.metrics.classification import f1_score
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)
from backfit.Backfit_Gen_DW import generate_run_files
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm.classes import SVC, LinearSVC
from math import pi
from random import randint

from backfit.BackfitUtils import init_objects
from backfit.BackfitTest import train_and_test

import pandas as pd
import numpy
import copy
from utils.utils import ATT_QID, ATT_COR, extract_runs_w_timestamp, balanced_subsample,\
    DW_STRETCH, SCORE_MODE_REPLACE

from matplotlib import pyplot as plt
#DEFINE SOME CONSTANTS

QENC_QUAL=False
QENC_DIFF=False
qenc_width = 36

FEAT_F33 = "F33"

featureset_to_use = FEAT_F33
force_balanced_classes = True
n_users = 1000
max_runs = None #10000
percTest = 0.1

do_gen_runs = True
train_clf = True
do_test = True
optimise_predictors = True

predictors = [
            # DummyClassifier(strategy="uniform"),
#               DummyClassifier(strategy="uniform"),
              #SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
            #SVC(class_weight='balanced'),
            #SVC(kernel="linear", class_weight='balanced'),
              #PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, tol=1e-3, n_jobs=-1),
            #MLPClassifier(max_iter=1000, nesterovs_momentum=True, early_stopping=True),
              #Perceptron(max_iter=1000),
            #GaussianNB(),
            #LogisticRegression(class_weight='balanced'),
              LinearSVC(max_iter=100, class_weight="balanced"),
#             OneClassSVM()
]

predictor_params = [
                    # None,
#                     None,
#                             {'alpha': numpy.logspace(-3, 2) },
                    #{'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
                    #{'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
#                             None,
#                     {'n_iter':200,'activation':['relu'], 'hidden_layer_sizes':[(100,),(100,50),(50,),(20,),(10,)], 'alpha': numpy.logspace(-6,-1) },
                    #{'n_iter':150,'hidden_layer_sizes':[(100,), (66,10)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
#                       None
                        #None,
                        {'n_iter':50,'C': numpy.logspace(-3, 2)},
#                         None,
                    ]

# def get_qenc(catix, qdiff, stretch, qpassqual):
#     qenc = numpy.zeros(shape=qenc_width)
#     qenc[:] = 0.0 #reset question encoding
#     qenc[catix]=1.0 # set the next q category and diff
#     if QENC_DIFF:
#         qenc[-3]=qdiff
#         qenc[-1]=stretch
#     if QENC_QUAL:
#         qenc[-2]=qpassqual
#     return qenc

if __name__ == '__main__':
    do_scaling = True
    force_balanced_classes = False
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users)
    
    passdiffs = {}
    passquals = {}
    stretches = {}
    diff_df = pd.read_csv("../pass_diffs.csv", header=None)
    print(diff_df)
    for dr in diff_df.iterrows():
        data = dr[1]
        qt = data[0].replace("|","~")
        passdiffs[qt] = data[1]
        stretches[qt] = data[2]
        passquals[qt] = data[3]
    all_qids = passdiffs.keys()
    
    n_users = len(users)
    reports =[]
    for w in [DW_STRETCH]: #, DW_STRETCH, DW_LEVEL]:
        for retain in [i/20.0 for i in range(21)]:
            report_name = "report_retain{}_{}_fb{}_opt{}_scale{}_{}.txt".format(retain, n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), featureset_to_use)
            if len(sys.argv) > 1 and sys.argv[1] == "generate":
                generate_run_files(retain, featureset_to_use, w, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, cat_ixs)
            else:
                report = open(report_name,"w")

                xfn = "F33_{}_{}_X.csv".format(str(retain), w)
                yfn = "F33_{}_{}_y.csv".format(str(retain), w)
                print(cat_ixs)
                X_train, X_test, y_pred_tr, y_pred, y_true, scaler = train_and_test(retain, predictors, predictor_params, xfn, yfn, n_users, percTest, featureset_to_use, w, force_balanced_classes, do_scaling, optimise_predictors, report=report)
                reports.append((retain, report_name, y_true, y_pred))
                report.close()

    print("complete, report file is:", report_name)    
    print(len(cats))
    
    retains = []
    f1s = []
    mx = numpy.ndarray(shape=(len(reports),3))
    for ix, (retain, r, ytr, ypd) in enumerate(reports):
        mx[ix,0]=retain
        f1s = f1_score(ytr, ypd, average=None)
        mx[ix,1:4]=f1s
    
    numpy.savetxt("retains_to_plot.csv", mx)
