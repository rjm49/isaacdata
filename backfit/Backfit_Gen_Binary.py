import multiprocessing
import os, sys
import threading
from multiprocessing.pool import Pool

from sklearn.naive_bayes import BernoulliNB, GaussianNB

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backfit.BackfitUtils import init_objects
from backfit.utils.utils import DW_STRETCH, DW_LEVEL, calc_qdiff, load_new_diffs, DW_NO_WEIGHT, DW_BINARY, DW_NATTS, \
    DW_PASSRATE, load_mcmc_diffs, DW_MCMC
from backfit.BackfitTest import train_and_test

print(sys.path)
from sklearn.metrics.classification import f1_score
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC, LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier

import pandas as pd
import numpy
from utils.utils import extract_runs_w_timestamp

QENC_QUAL=False
QENC_DIFF=False
qenc_width = 33
n_classes = 2

n_users = 1000
max_runs = None
percTest = 0.20

predictors = [
    DummyClassifier(strategy="stratified"),
    DummyClassifier(strategy="uniform"),
    BernoulliNB(),
    LinearSVC(max_iter=100, class_weight="balanced"),
    MLPClassifier(max_iter=100, nesterovs_momentum=True, early_stopping=True), #, activation="logistic"),
    LogisticRegression(class_weight='balanced'),
    GaussianNB(),
]

predictor_params = [
    None,
    None,
    {'n_iter':50, 'alpha': numpy.logspace(-3, 2) },
    {'n_iter':50,'C': numpy.logspace(-3, 2)},
    {'n_iter':250,'hidden_layer_sizes':[(100,), (66,10)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
    {'n_iter':50,'C': numpy.logspace(-3, 2)},
    None,
]


def generate_binwgt_run_files(xfn, yfn, alpha, _w, fade, cats, cat_lookup, all_qids, users, stretches, passrates, passquals, levels, mcmcdiffs, cat_ixs, n_classes):
    X_file = open(xfn,"w")
    y_file = open(yfn,"w")

    n_features = len(cats)

    print("using n_features=", n_features)

    run_ct= 0
    X = numpy.zeros(shape=n_features) #init'se a new feature vector w same width as all_X
    print("Generating files for {} users...".format(len(users)))
    for u in users:
        #print("user = ", u)
        X[:]= 0.0

        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)

        runs = extract_runs_w_timestamp(attempts)
        for run in runs:
            run_ct+=1
            ts, q, n_atts, n_pass = run
            qt = q.replace("|","~")
            lev = levels[qt]
            if lev==0:
                continue

            qdiff = calc_qdiff(qt, passrates, stretches, levels, mcmcdiffs, mode=_w)

            catix = cat_ixs[ cat_lookup[qt] ]

            qenc = numpy.zeros(shape=qenc_width)
            qenc[catix] = 1.0

            X_file.write(",".join([str(x) for x in X])+","+",".join([str(e) for e in qenc])+"\n")
            X = X * fade

            if (n_pass>0):
                if n_classes == 2:
                    y = 0
                else:
                    y = (-1 if n_atts==1 else 0)
                X[catix] =(1.0-alpha)*X[catix] + alpha*1.0
            else:
                y = 1
            y_file.write(str(y)+"\n")

        X_file.flush()
        y_file.flush()
    X_file.close()
    y_file.close()
    print(n_users, "users", run_ct,"runs", run_ct/float(n_users), "rpu")

if __name__ == '__main__':
    cmd='test'
    if len(sys.argv) < 2:
        cmd = input("command please?")
    else:
        cmd = sys.argv[1]

    if cmd.startswith('g'):
        do_test = False
    else:
        do_test = True

    n_classes = 2
    force_balanced_classes = False
    do_scaling = True
    optimise_predictors = True
    print("n_users",n_users)
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)

    passdiffs, stretches, passquals, all_qids = load_new_diffs()
    mcmc_diffs = load_mcmc_diffs()

    reports =[]
    report_name = "report_DW{}_{}_fb{}_opt{}_scale{}_{}.txt".format(0, n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), "BW33")
    if do_test:
        report = open(report_name,"w")
    jobs = []
    for w in [DW_BINARY]:
        for alpha in [0.9, 0.6, 0.3, 0.1]:
            for phi_retain in [1.0, 0.75, 0.25, 0.0]:
                xfn = "BW33_{}_{}_{}_X.csv".format(str(alpha), str(phi_retain), w)
                yfn = "BW33_{}_{}_{}_y.csv".format(str(alpha), str(phi_retain), w)
                if do_test:
                    print("testing")
                    X_train, X_test, y_pred_tr, y_pred, y_true, scaler = train_and_test(alpha, predictors, predictor_params, xfn, yfn, n_users, percTest, "BW33", w, phi_retain, force_balanced_classes, do_scaling, optimise_predictors, report=report)
                    reports.append((alpha, report_name, y_true, y_pred))
                else:
                    prms = (xfn, yfn, alpha, w, phi_retain, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, mcmc_diffs, cat_ixs, n_classes)
                    #p = threading.Thread(target= generate_binwgt_run_files, args = prms)
                    #jobs.append(p)
                    #p.start()
                    generate_binwgt_run_files(*prms)
                    print("gen complete, train files are",xfn,yfn)
    if do_test:
        report.close()
        print("complete, report file is:", report_name)

    wid = n_classes+1
    if do_test:
        retains = []
        f1s = []
        mx = numpy.ndarray(shape=(len(reports), wid))
        for ix, (alpha, r, ytr, ypd) in enumerate(reports):
            mx[ix, 0] = alpha
            f1s = f1_score(ytr, ypd, average=None)
            mx[ix, 1:wid] = f1s
        numpy.savetxt(report_name+"_to_plot.csv", mx)