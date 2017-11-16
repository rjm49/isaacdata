import os, sys

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

FEAT_F33 = "F33"

n_users = 1000
max_runs = None #10000
percTest = 0.1

predictors = [
#            DummyClassifier(strategy="stratified"),
#              DummyClassifier(strategy="uniform"),
              #SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
            # SVC(max_iter=10000, class_weight='balanced'),
            #SVC(max_iter=10000, kernel="linear", class_weight='balanced'),
              LinearSVC(max_iter=100, class_weight="balanced"),
              # PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, tol=1e-3, n_jobs=-1),
             MLPClassifier(max_iter=100, nesterovs_momentum=True, early_stopping=True), #, activation="logistic"),
              #Perceptron(max_iter=1000),
            #GaussianNB(),
            LogisticRegression(class_weight='balanced'),
#             OneClassSVM()numpy
]

predictor_params = [
                    # None,
                    # None,
#                             {'alpha': numpy.logspace(-3, 2) },
#                     {'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
#                     {'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
                     {'n_iter':50,'C': numpy.logspace(-3, 2)},
#                             None,
#                     {'n_iter':200,'activation':['relu'], 'hidden_layer_sizes':[(100,),(100,50),(50,),(20,),(10,)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,-1) },
                    {'n_iter':100,'hidden_layer_sizes':[(100,), (66,10)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
                      # None,
#                         None,
                        {'n_iter':50,'C': numpy.logspace(-3, 2)},
                        #  None,
                    ]

def get_qenc(catix, passrate, stretch, lev, mcmc, mode=None):
    qenc = numpy.zeros(shape=qenc_width)
    #qenc[:] = 0.0 #reset question encoding
    weight = 1.0
    if mode== DW_NATTS or mode==DW_STRETCH:
        weight = stretch
    elif mode==DW_PASSRATE:
        weight = passrate
    elif mode==DW_LEVEL:
        weight = lev
    elif mode==DW_MCMC:
        weight = mcmc
    qenc[catix]=weight # set the next q category and diff
    return qenc

def generate_run_files(alpha, _featureset_to_use, _w, fade, cats, cat_lookup, all_qids, users, stretches, passrates, passquals, levels, mcmcdiffs, cat_ixs):
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

            qenc = get_qenc(catix, passrate, stretch, lev, mcmc, mode=_w)
            X_file.write(",".join([str(x) for x in X])+","+",".join([str(e) for e in qenc])+"\n")
            X = X * fade

            upd = 1.0
            if _w == DW_BINARY:
                upd = 1.0
            elif _w == DW_NATTS:
                upd = n_atts
            elif _w == DW_NO_WEIGHT:
                upd = 1.0 / n_atts
            elif _w == DW_PASSRATE:
                upd = passrate / n_atts
            elif _w == DW_STRETCH:
                upd = stretch / n_atts
            elif _w == DW_MCMC:
                upd = mcmc / n_atts
            elif _w == DW_LEVEL:
                upd = lev / n_atts

            if (n_pass>0):
                if n_classes==2:
                    y = 0
                else:
                    y = (-1 if n_atts==1 else 0)
                X[catix] = 1 #(1.0-alpha)*X[catix] + alpha*upd
            else:
                y = 1
                #X[catix] = 0
                #X[catix] = retain*X[catix] -(1-retain)*upd

                #print("in",catix,"put diff",(qdiff/n_atts))
                # if retain < 0:
                # else:
                #     X[catix] += qdiff / n_atts
            # else:
            #     X[catix] = - mcmc / n_atts

            y_file.write(str(y)+"\n")
        #             print("did run")
        X_file.flush()
        y_file.flush()
    X_file.close()
    y_file.close()
    print(n_users, "users", run_ct,"runs", run_ct/float(n_users), "rpu")
    return x_filename,y_filename


if __name__ == '__main__':
    featureset_to_use=FEAT_F33
    cmd='test'
    if len(sys.argv) < 2:
        cmd = input("command please?")
    else:
        cmd = sys.argv[1]

    if cmd.startswith('g'):
        do_test = False
    else:
        do_test = True

    do_scaling = True
    force_balanced_classes = True
    optimise_predictors = True
    print("n_users",n_users)
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users, seed=666)

    #users = open("../mcmc/mcmc_uesrs.txt").read().splitlines()

    passdiffs, stretches, passquals, all_qids = load_new_diffs()
    mcmcdiffs = load_mcmc_diffs()

    reports =[]
    report_name = "report_DW{}_{}_fb{}_opt{}_scale{}_{}.txt".format(0, n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), featureset_to_use)
    if do_test:
        report = open(report_name,"w")
    for w in [DW_BINARY]: #, DW_NO_WEIGHT, DW_NATTS]: #, DW_LEVEL, DW_PASSRATE, DW_MCMC, DW_STRETCH]:
        # for alpha in [0.0, 0.33, 0.67, 1.0]:
        #     for fade in [0.0, 0.33, 0.5, 0.67, 1.0]:
        for alpha in [1.0]:
            for phi_retain in [1.0]:
                print(cat_ixs)

                if do_test:
                    print("testing")
                    xfn = "F33_{}_{}_{}_X.csv".format(str(alpha), str(phi_retain), w)
                    yfn = "F33_{}_{}_{}_y.csv".format(str(alpha), str(phi_retain), w)
                    X_train, X_test, y_pred_tr, y_pred, y_true, scaler = train_and_test(alpha, predictors, predictor_params, xfn, yfn, n_users, percTest, featureset_to_use, w, phi_retain, force_balanced_classes, do_scaling, optimise_predictors, report=report)
                    #reports.append((alpha, report_name, y_true, y_pred))
                else:
                    xfn, yfn = generate_run_files(alpha, featureset_to_use, w, phi_retain, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, mcmcdiffs, cat_ixs)
                    print("gen complete, train files are",xfn,yfn)
                    #reports.append((0, report_name, y_true, y_pred))

    if do_test:
        report.close()
        print("complete, report file is:", report_name)


    # wid = n_classes+1
    # if do_test:
    #     retains = []
    #     f1s = []
    #     mx = numpy.ndarray(shape=(len(reports), wid))
    #     for ix, (alpha, r, ytr, ypd) in enumerate(reports):
    #         mx[ix, 0] = alpha
    #         f1s = f1_score(ytr, ypd, average=None)
    #         mx[ix, 1:wid] = f1s
    #     numpy.savetxt(report_name+"_to_plot.csv", mx)