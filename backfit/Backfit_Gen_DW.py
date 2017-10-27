import os, sys
from BackfitUtils import init_objects
from BackfitTest import train_and_test
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)
from sklearn.metrics.classification import f1_score
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC, LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.dummy import DummyClassifier



import pandas as pd
import numpy
import copy
from utils.utils import ATT_QID, ATT_COR, extract_runs_w_timestamp, balanced_subsample,\
    calc_qdiff, DW_LEVEL, DW_NO_WEIGHT, SCORE_MODE_REPLACE, DW_STRETCH,\
    load_new_diffs, SCORE_MODE_DECAY

from matplotlib import pyplot as plt
#DEFINE SOME CONSTANTS

QENC_QUAL=False
QENC_DIFF=False
qenc_width = 33

FEAT_F33 = "F33"

featureset_to_use = FEAT_F33
force_balanced_classes = True
n_users = 1000
max_runs = None #10000
percTest = 0.1

predictors = [
#            DummyClassifier(strategy="stratified"),
#              DummyClassifier(strategy="uniform"),
              #SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=-1, class_weight="balanced"),
#             SVC(max_iter=100, class_weight='balanced'),
            #SVC(max_iter=10000, kernel="linear", class_weight='balanced'),
              #LinearSVC(max_iter=100, class_weight="balanced"),
              #PassiveAggressiveClassifier(class_weight="balanced", max_iter=1000, tol=1e-3, n_jobs=-1),
             MLPClassifier(max_iter=500, nesterovs_momentum=True, early_stopping=True),
              #Perceptron(max_iter=1000),
            #GaussianNB(),
            # LogisticRegression(class_weight='balanced'),
#             OneClassSVM()numpy
]

predictor_params = [
                    None,
                    #None,
#                             {'alpha': numpy.logspace(-3, 2) },
#                     {'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
#                     {'n_iter':50,'C': numpy.logspace(-3, 2), 'gamma': numpy.logspace(-3, 2)},
#                      {'n_iter':50,'C': numpy.logspace(-3, 2)},
#                             None,
#                     {'n_iter':200,'activation':['relu'], 'hidden_layer_sizes':[(100,),(100,50),(50,),(20,),(10,)], 'alpha': numpy.logspace(-6,-1) },
#                     {'n_iter':150,'hidden_layer_sizes':[(100,), (66,10)], 'learning_rate_init':[0.001, 0.01, 0.1], 'alpha': numpy.logspace(-6,2) },
#                       None
                        #None,
                        # {'n_iter':50,'C': numpy.logspace(-3, 2)},
                        #  None,
                    ]

def get_qenc(catix, qdiff, stretch, lev, qpassqual, mode=None):
    qenc = numpy.zeros(shape=qenc_width)
    #qenc[:] = 0.0 #reset question encoding
    weight = (stretch if mode==DW_STRETCH else (lev if mode==DW_LEVEL else 1.0))
    qenc[catix]=weight # set the next q category and diff    
    return qenc
    
def generate_run_files(retain, _featureset_to_use, _w, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, cat_ixs):
    stem = _featureset_to_use+"_"+str(retain)+"_"+_w
    x_filename= stem+"_X.csv"
    y_filename= stem+"_y.csv"
    
    runs_file = open("runs.csv", "w")
    X_file = open(stem+"_X.csv","w")
    y_file = open(stem+"_y.csv","w")

    n_features = len(cats)
#     all_X = numpy.zeros(shape=(0,n_features))

    print("using n_features=", n_features)
    
    run_ct= 0
    X = numpy.zeros(shape=n_features) #init'se a new feature vector w same width as all_X
    for u in users:
        print("user = ", u)
        X[:]= 0.0
        
        attempts = pd.read_csv("../by_user/{}.txt".format(u), header=None)

        runs = extract_runs_w_timestamp(attempts)
        
        print("run_ct",run_ct)
        for run in runs:
            run_ct+=1
            ts, q, n_atts, n_pass = run
            qt = q.replace("|","~")
            lev = levels[qt]
#             if lev<1:
#                 continue
            
            qdiff = calc_qdiff(qt, stretches, levels, mode=_w)

            y = 2
            
            catix = cat_ixs[ cat_lookup[qt] ]

            if retain >=0:
                X[X>0] = X[X>0] * retain
                
            if (n_pass>0):
                X[catix] = qdiff / n_atts
                y = (1 if n_atts>1 else 0)
                #print("in",catix,"put diff",(qdiff/n_atts))                
                # if retain < 0:
                # else:
                #     X[catix] += qdiff / n_atts
#             else:
#                 X[catix] = 0.0
            
            qpassqual = passquals[qt]
            stretch = stretches[qt]
            qenc = get_qenc(catix, qdiff, stretch, lev, qpassqual, mode=_w)
            
            y_file.write(str(y)+"\n")
            X_file.write(",".join([str(x) for x in X])+","+",".join([str(e) for e in qenc])+"\n")
#             print("did run")
        X_file.flush()
        y_file.flush()                    
    runs_file.close()
    X_file.close()
    y_file.close()
    return x_filename,y_filename

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'generate':
        do_test = False
    else:
        do_test = True
    do_scaling = True
    force_balanced_classes = True
    optimise_predictors = True
    cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(n_users)
    
    passdiffs, stretches, passquals, all_qids = load_new_diffs()

    n_users = len(users)
    reports =[]
    report_name = "report_DW{}_{}_fb{}_opt{}_scale{}_{}.txt".format(0, n_users, str(1 if force_balanced_classes else 0), ("001" if optimise_predictors else "0"), ("1" if do_scaling else "0"), featureset_to_use)
    if do_test:
        report = open(report_name,"w")
    for w in [DW_STRETCH, DW_LEVEL, DW_NO_WEIGHT]:
        for retein in [0.5, 1 ]: #, SCORE_MODE_ACCUM]
            print(cat_ixs)
            
            if do_test:
                xfn = "F33_{}_{}_X.csv".format(str(retein),w)
                yfn = "F33_{}_{}_y.csv".format(str(retein),w)
                X_train, X_test, y_pred_tr, y_pred, y_true, scaler = train_and_test(retein, predictors, predictor_params, xfn, yfn, n_users, percTest, featureset_to_use, w, force_balanced_classes, do_scaling, optimise_predictors, report=report)
            else:
                xfn, yfn = generate_run_files(retein, featureset_to_use, w, cats, cat_lookup, all_qids, users, stretches, passdiffs, passquals, levels, cat_ixs)
            #reports.append((0, report_name, y_true, y_pred))
    if do_test:
        report.close()

    print("complete, report file is:", report_name) 
