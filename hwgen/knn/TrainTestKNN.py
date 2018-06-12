import gc
import os
import pickle
import random
import zlib
from random import seed, shuffle

import numpy
import pandas
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from hwgen import hwgengen
from hwgen.common import init_objects, get_meta_data, get_n_hot, split_assts
from hwgen.concept_extract import page_to_concept_map
from hwgen.deep.TrainTestBook import xy_generator

base = "../../../isaac_data_files/"

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(n_users)

hwdf = get_meta_data()
concepts_all = set()
hwdf.index = hwdf["question_id"]
hwdf["related_concepts"] = hwdf["related_concepts"].map(str)
for concepts_raw in hwdf["related_concepts"]:
    print(concepts_raw)
    if concepts_raw != "nan":
        concepts = eval(concepts_raw)
        if concepts is not None:
            concepts_all.update(concepts)
concepts_all = list(concepts_all)


asst_fname = base+"assignments.pkl"
con_page_lookup = page_to_concept_map()

# concepts_all = list(concept_extract())
numpy.set_printoptions(threshold=numpy.nan)
def train_knn_model(assts, n_macroepochs=100, n_epochs=10):
    TUNE = False
    #Best params f1_macro
    # KNeighborsClassifier(algorithm='brute', leaf_size=30, metric='minkowski',
    #                      metric_params=None, n_jobs=1, n_neighbors=1, p=2,
    #                      weights='distance')

    #Best params f1_micro
    # NeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
    #                     metric_params=None, n_jobs=1, n_neighbors=20, p=2,
    #                     weights='uniform')

    #Best params accuracy
    # KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
    #                      metric_params=None, n_jobs=1, n_neighbors=20, p=2,
    #                      weights='uniform')

    #we start by fitting pca across the whole population (random sample)
    sgen = xy_generator(assts, batch_size=5000)
    pca = PCA(n_components=48)
    for S,X,y,assids in sgen:
        print("fitting PCA...")
        X = numpy.array(X, dtype=numpy.int8)
        y = numpy.array(y).ravel()
        # samp = numpy.random.randint(0, X.shape[0], 40000)
        # print(samp[0:10])
        # X[samp,:]
        # X=X[numpy.random.choice(X.shape[0], size=1000, replace=False), :]
        pca.fit_transform(X)
        if TUNE:
            tuned_parameters = [{'n_neighbors': [1, 20, 50, 100],
                                 'weights': ['distance', 'uniform'],
                                 'algorithm': ['ball_tree', 'kd_tree', 'brute']
                                 }]
            scores = ['f1_macro', 'f1_micro', 'accuracy']
            # scores = ['accuracy']
            performances = []
            print("Tuning")
            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring=score, verbose=0, n_jobs=7)
                clf.fit(X, y)
                print("Best parameters set found on development set:")
                print(clf.best_estimator_)
                print("Grid scores on development set:")
                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_score, scores.std() / 2, params))

        break #half-loop just to get one sample from sgen
    exit()
    del sgen
    print("fitted")

    gc.collect()

    # xygen = xy_generator(assts, batch_size=5000)  # make generator object
    xygen = hwgengen.xy_generator(assts, batch_size=5000)
    clfs = []
    i = 0
    for S,X,y,assids in xygen:
        X = numpy.array(X, dtype=numpy.int8)
        y = numpy.array(y)
        X = pca.transform(X)

        # y2 = []
        # print("binarising sh!t")
        # for yel in y:
        #     y2.append( get_n_hot(yel, all_page_ids) )
        # y = numpy.array(y2)
        # print("binarised, shape is ", y.shape)

        #Non-binary encoding
        # y2 = []
        # for yel in y:
        #     y2.append(all_page_ids.index(yel))
        # y = numpy.array(y2)
        # del y2

        # algorithm = 'kd_tree', leaf_size = 30, metric = 'minkowski',
        #                      metric_params=None, n_jobs=1, n_neighbors=20, p=2,
        #                      weights='uniform'

        voter = KNeighborsClassifier(n_neighbors=20, n_jobs=7, algorithm='kd_tree', leaf_size=30, metric='minkowski', weights='uniform')
        voter.fit(X,y)
        clfs.append(voter)
        i += 1
    model = EnsembleVoteClassifier(clfs=clfs, refit=False)

    X_for_classes = []
    y_for_classes = []
    for classlabel in all_page_ids:
        X_for_classes.append(numpy.zeros(256))
        y_for_classes.append(classlabel)

    model.fit(X_for_classes,y_for_classes)
    return model, pca, None, None #, sscaler, levscaler, volscaler


def get_top_k_hot(raw, k):
    args = list(reversed(raw.argsort()))[0:k]
    print(args)
    k = numpy.zeros(len(raw))
    k[args] = 1.0
    return k, args

def evaluate_knn_phybook_loss(tt, model, pca): #, sscaler,levscaler,volscaler): #, test_meta):
    print("ready to evaluate...")
    S,X,y,assids = next(hwgengen.xy_generator(tt, batch_size=-1))
    X = numpy.array(X, dtype=numpy.int8)
    X = pca.transform(X)
    print("predicting...")
    y_preds = model.predict(X)
    score = 0
    act = 0
    tot = 0
    print("post processing...")
    print("lenX:",len(X))
    print("len tt",len(tt))
    en_score = 0
    en_tot = 0
    assid = None
    for assid, true_y, pred_y in zip(assids,y,y_preds):
        print(true_y, pred_y)






if __name__ == "__main__":
    print("Initialising K-NN HWGen....")
    os.nice(3)

    model = None
    print("loading...")
    with open(asst_fname, 'rb') as f:
        asses = pickle.load(f)

    print("loaded {} assignments".format(len(asses)))
    print("Splitting...")

    do_train = True
    do_testing = True
    ass_n = 20500
    split = 500
    n_macroepochs = 10
    n_epochs = 10
    ass_keys = list(asses.keys())  # train on a subset of the data
    random.seed(666)
    print("first 10 keys preshuff = ", ass_keys[0:10])
    shuffle(ass_keys)
    print("first 10 keys postshuff = ", ass_keys[0:10])

    assert_profile_len = False
    if assert_profile_len:
        for k in ass_keys:
            ass = asses[k]
            students = ass[6]
            profiles = pickle.loads(zlib.decompress(ass[7]))
            assert len(students) == len(profiles)

    BOOK_ONLY = True
    print("Splitting...")
    tr, tt = split_assts(asses, ass_n, split, BOOK_ONLY)
    del asses
    gc.collect()
    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    if do_train:
        print("training")
        model, mlb, mlbc, mlbt = train_knn_model(tr, n_macroepochs, n_epochs)
        print("...deleted original X,y")
        joblib.dump(model,base + 'hwg_knn_model.hd5')
        joblib.dump((mlb, mlbc, mlbt), base + 'hwg_knn_mlb.pkl')
        print("training done")

    if do_testing:
        print("testing")
        if model is None:
            model = joblib.load(base + "hwg_knn_model.hd5")
            (mlb, mlbc, mlbt) = joblib.load(base + 'hwg_knn_mlb.pkl')
        evaluate_knn_phybook_loss(tt, model, mlb) #, sscaler,levscaler,volscaler)
        print("testing done")