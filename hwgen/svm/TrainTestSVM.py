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
from sklearn.svm import SVC

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
def convert_to_Xy(assts, split=None):
    ret = []
    ct = 0
    no = len(assts)
    if(split):
        split = int(split * len(assts))
        seed(666)
        shuffle(assts)
        tt = assts[0:split]
        tr = assts[split:]
    for s in [tr,tt]:
        X = []
        y = []
        for a in s:
            ct += 1
            profiles = pickle.loads(zlib.decompress(a[5]))
            print("{}/{}/{}".format(ct, len(s), no))
            hexagons = a[3]
            # qns_list = [s for s in all_qids if any(s.startswith(hx) for hx in hexagons)]

            for psi in profiles:  # set up the training arrays here
                X.append(profiles[psi])
                y.append(hexagons)
                # y.append(qns_list)
        ret.append(X)
        ret.append(y)
    ret.append(tt) # get the testing metadata too
    return ret


# concepts_all = list(concept_extract())
numpy.set_printoptions(threshold=numpy.nan)
def train_knn_model(assts, n_macroepochs=100, n_epochs=10):
    TUNE = False
    #we start by fitting pca across the whole population (random sample)
    sgen = xy_generator(assts, batch_size=5000)
    pca = PCA(n_components=48)
    for _,X,y,_,_,_,_ in sgen:
        print("fitting PCA...")
        X = numpy.array(X, dtype=numpy.int8)
        y = numpy.array(y).ravel()
        pca.fit_transform(X)
        # if TUNE:
        #     tuned_parameters = [{'n_neighbors': [1, 20, 50, 100],
        #                          'weights': ['distance', 'uniform'],
        #                          'algorithm': ['ball_tree', 'kd_tree', 'brute']
        #                          }]
        #     scores = ['f1_macro', 'f1_micro', 'accuracy']
        #     # scores = ['accuracy']
        #     performances = []
        #     print("Tuning")
        #     for score in scores:
        #         print("# Tuning hyper-parameters for %s" % score)
        #         clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring=score, verbose=0, n_jobs=7)
        #         clf.fit(X, y)
        #         print("Best parameters set found on development set:")
        #         print(clf.best_estimator_)
        #         print("Grid scores on development set:")
        #         for params, mean_score, scores in clf.grid_scores_:
        #             print("%0.3f (+/-%0.03f) for %r"
        #                   % (mean_score, scores.std() / 2, params))
        #
        # break #half-loop just to get one sample from sgen
    exit()
    del sgen
    print("fitted")

    gc.collect()

    xygen = xy_generator(assts, batch_size=5000)  # make generator object
    clfs = []
    i = 0
    for S,X, y, yc, yt, ylv, yv in xygen:
        X = numpy.array(X, dtype=numpy.int8)
        y = numpy.array(y)
        X = pca.transform(X)
        voter = SVC()
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
    _,X,y,_,_,_,_ = next(xy_generator(tt, batch_size=-1))
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

    for tt_a in tt: # for each test assigment
        psi_list = pickle.loads(zlib.decompress(tt_a[7]))
        true_hexes = numpy.array( tt_a[3] )
        ensemble_sum = None
        for psi in psi_list:
            print(">>>>",act)
            y_pred_lab : str = y_preds[act]
            y_pred_raw = get_n_hot([y_pred_lab], all_page_ids)
            y_true = get_n_hot(true_hexes, all_page_ids)
            assert numpy.sum(y_true) == len(true_hexes)
            assert len(y_pred_raw) == len(y_true)

            if ensemble_sum is None:
                ensemble_sum = y_pred_raw
            else:
                ensemble_sum += y_pred_raw

            # done_qns=[]
            # done_ixs=[]
            #
            # for ix,v in enumerate(XP):
            #     if v==1:
            #         done_ixs.append(ix)
            #         done_qns.append(all_qids[ix].split("|")[0])

            y_pred,indices = get_top_k_hot(y_pred_raw, len(true_hexes))

            pred_hexes = [all_page_ids[i] for i in indices]
            print(pred_hexes,"\n", true_hexes,"\n-------")

            tscore = numpy.logical_and(y_pred, y_true)
            score += sum(tscore)
            tot+=len(true_hexes)
            act+=1
            print(score)
            print(tot)

        en_y,en_indices = get_top_k_hot(ensemble_sum, len(true_hexes))
        this_en_score = sum(numpy.logical_and(en_y, y_true))

        pred_hexes = [all_page_ids[i] for i in en_indices]
        print("***ENSEMBLE***")
        print(pred_hexes, "\n", true_hexes, "\n-------")
        print(this_en_score/len(true_hexes))
        print("***************")
        en_score += this_en_score
        en_tot += len(true_hexes)

    print("***IND")
    print(score/tot)
    print("***ENS")
    print(en_score/en_tot)


def evaluate_predictions(tt, model, scaler, sscaler): #, test_meta):
    fout = open(base + "hw_gobbler.out", "w")
    S,X,y,yc,yt,yl = next(xy_generator(tt, batch_size=-1))
    # X = scaler.transform(X)
    # S = sscaler.transform(S)
    # y_preds, c_preds, t_preds, l_preds = model.predict([S,X], verbose=True)
    y_preds, c_preds, t_preds, l_preds = model.predict([S,X], verbose=True)
    X = None

    numpy.set_printoptions(precision=3, suppress=True)
    for ct, tt_a in enumerate(tt): # for each test assigment
        #get the student-level profiles
        ts = tt_a[0]
        board = tt_a[1]
        grp = tt_a[2]
        psi_list = tt_a[6]
        true_hexes = tt_a[3]
        # true_cs = tt_a[4]
        # true_lvs = tt_a[5]
        # true_binsd = mlb.transform(true_hexes)

        true_topics = [cat_page_lookup[hx] for hx in true_hexes]
        true_cs = [con_page_lookup[hx] for hx in true_hexes]
        true_lvs = [lev_page_lookup[hx] for hx in true_hexes]

        ensemble_sum = None
        c_sum = None
        t_sum = None
        l_pred_list = []
        for offset, student in enumerate(psi_list):
            y_pred = y_preds[ct+offset]
            c_nh_pred = c_preds[ct+offset]
            l_pred = l_preds[ct + offset]
            t_pred = t_preds[ct + offset]

            l_pred_list.append( l_preds[ct+offset] )
            #y_pred = (y_preds[ct+offset] >= 0.5).astype(int)
            y_pred = y_preds[ct + offset]
            c_sum = c_nh_pred if c_sum is None else c_sum + c_nh_pred
            t_sum = t_pred if t_sum is None else (t_sum + t_pred)
            ensemble_sum = y_pred if ensemble_sum is None else ensemble_sum + y_pred # keep running total

            fout.write("STUDENT {}\n".format(student))
            # numpy.set_printoptions(threshold=numpy.nan)
            # fout.write("Yp{}\n".format(y_pred))
            # yixs = numpy.argsort(y_pred)
            # reversed(yixs)
            # for ix, t_hx_name in enumerate(true_hexes):
            #     hx_ix = yixs[ix]
            #     hx_name = all_page_ids[hx_ix]
            #     pred_hexes.append(hx_name)
            # fout.write("Cp{}\n".format(c_nh_pred))
            # fout.write("Tp{}\n".format(t_pred))
            # fout.write("Lp{}\n".format(l_pred))

        yixs = numpy.argsort(ensemble_sum)
        yixs = list(reversed(yixs))

        cixs = numpy.argsort(c_sum)
        cixs = list(reversed(cixs))

        tixs = numpy.argsort(t_sum)
        tixs = list(reversed(tixs))

        fout.write("Yp20:\n")
        fout.write("{}\n".format(true_hexes))
        pred_hexes = []
        for yix in yixs[0:20]:
            hx_ix = yixs[yix]
            hx_name = all_page_ids[hx_ix]
            pred_hexes.append(hx_name)
            fout.write("{} {:.3f}\n".format(hx_name, ensemble_sum[yix]))

        mixed_loss(pred_hexes, true_hexes)


        fout.write("Cp10:\n")
        fout.write("{}\n".format(true_cs))
        for cix in cixs[0:10]:
            con = concepts_all[cix]
            score = c_sum[cix]
            fout.write("{} {:.3f}\n".format(con, score))

        fout.write("Tp10:\n")
        fout.write("{}\n".format(true_topics))
        for tix in tixs[0:10]:
            top = cats[tix]
            score = t_sum[tix]
            fout.write("{} {:.3f}\n".format(top, score))
    fout.close()

def mixed_loss(pred_hexes, true_hexes):
    # yixs = numpy.argsort(ensemble_sum)
    # yixs = list(reversed(yixs))
    #get concepts for next_pgs
    volume = 0
    concepts = set()
    topics=set()
    levels=[]
    for np in pred_hexes:
        # qids = [qid for qid in all_qids if qid.startswith(np)]
        # volume += len(qids)
        volume+=1
        if np in cat_page_lookup:
            topics.add(cat_page_lookup[np])
        if np in con_page_lookup:
            concepts.update(con_page_lookup[np])
        if np in lev_page_lookup:
            levels.append(lev_page_lookup[np])

    tvolume = 0
    tconcepts = set()
    ttopics = set()
    tlevels = []
    for tp in true_hexes:
        # tqids = [qid for qid in all_qids if qid.startswith(tp)]
        # tvolume += len(tqids)
        tvolume+=1
        if tp in cat_page_lookup:
            ttopics.add(cat_page_lookup[tp])
        if tp in con_page_lookup:
            tconcepts.update(con_page_lookup[tp])
        if tp in lev_page_lookup:
            tlevels.append(lev_page_lookup[tp])

    # if len(tconcepts) == 0:
    #     return None
    print("levels: {} vs {}".format(levels, tlevels))
    print("median levels: {} vs {}".format(numpy.median(levels), numpy.median(tlevels)))
    print("volumes: {} vs {} (SQErr {})".format(volume,tvolume, (volume-tvolume)**2))
    # if 0 == len(topics) == len(ttopics):
    #     tscore=1
    # else:
    tscore= sum([1 for t in topics if t in ttopics])/len(ttopics) if len(ttopics)>0 else None
    print("{}\n\t vs t{}".format(topics, ttopics))
    print("topic score = {}".format(tscore))

    # if 0 == len(concepts) == len(tconcepts):
    #     cscore=1
    # else:
    cscore= sum([1 for c in concepts if c in tconcepts])/len(tconcepts) if len(tconcepts)>0 else None


    print("{}\n\t vs t{}".format(concepts, tconcepts))
    print("concept score = {}".format(cscore))

    hxscore=0
    for hx in pred_hexes:
        if hx in true_hexes:
            hxscore+=1
    hxscore = hxscore/len(true_hexes)
    print("{}\n\t vs t{}".format(pred_hexes, true_hexes))
    print("hex score = {}".format(hxscore))
    print("----------------------------------")
    return (numpy.median(levels), numpy.median(tlevels)), (volume,tvolume), tscore, cscore, hxscore


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
    ass_n = 10500
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