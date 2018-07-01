import gc
import math
import os
import pickle
import random
import zlib
from collections import Counter, defaultdict
from copy import copy
from pickletools import float8
from random import seed, shuffle
from statistics import mean
from typing import Union

import keras
import numpy
import pandas
import pylab
from keras import callbacks, Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, concatenate, LSTM, Embedding
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model, Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelBinarizer
from sklearn.utils import compute_class_weight

from hwgen.common import init_objects, get_meta_data, get_n_hot, split_assts, get_page_concepts, jaccard, \
    get_all_assignments, get_student_list, make_gb_question_map, get_user_data
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.hwgengen2 import hwgengen2, gen_qhist, build_oa_cache, build_dob_cache
from hwgen.profiler import get_attempts_from_db

from matplotlib import pyplot as plt, pyplot

import tracemalloc

base = "../../../isaac_data_files/"

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(
    n_users)

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

asst_fname = base + "assignments.pkl"
con_page_lookup = page_to_concept_map()


def cluster_and_print(assts):
    xygen = hwgengen2(assts, batch_size=-1, FRESSSH=False)  # make generator object
    for S, X, U, y, ai, awgt in xygen:
        y_labs = numpy.array(y)
        if (X == []):
            continue
        S = numpy.array(S)  # strings (labels)
        X = numpy.array(X)  # floats (fade)
        U = numpy.array(U)  # signed ints (-1,0,1)
        print("S", S.shape)
        print("X", X.shape)
        print("U", U.shape)
        assert y_labs.shape[1] == 1  # each line shd have just one hex assignment

        # c_labs = [concept_map[label[0]] for label in y_labs]
        # c = clb.transform(c_labs)
        # print(c_labs[0:10])
        # print(c[0:10])

        # try:
        #     y = ylb.transform(y_labs) / awgt
        # except:
        #     y = ylb.fit_transform(y_labs) / awgt
        # print(y_labs)
        # input(y)
        # input(awgt)
        # assert numpy.sum(y[0]) == 1 # Each line should be one-hot
        n = 5000
        lab_set = list(numpy.unique(y_labs))
        colors = numpy.array([lab_set.index(l) for l in y_labs])[0:n]

        # calc_entropies(X,y_labs)
        # exit()

        # pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)
        # converted = pca.fit_transform(X) # convert experience matrix to points
        converted = tsne.fit_transform(X[0:n])

        plt.scatter(x=converted[:, 0], y=converted[:, 1], c=colors, cmap=pylab.cm.cool)
        plt.show()


def calc_entropies(X, y):
    d = defaultdict(list)
    for x, lab in zip(X, y):
        d[str(lab)].append(x)
    for l in d:
        # print("calc for {}, len {}".format(l,len(d[l])))
        H = entropy(d[l])
        print("{}\t{}\t{}".format(l, H, len(d[l])))


def entropy(lizt):
    "Calculates the Shannon entropy of a list"
    # get probability of chars in string
    lizt = [tuple(e) for e in lizt]
    prob = [float(lizt.count(entry)) / len(lizt) for entry in lizt]
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy


def make_phybook_model(n_S, n_X, n_U, n_A, n_P):
    n_Voc = 10000
    n_Emb = 32
    # this is our input placeholder
    # input_S = Input(shape=(n_S,), name="s_input")
    # input_X = Input(shape=(n_X,), name="x_input")
    # input_U = Input(shape=(n_U,), name="u_input")

    # iw = (n_S + n_X + n_U + n_A)
    iw = (n_S + n_A)
    input = Input(shape=(iw,), name="input")

    # e = Embedding(n_Voc, n_Emb, input_length=20)
    # lstm1 = LSTM(256, input_shape=(20,))

    # w = 1024 # best was 256
    # w = (n_S + n_X + n_P)//2
    w = 1024

    # i_X = Dense(200, activation='relu')(input_X)
    # i_U = Dense(200, activation='relu')(input_U)

    hidden = Dense(w, activation='relu')(input)  # (concatenate([input_S, input_X, input_U]))
    # hidden = Dropout(.5)(hidden)
    # hidden = Dense((w + n_P) // 2, activation='relu')(hidden)
    # hidden = Dropout(.5)(hidden)

    next_pg = Dense(n_P, activation='softmax', name="next_pg")(hidden)

    # m = Model(inputs=input_X, outputs=[next_pg,next_concept] )
    # m.compile(optimizer="adam", loss=['categorical_crossentropy','binary_crossentropy'])

    o = Adam()
    # o = RMSprop(lr=0.05)

    m = Model(inputs=input, outputs=next_pg)
    m.compile(optimizer=o, loss='categorical_crossentropy', metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    m.summary()
    return m


gb_qmap = make_gb_question_map()

numpy.set_printoptions(threshold=numpy.nan)


def train_deep_model(tr, sxua, n_macroepochs=100, n_epochs=10):
    model = None

    concept_list = list(set().union(*concept_map.values()))
    print(concept_list)

    yship = []
    qlist = pid_override
    hex_counter = Counter()
    tot = 0

    for i, ass in enumerate(tr.iterrows()):
        gb_id = ass[1]["gameboard_id"]
        gr_id = ass[1]["group_id"]

        hexagons = [gb_qmap[gb_id][0]]
        students = get_student_list(gr_id)
        for psi in students:
            for hx in hexagons:
                if hx not in qlist:
                    print(hx, " not in qlist")
                    qlist.append(hx)
                yship.append(hx)
                hex_counter[hx] += 1
                tot += 1

    # yship = list(concept_map.keys()) +yship
    ylb = LabelBinarizer()  # (classes=qlist)
    qlist = numpy.unique(yship)
    ylb.fit(qlist)
    # ylb.classes_ = yship  # start by fitting the binariser to the shortlist of book qns

    for hx in hex_counter.most_common():
        print(hx[0], hx[1])
    print(tot)

    # print(qlist)
    # print(ylb.classes_)
    # assert len(list(qlist)) == len(list(ylb.classes_))
    # assert list(qlist) == list(ylb.classes_)

    weights = {}
    class_wgt = compute_class_weight('balanced', ylb.classes_, yship)
    for clix, (cls, wgt) in enumerate(zip(ylb.classes_, class_wgt)):
        print(clix, cls, wgt)
        weights[clix] = wgt

    s_list = []
    x_list = []
    y_list = []
    # master_page_ids = all_page_ids
    # all_page_ids = pid_override
    for row in tr.iterrows():
        print(row)
        ts = row[1]["creation_date"]
        gr_id = row[1]["group_id"]
        gb_id = row[1]["gameboard_id"]
        student_ids = list(get_student_list(gr_id)["user_id"])
        print(student_ids)
        student_data = get_user_data(student_ids)
        hexes= list(gb_qmap[gb_id])
        print(hexes)

        #n-hot binarise the y vector here
        y_true = numpy.zeros(len(all_page_ids))
        N = len(hexes)
        for hx in hexes:
            hxix = all_page_ids.index(hx)
            y_true[hxix]=1.0 #/N

        for psi in student_ids:
            y_list.append(y_true)
            # print(psi)
            S,X,U,A = pickle.loads(zlib.decompress(sxua[psi][ts]))
            print(numpy.sum(A))
            # print([all_page_ids[hx] for hx,el in enumerate(A) if el==1])
            s_list.append(S)
            # x_list.append(numpy.concatenate((U,A))) #X
            x_list.append(A)  # X
        # input("stank")

    gc.collect()

    s_list = numpy.array(s_list)
    x_list = numpy.array(x_list)
    y_list = numpy.array(y_list, dtype=numpy.int8)


    SCALE = True
    sc = StandardScaler()
    if SCALE:
        # print(x_list.shape)
        lenX = s_list.shape[0]
        # for ix,x_el in enumerate(x_list):
        #     x_list[ix] = sc.transform(x_list[ix].reshape(1,-1))

        start = 0
        gap = 5000
        while(start<lenX):
            end = min(start+gap, lenX)
            print("fitting scaler",start,end)
            sc.partial_fit(s_list[start:end,:])
            start += gap

        start = 0
        while(start<lenX):
            end = min(start+gap, lenX)
            print("scaling",start,end)
            s_list[start:end,:] = sc.transform(s_list[start:end,:])
            start += gap

    print(s_list.shape, x_list.shape)
    x_list = numpy.concatenate((s_list, x_list), axis=1)
    print(x_list.shape)
    gc.collect()

    #OK, we now have the four student profile vectors, and the true y vector, so we can fit the model
    if model is None:
        print("making model")
        print(S.shape, X.shape, U.shape, A.shape, y_list.shape)
        model = make_phybook_model(S.shape[0], X.shape[0], U.shape[0], A.shape[0], y_list.shape[1])
        print("model made")
        es = EarlyStopping(monitor='categorical_accuracy', patience=0, verbose=0, mode='auto')
        history = model.fit(x_list, y_list, verbose=1, epochs=100, callbacks=[es], shuffle=True, batch_size=32)
    pyplot.plot(history.history['categorical_accuracy'])
    # pyplot.plot(history.history['binary_crossentropy'])
    # pyplot.plot(history.history['categorical_crossentropy'])
    pyplot.plot(history.history['top_k_categorical_accuracy'])
    pyplot.show()
    scores = model.evaluate(x_list,y_list)
    print(scores)

    return model, sc  # , sscaler, levscaler, volscaler


# def get_top_k_hot(raw, k): # TODO clean up
#     args = list(reversed(raw.argsort()))[0:k]
#     print(args)
#     k = numpy.zeros(len(raw))
#     k[args] = 1.0
#     return k, args


def evaluate_hits_against_asst(ailist, y, max_y, y_preds, ylb):
    '''This method should take an N~question assignment, compare it against the top N question candidates from the hwgen
    system and return a score between [0..1]'''
    true_qs = set(y)
    N = len(true_qs)
    # print("N",N)

    agg_cnt = Counter()
    for row in y_preds:
        args = numpy.argsort(row)
        topN = list(reversed(args))[0:N]
        for t in topN:
            # print(t)
            tlab = ylb.classes_[t]
            # print(tlab)
            agg_cnt[tlab] += 1
    print(agg_cnt.most_common(N))

    agg_pred = y_preds.sum(axis=0)
    print(agg_pred.shape)
    print(y_preds.shape)

    assert agg_pred.shape[0] == y_preds.shape[1]  # shd have same num of columns as y_preds
    sortargs = numpy.argsort(agg_pred)
    sortargs = list(reversed(sortargs))
    maxN = sortargs[0:N]  # get the top N
    pred_qs = [ylb.classes_[ix] for ix in maxN]
    print(true_qs, " vs ", pred_qs)
    score = len(true_qs.intersection(pred_qs)) / N
    print("score = {}".format(score))
    return score


def get_top_subjs(x, ylb, n):
    top_topics = Counter()
    for ix in range(len(x)):
        v = x[ix]
        if v != 0:
            qid = all_qids[ix]
            # print(qid)
            cat = cat_lookup[qid]
            top_topics[cat] += 1
    return top_topics.most_common(n)


def print_student_summary(ai, psi, s, x, ylb, clz, t, p, pr):
    top_subjs = get_top_subjs(x, ylb, 5)
    pid = None if clz is None else ylb.classes_[clz]
    xlevs = []
    for ix, el in enumerate(x):
        if el == 1:
            xlevs.append(levels[all_qids[ix]])

    print("{}:{}\t{}|\t{}\t{}\t{}\t({})\t{}\t{}\t{}\t{}".format(ai, psi, pr, t, p, pid, clz, numpy.sum(x), s,
                                                                numpy.mean(xlevs),
                                                                top_subjs))


def save_class_report_card(ailist, S, U, X, y, y_preds, awgt, slist, qhist, ylb):
    max_probs = y_preds.max(axis=1)
    max_labs = y_preds.argmax(axis=1)
    fn_ai = None
    for ai, s, u, x, t, p, pr, wgt, psi, qh in zip(ailist, S, U, X, y, max_labs, max_probs, awgt, slist, qhist):
        if fn_ai is None:
            fn_ai = str(ai)
            f = open("report_cards/" + fn_ai + ".csv", "w")
            f.write("student,age,months_on_isaac,qns_tried,successes,hexes_done,top_5_topics,last_5_qns\n")

        tabu = []
        for ix in range(u.shape[0]):
            if u[ix] > 0:
                label = all_qids[ix]
                page = label.split("|")[0]
                if page not in tabu:
                    tabu.append(page)
        if len(tabu) > 0:
            tabu = "\n".join(map(str, tabu))
            tabu = '"{}"'.format(tabu)

        big5 = get_top_subjs(x, ylb, 5)
        if len(big5) > 0:
            big5 = "\n".join(map(str, big5))
            big5 = '"{}"'.format(big5)
        if len(qh) > 0:
            ql, tl = zip(*qh)
            last5 = [q for q in numpy.unique(ql)[-5:]]
            last5 = "\n".join(last5)
            last5 = '"{}"'.format(last5)  # wrap in quotes
        else:
            last5 = []
        f.write("{},{},{:0.1f},{},{},{},{},{}\n".format(psi, int(10 * s[0]) / 10.0, s[1] / 30.44, numpy.sum(x),
                                                        numpy.sum((u > 0)), tabu, big5, last5))
    f.close()


def evaluate_phybook_loss(tt,sxua, model, sc):
    x_list = []
    y_list = []
    s_list = []
    hex_list = []
    # all_page_ids = pid_override
    for row in tt.iterrows():
        print(row)
        ts = row[1]["creation_date"]
        gr_id = row[1]["group_id"]
        gb_id = row[1]["gameboard_id"]
        student_ids = list(get_student_list(gr_id)["user_id"])
        print(student_ids)
        student_data = get_user_data(student_ids)
        hexes= list(gb_qmap[gb_id])
        print(hexes)

        #n-hot binarise the y vector here
        y_true = numpy.zeros(len(all_page_ids), dtype=numpy.int8)
        for hx in hexes:
            hxix = all_page_ids.index(hx)
            y_true[hxix]=1.0

        for psi in student_ids:
            hex_list.append(hexes)
            y_list.append(y_true)
            # print(psi)
            # S,X,U,A = sxua[psi][ts]
            S,X,U,A = pickle.loads(zlib.decompress(sxua[psi][ts]))
            s_list.append(S)
            # x_list.append(numpy.concatenate((S,U,A))) #X
            x_list.append(A)

    # y_list = numpy.array(y_list)
    try:
        s_list = sc.transform(s_list)
    except:
        s_list = numpy.array(s_list)
    # scores = model.evaluate(x_list,y_list)
    # print(scores)
    x_list = numpy.concatenate((s_list,x_list), axis=1)

    print("results")
    predictions = model.predict(x_list)
    j_max = 0
    thresh_max = 0
    for j_thresh in [0.4]:
        j_sum = 0
        incl_sum = 0
        N = len(predictions)
        for p, t in zip(predictions, hex_list):
            phxs = []
            probs = []
            for ix, el in enumerate(p):
                if el>j_thresh:
                    phxs.append(all_page_ids[ix])
                    probs.append(p[ix])
            probs_shortlist = list(reversed(sorted(probs)))
            Z = reversed( [x for _, x in sorted(zip(probs, phxs))] )
            print(t, list(Z))
            print(probs_shortlist)
            t=set(t)
            phxs=set(phxs)
            if len(t.intersection(phxs)) > 0:
                incl_sum += 1
            j_sum  += len(t.intersection(phxs)) / len(t.union(phxs))
        if j_sum > j_max:
            j_max = j_sum
            thresh_max = j_thresh
        print("j_thresh =",j_thresh)
        print("Jaccard:", j_sum/N)
        print("Incl:", incl_sum/N)
        print("~ ~ ~ ~")
    print("max thresh/jacc:", thresh_max, j_sum)

def filter_assignments(assignments, book_only):
    # query = "select id, gameboard_id, group_id, owner_user_id, creation_date from assignments order by creation_date asc"
    assignments["include"] = True
    print(assignments.shape)
    map = make_gb_question_map()
    meta = get_meta_data()
    for ix in range(assignments.shape[0]):
        include = True
        gr_id = assignments.loc[ix, "group_id"]

        if book_only:
            gb_id = assignments.loc[ix, "gameboard_id"]
            hexes = map[gb_id]
            for hx in hexes:
                hx = hx.split("|")[0]
                if not (hx.startswith("ch_") or hx.startswith("ch-i")):
                    include = False
                    break

        if include:
            students = get_student_list([gr_id])
            if students.empty:
                include = False

        if include:
            include = False
            for psi in list(students["user_id"]):
                # print("checking",psi)
                atts = get_attempts_from_db(psi)
                if not atts.empty:
                    # print("OK")
                    include = True
                    break

        if not include:
            assignments.loc[ix, "include"] = False

    # assignments = assignments[assignments["include"]==True]
    print(assignments.shape)
    return assignments


page_list = []
if __name__ == "__main__":
    # tracemalloc.start()
    print("Initialising deep learning HWGen....")

    os.nice(3)

    concept_map = {}
    topic_map = {}
    concept_list = []
    meta_df = pandas.DataFrame.from_csv(base + "book_question_meta.csv")
    for thing in meta_df.iterrows():
        thing = thing[1]
        k = thing["URL:"].split("/")[-1]
        page_list.append(k)
        sft = "/".join((thing["Subject"], thing["Field"], thing["Topic"]))
        concepts = thing["Related Concepts"].split(",")
        concept_map[k] = concepts
        topic_map[k] = sft
        for c in concepts:
            if c not in concept_list:
                concept_list.append(c)

    pid_override = list(concept_map.keys())

    model = None
    # print("loading...")
    # with open(asst_fname, 'rb') as f:
    #     assignments = pickle.load(f)
    #
    # print("loaded {} assignments".format(len(assignments)))
    #

    USE_CACHED_ASSGTS = True
    SAVE_CACHED_ASSGTS = True
    cache_fname = base + "cached_assgts.csv"
    if USE_CACHED_ASSGTS:
        assignments = pandas.DataFrame.from_csv(cache_fname)
    else:
        assignments = get_all_assignments()
        assignments = filter_assignments(assignments, book_only=True)
        if SAVE_CACHED_ASSGTS:
            assignments.to_csv(cache_fname)
    # Now filter and split
    assignments = assignments[assignments["include"] == True]
    assignments["creation_date"] = pandas.to_datetime(assignments["creation_date"])

    print("building/loading SXUA")
    BUILD_SXUA = False
    if BUILD_SXUA:
        SXUA = {}
        student_static = {}
        last_ts = {}
        last_hexes = {}
        print("build dob cache")
        dob_cache= build_dob_cache(assignments)
        print("done")
        for row in assignments.iterrows():
            gc.collect()
            print(row)
            ts = row[1]["creation_date"]
            gr_id = row[1]["group_id"]
            gb_id = row[1]["gameboard_id"]
            student_ids = list(get_student_list(gr_id)["user_id"])
            print(student_ids)
            student_data = get_user_data(student_ids)
            now_hexes= list(gb_qmap[gb_id])
            print(now_hexes)
            for psi in student_ids:
                # print(psi)
                if psi not in SXUA:
                    S = numpy.zeros(4)
                    X = numpy.zeros(len(all_qids), dtype=numpy.int8)
                    U = numpy.zeros(len(all_qids))
                    A = numpy.zeros(len(all_page_ids), dtype=numpy.int8)
                    SXUA[psi] = {}
                    print("+",psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                    psi_data = student_data[student_data["id"]==psi]
                    rd = pandas.to_datetime(psi_data.iloc[0]["registration_date"])
                    # print(rd)
                    student_static[psi] = (rd,)
                else:
                    l_ts = last_ts[psi]
                    l_hexes = last_hexes[psi]
                    S,X,U,A = pickle.loads(zlib.decompress(SXUA[psi][l_ts]))
                    # S,X,U,A = copy(S),copy(X),copy(U),copy(A)
                    #make updates
                    attempts = get_attempts_from_db(psi)
                    recent_attempts = attempts[(attempts["timestamp"]>=l_ts) & (attempts["timestamp"]<ts)]
                    qids = list(recent_attempts["question_id"])
                    wins = list(recent_attempts[(recent_attempts["correct"] == True)]["question_id"])
                    for qid in qids:
                        try:
                            qix = all_qids.index(qid)
                        except:
                            print("UNK Qn ", qid)
                            continue
                        X[qix] = 1
                        if qid in wins:
                            attct = (attempts["question_id"] == qid).sum()
                            U[qix] = 1.0 / attct

                    for hx in l_hexes:
                        hxix = all_page_ids.index(hx)
                        A[hxix] = 1

                    S[0] = (ts - dob_cache[psi]).days / 365.242 if dob_cache[psi] is not None else 0
                    # print(ts, l_ts)
                    day_delta = max(1, (ts-l_ts).seconds)/ 86400.0
                    att_delta = recent_attempts.shape[0]
                    # print(day_delta, att_delta)
                    reg_date = student_static[psi][0]
                    # print(reg_date)
                    S[1] = (ts - reg_date).days
                    S[2] = (att_delta/day_delta)
                    S[3] = (len(wins)/att_delta if att_delta else 0)
                last_ts[psi] = ts
                last_hexes[psi] = now_hexes
                print("~",psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                SXUA[psi][ts] = zlib.compress(pickle.dumps((S,X,U,A)))

        # f = open(base+"SXUA.pkl", 'wb')
        # pickle.dump(SXUA, f)
        # f.close()
        # print("*** *** *** SAVED")

        # print("compressing SXUA")
        # for st in SXUA:
        #     for tstamp in SXUA[st]:
        #         SXUA[st][tstamp] = zlib.compress(pickle.dumps(SXUA[st][tstamp]))
        f = open(base + "SXUA.comp.pkl", 'wb')
        pickle.dump(SXUA, f)
        f.close()
        print("compressed and SAVED")

    else:
        f = open(base + "SXUA.comp.pkl", 'rb')
        SXUA = pickle.load(f)
        f.close()
    print("loaded")

    gc.collect()

    COUNT_TEACHERS=False
    if COUNT_TEACHERS:
        ct = Counter()
        for rix, row in enumerate(assignments.iterrows()):
            # print(row)
            oud = row[1]["owner_user_id"]
            ct[oud]+=1
        print(ct.most_common(20))
        exit()

    assignments = assignments[assignments["owner_user_id"]==7062]

    POST_FILTER=False
    if POST_FILTER:
        assignments["include"] = False
        for rix,row in enumerate(assignments.iterrows()):
            # print(row)
            aid = row[1]["id"]
            ts = row[1]["creation_date"]
            gr_id = row[1]["group_id"]
            gb_id = row[1]["gameboard_id"]
            student_ids = list(get_student_list(gr_id)["user_id"])
            # print(student_ids)
            # student_data = get_user_data(student_ids)
            hexes = list(gb_qmap[gb_id])
            # print(hexes)
            if len(hexes)==1 and len(student_ids)<=10:
                assignments.iloc[rix,6] = True

        print(assignments.shape[0])
        assignments = assignments[assignments["include"]==True]
        print(assignments.shape[0])

    do_train = True
    do_testing = True
    frisch_backen = True
    ass_n = 1050
    split = 50
    n_macroepochs = 1
    n_epochs = 100

    len_ass = assignments.shape[0]
    if ass_n<=0 or len_ass<=ass_n:
        ass_n = len_ass

    assignments = assignments[0:ass_n]

    # TODO it would be nicer to build this inside the generator object, but currently this is the only place where
    # the full assignment set is available
    # print("building oac")
    # open_asst_cache = build_oa_cache(assignments, gb_qmap)

    # assignments = assignments.sample(n=ass_n, random_state=666)
    # print(assignments["id"][0:10])
    tr = assignments[0:(ass_n - split)]
    tt = assignments[-split:]

    gc.collect()
    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    if do_train:
        print("training")
        model, sc = train_deep_model(tr, SXUA, n_macroepochs, n_epochs)
        print("...deleted original X,y")
        model.save(base + 'hwg_model.hd5')
        joblib.dump(sc, base + 'hwg_mlb.pkl')
        # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')

    if do_testing:
        print("testing")
        if model is None:
            model = load_model(base + "hwg_model.hd5")
            sc = joblib.load(base + 'hwg_mlb.pkl')

        evaluate_phybook_loss(tt, SXUA, model, sc)  # , sscaler,levscaler,volscaler)
        print("DEEP testing done")
