import gc
import math
import os
import pickle
import zlib
from collections import Counter, defaultdict
from random import choice
from statistics import mean, median, stdev

import numpy
import openpyxl
import pandas
import pylab
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, concatenate
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt, pyplot
from openpyxl.styles import Alignment
from sklearn import cross_validation, feature_selection
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from hwgen.common import init_objects, get_meta_data, get_all_assignments, get_student_list, make_gb_question_map, \
    get_user_data, get_q_names
from hwgen.concept_extract import page_to_concept_map
from hwgen.hwgengen2 import hwgengen2, build_dob_cache
from hwgen.profiler import get_attempts_from_db

use_saved = True
do_train = False
do_testing = False
create_scorecards = True

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
        plt.savefig("learning_plot.png")


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


def make_phybook_model(n_S, n_X, n_U, n_A, n_P, lr):

    # this is our input placeholder
    if n_S is not None:
        input_S = Input(shape=(n_S,), name="s_input")
        inner_S = Dense(10, activation="relu")(input_S)

    w=200
    #w100: 17% to beat
    do=.2

    input_U = Input(shape=(n_X,), name="u_input")
    inner_U = Dense(300, activation="relu")(input_U)

    if n_S is not None:
        # hidden = concatenate([inner_U, inner_S])
        hidden = concatenate([inner_U, inner_S])
    else:
        hidden = inner_U

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    hidden = Dense(w, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)
    hidden = Dense(w, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)
    next_pg = Dense(n_P, activation='softmax', name="next_pg")(hidden)

    o = Adam(lr= lr)

    # m = Model(inputs=[input_S, input_U], outputs=[next_pg, output_U])
    # m.compile(optimizer=o, loss=["binary_crossentropy","binary_crossentropy"], metrics={'next_pg':["binary_accuracy", "top_k_categorical_accuracy"], 'outAuto':['binary_accuracy']})
    if n_S is not None:
        ins = [input_S, input_U]
    else:
        ins = input_U

    m = Model(inputs=ins, outputs=[next_pg])
    m.compile(optimizer=o, loss='categorical_crossentropy', metrics={'next_pg':['acc', 'top_k_categorical_accuracy']})
    plot_model(m, to_file='hwgen_model.png')
    m.summary()
    # input(",,,")
    return m


gb_qmap = make_gb_question_map()

numpy.set_printoptions(threshold=numpy.nan)

def feature_check(s_list, x_list, y_list):
    currMax =0
    sN=y_list.shape[0]
    x_list = x_list[numpy.random.choice(x_list.shape[0], sN, replace=False)]
    print(s_list.shape)
    print(x_list.shape)
    print(y_list.shape)
    nouveau_y = []
    for yel in y_list:
        ix = numpy.where(yel==1)[0]
        nouveau_y.append(ix)
    y_labs = numpy.array(nouveau_y).ravel()

    #X = numpy.concatenate((s_list, x_list), axis=1)[0:1000,:]
    #print(X.shape)
    kcounts = range(1, x_list.shape[1], 5)
    results = []
    x_list = x_list[0:sN, :]
    y_labs = y_labs[0:sN]
    # s_list = s_list[0:sN, :]
    y_list = y_list[0:sN, :]

    last_f_ct = 0
    es = EarlyStopping(monitor='acc', patience=1, verbose=0, mode='auto')
    for i in kcounts:
        print("building fs")
        fs = feature_selection.SelectKBest(feature_selection.f_classif, k=i)
        X_train_fs = fs.fit_transform(x_list, y_labs)
        print(X_train_fs.shape)
        f_ixs = fs.get_support(indices=True)
        print(i, "\n", f_ixs)
        if len(f_ixs) <= last_f_ct:
            break
        last_f_ct = len(f_ixs)
        #clf = SVC()
        bfn = lambda : make_phybook_model(None, X_train_fs.shape[1], 0,0, y_list.shape[1])
        model = KerasClassifier(build_fn=bfn, epochs=100, batch_size=32, verbose=1, shuffle=True)
        scores = cross_validation.cross_val_score(model, X_train_fs, y_list, cv=5, fit_params={'callbacks':[es]})
        # scores = cross_validation.cross_val_score(clf, numpy.concatenate([s_list, X_train_fs], axis=1), y_labs[0:sN], cv=5)

        # print i,scores.mean()
        results = numpy.append(results, scores.mean())
        if results.max() > currMax:
            fs_mx = fs
            currMax = results.max()
    # print (percentiles[optimal_percentil[0]])
    print("Optimal number of features:", len(fs_mx.get_support()), "\n")
    import pylab as pl
    pl.figure()
    pl.xlabel("Number of features selected")
    pl.ylabel("Cross validation accuracy)")
    pl.plot(kcounts,results)
    pl.show()
    print ("Mean scores:",results)

    return fs_mx

def top_n_of_X(X, fs):
    # top_n_features = [412, 413, 415, 417, 420, 421, 429, 431, 655, 656, 657, 658, 659, 660, 661,
    #                   662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 673, 683, 685, 686, 687,
    #                   688, 689, 690, 691, 693, 694, 695, 696, 697, 2494]
    # print(X.shape)
    # Xa = X[:,top_n_features]
    #     return Xa
    if fs is None:
        return X
    else:
        print(fs)
        Xa = fs.transform(X)
        return Xa

def augment_data(tr, sxua, filter_length=True):
    inverse_all_page_ids = {}
    for pix,pid in enumerate(all_page_ids):
        inverse_all_page_ids[pid] = pix

    psi_atts_cache = {}
    group_ids = pandas.unique(tr["group_id"])

    aid_list = []
    s_list = []
    x_list = []
    u_list = []
    a_list = []
    y_list = []
    psi_list = []
    hexes_to_try_list = []
    hexes_tried_list = []
    s_raw_list = []
    gr_id_list = []
    ts_list = []

    fout = open("tr_summ.csv","w")

    # orig_tss_map = {}
    # orig_tss = list(tr["creation_date"])
    # d_tss = list(tr["creation_date"].dt.floor("D"))
    # assert len(d_tss) == len(orig_tss)

    # tr["creation_date"] = tr["creation_date"].dt.floor("D")

    # for ts,ots in zip(d_tss, orig_tss):
    #     if ts not in orig_tss_map:
    #         orig_tss_map[ts] = ots # has the effect of storing the EARLIEST ots

    s_filter = set()
    for gr_id in group_ids:
        gr_ass = tr[tr["group_id"] == gr_id]
        student_ids = list(get_student_list(gr_id)["user_id"])
        tss = sorted(list(set(gr_ass["creation_date"])))
        if filter_length and len(tss)<5:
            pass
        else:
            s_filter.update(student_ids)

    group_track = {}
    for gr_id in group_ids:
        gr_ass = tr[tr["group_id"] == gr_id]
        student_ids = list(get_student_list(gr_id)["user_id"])
        tss = sorted(list(set(gr_ass["creation_date"])))
        for ts in tss:
            ts_rows = gr_ass[gr_ass["creation_date"]==ts]
            aids = list(ts_rows["id"])
            aid = aids[0]
            gb_ids = list(ts_rows["gameboard_id"])
            hexes = set()
            for gb_id in gb_ids:
                hexes.update(list(gb_qmap[gb_id]))

            for psi in student_ids:
                if psi not in s_filter:
                    continue

                if psi in group_track and group_track[psi]!=gr_id:
                    print("skipping n-th group")
                    continue
                else:
                    group_track[psi]=gr_id

                sxua_psi = sxua[psi]
                print(aid, psi, ts)
                S,_,U,A = pickle.loads(zlib.decompress(sxua_psi[ts]))
                if S[0]<16 or S[0]>18: #i.e. if student has no valid age TODO honolulu
                    continue
                # if S[1]==0: #no time in platform
                #     continue

                hexes_tried = []
                hexes_to_try = []
                # if len(hexes)==1:
                #     hexes_to_try = hexes
                # else:

                Xa = numpy.zeros(shape=len(all_page_ids))
                if psi in psi_atts_cache:
                    atts = psi_atts_cache[psi]
                else:
                    atts = get_attempts_from_db(psi)
                    psi_atts_cache[psi]=atts

                fatts = atts[atts["timestamp"] < ts]
                for qid in fatts["question_id"]:
                    pid = qid.split("|")[0]
                    if pid not in hexes_tried:
                        if pid in inverse_all_page_ids:
                            hexes_tried.append(pid)
                            Xa[inverse_all_page_ids[pid]]=1

                natts = fatts.shape[0]
                nsucc = len(set(fatts[fatts["correct"] == True]["question_id"]))
                ndist = len(set(fatts["question_id"]))
                dop = S[1]
                # passrate = nsucc/dop if dop>0 else -1
                # passprob = nsucc/natts if natts>0 else 0
                # passprob_perday = passprob / dop if dop>0 else -1

                crapness = dop * natts / (nsucc if nsucc > 0 else 0.1)

                del fatts

                for hx in hexes:
                    if hx not in hexes_tried:
                        hexes_to_try.append(hx)

                if hexes_to_try==[]:
                    print("no hexes to try")
                    continue

                y_true = numpy.zeros(len(pid_override))  # numpy.zeros(len(all_page_ids))
                hexes_to_try = sorted(hexes_to_try)
                #for hx in sorted(hexes_to_try):
                # hx = sorted(hexes_to_try)[ (len(hexes_to_try)-1)//2 ]

                TARGET_MODE = "first"
                if TARGET_MODE=="decision_weighted":
                    for hx in hexes_to_try:
                        hxix = pid_override.index(hx)
                        y_true[hxix] = 1.0 / len(hexes_to_try)
                elif TARGET_MODE=="no_weight":
                    for hx in hexes_to_try:
                        hxix = pid_override.index(hx)
                        y_true[hxix] = 1.0
                elif TARGET_MODE=="first":
                        hx = sorted(hexes_to_try)[0]
                        hxix = pid_override.index(hx)
                        y_true[hxix] = 1.0
                elif TARGET_MODE=="middle":
                        hx = sorted(hexes_to_try)[(len(hexes_to_try) - 1) // 2]
                        hxix = pid_override.index(hx)
                        y_true[hxix] = 1.0
                else:
                    raise ValueError("'{}' is not a valid target mode!".format(TARGET_MODE))

                X=Xa

                print("hexes t try: {}".format(hexes_to_try))
                print("hexes      : {}".format(hexes))

                aid_list.append(aid)
                s_raw_list.append(S)

                #age_1dp = int(10.0*S[0])/10.0
                onedp = lambda z : int(10.0*z)/10.0
                # nsucc = int(10.0 *nsucc / age_1dp)/10.0
                # s_list.append([(int(10*S[0])/10.0), S[1], natts, ndist, nsucc])
                # s_list.append([natts, ndist, nsucc])
                Sa = [onedp(S[0]), dop, natts, ndist, nsucc]
                # Sa = [0]
                s_list.append(Sa)# (nsucc/natts if natts>0 else 0)])
                x_list.append(X)
                u_list.append(U)
                a_list.append(A)
                y_list.append(y_true)
                psi_list.append(psi)
                hexes_to_try_list.append(hexes_to_try)
                hexes_tried_list.append(hexes_tried)
                gr_id_list.append(gr_id)
                ts_list.append(ts)

                # x_list = numpy.array(x_list)
                # input(x_list.shape)
                # x_list = x_list[:, numpy.nonzero(numpy.any(x_list != 0, axis=0))[0]]
                # input(x_list.shape)

                fout.write("{},{},{},{},{},{},{},\"{}\",\"{}\"\n".format(ts,gr_id,psi,",".join(map(str,Sa)), X.sum(), numpy.sum(X>0), numpy.sum(U), "\n".join(hexes_tried), "\n".join(hexes_to_try)))
    fout.close()
    # exit()
    # input("nibit")
    gc.collect()



    s_list = numpy.array(s_list)
    x_list = numpy.array(x_list, dtype=numpy.int16)
    # print(x_list.shape)
    # x_mask = numpy.nonzero(numpy.any(x_list != 0, axis=0))[0]
    # x_list = x_list[:, x_mask]
    # print(x_list.shape)
    u_list = numpy.array(u_list, dtype=numpy.int8)
    a_list = numpy.array(a_list, dtype=numpy.int8)
    y_list = numpy.array(y_list, dtype=numpy.int8)
    psi_list = numpy.array(psi_list)
    return aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list

def train_deep_model(tr, sxua, n_macroepochs=100, n_epochs=10, use_linear=False, load_saved_tr=False):
    model = None
    fs = None
    if load_saved_tr:
        # try:
        #     fs = joblib.load(base + 'hwg_fs.pkl')
        # except:
        #     print("no fs found, will create")
        #     fs=None
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load("tr.data")
    else:
        fs = None
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(tr, sxua)
        joblib.dump( (aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tr.data")

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
            partial_s = s_list[start:end,:]
            sc.partial_fit(partial_s)
            start += gap
        # sc.fit(s_list)

        start = 0
        while(start<lenX):
            end = min(start+gap, lenX)
            print("scaling",start,end)
            s_list[start:end,:] = sc.transform(s_list[start:end,:])
            start += gap
        # s_list = sc.transform(s_list)

    print(s_list.shape, x_list.shape)
    # x_list = numpy.concatenate((s_list, x_list), axis=1)
    print(x_list.shape)
    gc.collect()

    # if fs is None:
    #     fs = feature_check(s_list, x_list, y_list)

    #OK, we now have the four student profile vectors, and the true y vector, so we can fit the model
    max_mod = None

    x_mask = numpy.nonzero(numpy.any(x_list != 0, axis=0))[0]
    x_list = x_list[:, x_mask]

    if model is None:
        lrs = []
        accs = []
        BSs = []
        max_acc = -1
        max_BS = -1

        print("making model")
        # x_list = top_n_of_X(x_list, fs)
        S,X,U,A = s_list[0], x_list[0], u_list[0], a_list[0]
        print(S.shape, X.shape, U.shape, A.shape, y_list.shape)

        es = EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')
        # cves = EarlyStopping(monitor='acc', patience=1, verbose=0, mode='auto')
        # for BS in [50, 64, 100]:
        #     for LR in [0.003, 0.0025, 0.002]:
        # for BS in [40,50,60,70,80]:
        for BS in [32]: #80
            # for LR in [0.0015, 0.002, 0.0025, 0.003, 0.0035]:
            for LR in [0.001]: #0.0015
                model = make_phybook_model(S.shape[0], X.shape[0], U.shape[0], A.shape[0], y_list.shape[1], lr=LR)
                print("model made")
                # es = EarlyStopping(monitor='categorical_accuracy', patience=0, verbose=0, mode='auto')
                # history = model.fit([s_list, x_list], [y_list, x_list], verbose=1, epochs=100, callbacks=[es], shuffle=True, batch_size=32)

                # cv = KFold(n_splits=3, shuffle=True, random_state=666)
                # splits = cv.split(s_list, y_list)
                # for trixs,ttixs in splits:
                #     s_tr = s_list[trixs]
                #     x_tr = x_list[trixs]
                #     s_tt = s_list[ttixs]
                #     x_tt = x_list[ttixs]
                #     y_tr = y_list[trixs]
                #     y_tt = y_list[ttixs]
                #     history = model.fit([s_tr, x_tr], y_tr, validation_data=([s_tt,x_tt],y_tt), verbose=1, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)

                history = model.fit([s_list, x_list], y_list, verbose=1, validation_split=0.20, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)
                scores = model.evaluate([s_list,x_list],y_list)
                print(scores)
                input("cha-wang!")
                lrs.append(LR)
                accs.append( scores[1])
                BSs.append(BS)
                if scores[1] > max_acc:
                    max_mod = model
                    max_acc = scores[1]
                print(scores)

            do_plot = False
            if do_plot:
                pyplot.plot(history.history['acc'])
                # pyplot.plot(history.history['binary_crossentropy'])
                # pyplot.plot(history.history['categorical_crossentropy'])
                pyplot.plot(history.history['top_k_categorical_accuracy'])
                pyplot.plot(history.history['loss'])
                pyplot.plot(history.history['val_acc'])
                pyplot.plot(history.history['val_loss'])
                pyplot.legend(["cat acc","top k acc","loss","val cat acc","val loss"])
                pyplot.show()

        max_acc_ix = accs.index(max(accs))
        input((max(accs), lrs[max_acc_ix], BSs[max_acc_ix]))
    return max_mod, x_mask, sc  # , sscaler, levscaler, volscaler


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



# def print_student_summary(ai, psi, s, x, ylb, clz, t, p, pr):
#     top_subjs = get_top_subjs(x, ylb, 5)
#     pid = None if clz is None else ylb.classes_[clz]
#     xlevs = []
#     for ix, el in enumerate(x):
#         if el == 1:
#             xlevs.append(levels[all_qids[ix]])
#
#     print("{}:{}\t{}|\t{}\t{}\t{}\t({})\t{}\t{}\t{}\t{}".format(ai, psi, pr, t, p, pid, clz, numpy.sum(x), s,
#                                                                 numpy.mean(xlevs),
#                                                                 top_subjs))


def save_class_report_card(ts, aid, gr_id, S, X, U, A, y, m_list, y_preds, slist, q_names_df, po_filtered):

    N = len(y_preds)
    print(N)
    sum_preds = numpy.sum(y_preds, axis=0)
    print("sum of sums", numpy.sum(sum_preds))
    sum_preds = sum_preds / N
    max_sum_ix = sum_preds.argmax()
    max_sum_prob = sum_preds.max()

    vote_ct = Counter()
    for yp in y_preds:
        yp_max_ix = numpy.argmax(yp)
        label = pid_override[yp_max_ix]
        vote_ct[label]+=1

    max_vote_lab = vote_ct.most_common(1)[0][0]
    max_sum_lab = pid_override[max_sum_ix]
    print("max sum lab =", max_sum_lab, max_sum_prob)
    print("votes counted:",vote_ct.most_common(5))
    print("most voted =", max_vote_lab)


    wb = openpyxl.Workbook()
    ws = wb.active

    fn_ai = str(aid)
    r = 1
    col_headers = ["student", "age", "months_on_isaac", "qns_tried", "successes", "prev_assignts", "hexes_attempted", "top_10_topics", "last_10_qns",
                   "ISAAC_SUGGESTS", "DIFF: (too easy 1..5 too hard)", "TOPIC:(bad 1..3 good)"]
    col_widths=[len(ch) for ch in col_headers]
    for c,cv in enumerate(col_headers):
        ws.cell(r,1+c,cv)

    ws.cell(2,2, ts)

    ws.cell(2,1, "Classroom sugg'n 1:")
    ws.cell(3,1, "Classroom sugg'n 2:")
    ws.cell(2,10, max_sum_lab)
    ws.cell(3,10, max_vote_lab)

    r=4
    months_on_list = []
    for s, x, u, a, t, psi, mop, y_predlist in zip(S, X, U, A, y, slist, m_list, y_preds):

        visited_pids = []
        nzixes = x.nonzero()
        # print(nzixes[0])
        for nzix in nzixes[0]:
            pid = po_filtered[nzix]
            visited_pids.append(pid)
        print(visited_pids)

        maxlab="-"
        max_ixs_raw = list(reversed(list(y_predlist.argsort())))
        for mix in max_ixs_raw:
            mpid = pid_override[mix]
            if mpid not in visited_pids:
                maxlab = mpid
                break

        assert type(maxlab) is str

        atts = get_attempts_from_db(psi)
        fatts = atts[atts["timestamp"] < ts]

        cats_visit_ct = Counter()
        cats_succ_ct = Counter()
        # dbvisited_pids =[]

        correct_qids = set(fatts[fatts["correct"] == True]["question_id"])
        visited_qids = set(fatts["question_id"])
        for qid in visited_qids:
            pid = qid.split("|")[0]
            if pid in all_page_ids:
                # if qid not in visited_qids:
                # if qid in fatts[fatts["correct"]==True]["question_id"]:
                cat = cat_page_lookup[pid]
                cats_visit_ct[cat] += 1
                if qid in correct_qids:
                    cats_succ_ct[cat] += 1
                # if pid not in dbvisited_pids:
                #     dbvisited_pids.append(pid)
                # visited_qids.append(qid)
        natts = fatts.shape[0]
        nsucc = len(set(fatts[fatts["correct"]==True]["question_id"]))
        ndist = len(set(fatts["question_id"]))

        print(";;;;")
        # print(sorted(dbvisited_pids))
        print(sorted(visited_pids))

        # print(len(dbvisited_pids), len(visited_pids))
        # print(set(dbvisited_pids).symmetric_difference(set(visited_pids)))
        # assert sorted(dbvisited_pids) == sorted(visited_pids)

        visited_pids = "\n".join(map(str, visited_pids))

        assigned = []
        for ix,el in enumerate(a):
            if el > 0:
                label = pid_override[ix]
                page = label.split("|")[0]
                if page not in assigned:
                    assigned.append(page)
        if len(assigned) > 0:
            assigned = "\n".join(map(str, assigned))
        else:
            assigned = "-"

        big5 = cats_succ_ct.most_common(20)
        if len(big5) == 0:
            big5 = "-"
        else:
            temp = []
            for cnt,succ in big5:
                v = cats_visit_ct[cnt]
                temp.append("{}: {} ({})".format(cnt,succ,v))
            big5 = "\n".join(temp)

        last5 = list(pandas.unique(fatts["question_id"])[-10:])
        temp5 = []
        for n in last5:
            if n in q_names_df.index:
                tit = q_names_df.loc[n, "title"]
                if str(tit)!="nan":
                    temp5.append("{} ({})".format(tit, n))
                else:
                    temp5.append(n)
            else:
                temp5.append("UNK")
        last5 = temp5
        last5 = "\n".join(map(str,last5))

        del fatts
        # if len(qh) > 0:
        #     ql, tl = zip(*qh)
        #     last5 = [q for q in numpy.unique(ql)[-5:]]
        #     last5 = "\n".join(last5)
        #     last5 = '{}'.format(last5)  # wrap in quotes
        # else:
        #     last5 = []

        months_on = mop /30.44 #s[1] / 30.44
        months_on_list.append(months_on)

        print(s)
        for it in [psi, int(10 * s[0]) / 10.0, "{:.1f}".format(months_on), str(ndist)+" ("+str(natts)+")", nsucc, assigned, visited_pids, "_", last5, maxlab]:
            print(it)

        c=1
        for cv in [psi, int(10 * s[0]) / 10.0, "{:.1f}".format(months_on), str(ndist)+" ("+str(natts)+")", nsucc, assigned, visited_pids, big5, last5, str(mix)+":"+maxlab]:
            if cv == []:
                cv = "-"
            elif len(str(cv).split("\n")[0])>col_widths[c-1]:
                col_widths[c-1] = len(str(cv))
            ws.cell(r,c,cv)
            c += 1
        r += 1

    # for ci, cw in enumerate(col_widths):
    #     ws.column_dimensions[get_column_letter(ci + 1)].width = cw
    #
    # for ri, rh in enumerate(row_heights):
    #     ws.row_dimensions[ri+2].height = rh

    for col in ws.columns:
        max_length = 0
        column = col[0].column  # Get the column name
        for cell in col:
            cell.alignment = Alignment(horizontal="center", vertical="top")
            try:  # Necessary to avoid error on empty cells
                this_max = max([len(s) for s in str(cell.value).split('\n')])
                if this_max > max_length:
                    max_length = this_max
            except:
                pass
        adjusted_width = max_length * 1.2
        ws.column_dimensions[column].width = adjusted_width

    for row in ws.rows:
        max_height = 0
        rowname = row[0].row  # Get the column name
        for cell in row:
            try:  # Necessary to avoid error on empty cells
                cell_h = len(str(cell.value).split('\n'))
                print("for row {} cell value is {} at height {}".format(rowname, cell.value, cell_h))
                if cell_h > max_height:
                    # print("{} super {}, replaceing".format(cell_h, max_height))
                    max_height = cell_h
            except:
                pass
        adjusted_height = max_height * 11.5 # convert to points??
        ws.row_dimensions[rowname].height = adjusted_height

    months_av = mean(months_on_list)
    wb.save('./report_cards/{:.1f}_{}_{}.xlsx'.format(months_av, gr_id, aid))


def class_evaluation(_tt, sxua, model, sc, fs, load_saved_data=False):
    names_df = get_q_names()
    names_df.index = names_df["question_id"]

    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load("tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(_tt, sxua)
        joblib.dump( (aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tt.data")


    # for row in tt.iterrows():
    lookup = {}
    ts_grid_lookup = {}
    for aid,s,s_raw,x,u,a,y,psi,grid,ts,hxtt in zip(aid_list, s_list, s_raw_list, x_list, u_list, a_list, y_list, psi_list, gr_id_list, ts_list, hexes_to_try_list):
        if aid not in lookup:
            lookup[aid] = ([],[],[],[],[],[],[],[])
            ts_grid_lookup[aid] = (ts,grid)
        sl,sr,xl,ul,al,yl,psil,hxl = lookup[aid]
        sl.append(s)
        sr.append(s_raw)
        xl.append(x)
        ul.append(u)
        al.append(a)
        yl.append(y)
        hxl.append(hxtt)
        psil.append(psi)
        lookup[aid] = (sl,sr,xl,ul,al,yl,psil,hxl)

    lkk = list(lookup.keys())

    result_lkup = {}

    sum_exact_match = 0
    vote_exact_match = 0
    sum_deltas = []
    vote_deltas = []
    for aid in lkk:
        m_list = []
        s_list = []
        x_list = []
        ts, gr_id = ts_grid_lookup[aid]
        sl, srl, xl, ul, al, yl, psil, hxl = lookup[aid]


        xl = numpy.array(xl)[:,fs]
        sl = sc.transform(sl)

        hxtt = hxl[0]

        # x_arr = top_n_of_X(x_arr,fs)
        y_preds = model.predict([sl, xl])
        N = y_preds.shape[0]
        print(N)
        sum_preds = numpy.sum(y_preds, axis=0)
        print("sum of sums", numpy.sum(sum_preds))
        sum_preds = sum_preds / N
        max_sum_ix = sum_preds.argmax()
        max_sum_prob = sum_preds.max()

        vote_ct = Counter()
        for yp in y_preds:
            yp_max_ix : int = numpy.argmax(yp)
            label = pid_override[yp_max_ix]
            vote_ct[label]+=1

        max_vote_lab = vote_ct.most_common(1)[0][0]
        max_vote_ix = pid_override.index(max_vote_lab)
        max_sum_lab = pid_override[max_sum_ix]
        print("max sum lab =", max_sum_lab, max_sum_prob)
        print("votes counted:",vote_ct.most_common(5))
        print("most voted =", max_vote_lab)

        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)
        sum_delta = abs(y_true - max_sum_ix)
        vote_delta = abs(y_true - max_vote_ix)
        if max_sum_lab == p_true:
            sum_exact_match += 1
        if max_vote_lab == p_true:
            vote_exact_match += 1
        print(p_true, max_vote_lab, max_sum_lab)
        sum_deltas.append(sum_delta)
        vote_deltas.append(vote_delta)

        result_lkup[aid] = (max_sum_ix, max_vote_ix)

    print("sum mean delta: {}".format(mean(sum_deltas)))
    print("vote mean delta: {}".format(mean(vote_deltas)))
    n_aids = len(lkk)
    print("sum exacts {}/{} = {}".format(sum_exact_match, n_aids, sum_exact_match/n_aids))
    print("vote exacts {}/{} = {}".format(vote_exact_match, n_aids, vote_exact_match/n_aids))
    return result_lkup


def create_student_scorecards(tt,sxua, model, sc,fs, load_saved_data=False):
    names_df = get_q_names()
    names_df.index = names_df["question_id"]

    aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(tt, sxua)
    joblib.dump( (aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tt.data")

    print(x_list.shape)
    x_list = x_list[:, fs]
    print(x_list.shape)

    po_filtered = [all_page_ids[fsix] for fsix in fs]

    # for row in tt.iterrows():
    lookup = {}
    ts_grid_lookup = {}
    for aid,s_raw,s,x,u,a,y,psi,grid,ts in zip(aid_list, s_raw_list, s_list, x_list, u_list, a_list, y_list, psi_list, gr_id_list, ts_list):
        if aid not in lookup:
            lookup[aid] = ([],[],[],[],[],[],[])
            ts_grid_lookup[aid] = (ts,grid)
        sr,sl,xl,ul,al,yl,psil = lookup[aid]
        sr.append(s_raw)
        sl.append(s)
        xl.append(x)
        ul.append(u)
        al.append(a)
        yl.append(y)
        psil.append(psi)
        lookup[aid] = (sr,sl,xl,ul,al,yl,psil)

    lkk = list(lookup.keys())

    for aid in lkk:
        m_list = []
        s_list = []
        x_list = []
        ts, gr_id = ts_grid_lookup[aid]
        sr, sl, xl, ul, al, yl, psil = lookup[aid]
        predictions = []
        for s_raw,s,x,u,psi in zip(sr,sl,xl,ul,psil):
            # s_list.append(s)
            # x_list.append(x)
            # s_raw_list.append(s_raw)
            m_list.append(s_raw[1])
            print("student {} done".format(psi))

            if len(s_list)==0:
                continue

        s_arr = numpy.array(sl)
        x_arr = numpy.array(xl)

        predictions = model.predict([s_arr,x_arr])
        save_class_report_card(ts, aid, gr_id, s_raw_list, xl, ul, al, yl, m_list, predictions, psil, names_df, po_filtered)

    with open("a_ids.txt", "w+") as f:
        f.write("({})\n".format(len(aid_list)))
        f.writelines([str(a)+"\n" for a in sorted(aid_list)])
        f.write("\n")


def evaluate2(tt,sxua, model, sc,fs, load_saved_data=False):
    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua)
    print(s_list.shape)
    try:
        s_list = sc.transform(s_list)
    except ValueError as ve:
        print(ve)
        print("Don't forget to check your flags... maybe you have do_train==False and have changed S ...")


    x_list = x_list[:, fs]
    deltas = []
    signed_deltas = []
    exacts = 0
    for sl,xl,hxtt,hxtd in zip(s_list,x_list,hexes_to_try_list, hexes_tried_list):
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1)])
        y_hats = list(reversed(numpy.argsort(predictions)[0]))
        for candix in y_hats:
            if pid_override[candix] not in hxtd:
                y_hat = candix
                break

        p_hat = pid_override[y_hat]
        # y_trues = []
        # for p_true in sorted(hxtt):
        # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
        #     y_trues.append(pid_override.index(p_true))
        # y_true = median(y_trues)
        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)
        if(y_true == y_hat):
            exacts+=1
        # p_true = pid_override[int(y_true)]
        print(y_hat, y_true, p_hat, p_true)
        deltas.append( abs((y_hat-y_true)))
        signed_deltas.append( (y_hat - y_true))
    mu = numpy.mean(deltas)
    signed_mu = numpy.mean(signed_deltas)
    sigma = numpy.std(deltas) if len(deltas)>1 else 0.0
    print("Mean of {} difference from teachers, std={}".format(mu,sigma))
    print("Signed delta is {}".format(signed_mu))
    print("{} exact matches out of {} = {}".format(exacts, len(s_list), (exacts/len(s_list))))
    # exit()




def evaluate3(tt,sxua, model, sc,fs, load_saved_data=False):
    maxdops = 360
    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua)
    print(s_list.shape)
    try:
        s_list = sc.transform(s_list)
    except ValueError as ve:
        print(ve)
        print("Don't forget to check your flags... maybe you have do_train==False and have changed S ...")

    x_list = x_list[:, fs]
    ct = 0
    if maxdops:
        dops = [ sr[1] for sr in s_raw_list if sr[1] <= maxdops ]
    else:
        dops = [ sr[1] for sr in s_raw_list ]
    dopdelta = max(dops)

    delta_dict = {}
    exact_ct = Counter()
    strat_list = ["hwgen","step","lin","random"]

    for sl,sr,xl,hxtt,hxtd in zip(s_list,s_raw_list,x_list,hexes_to_try_list, hexes_tried_list):
        if sr[1] > maxdops:
            continue
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1)])
        y_hats = list(reversed(numpy.argsort(predictions)[0]))
        for candix in y_hats:
            if pid_override[candix] not in hxtd:
                hwgen_y_hat = candix
                break

        options = [p for p in pid_override if p not in hxtd]
        hts = [h for h in hxtd if (h in pid_override)]

        p_hat = pid_override[hwgen_y_hat]
        random_p_hat = choice(options)
        random_y_hat = pid_override.index(random_p_hat)
        step_y_hat = 0 if not hts else min(len(pid_override) - 1, pid_override.index(hts[-1]) + 1)
        #lin_y_hat = int((len(pid_override) - 2) * (sr[1] / dopdelta))
        lin_y_hat = int((41) * (sr[1] / dopdelta))

        # y_trues = []
        # for p_true in sorted(hxtt):
        # p_true = sorted(hxtt)[(len(hxtt)-1) //2]
        #     y_trues.append(pid_override.index(p_true))
        # y_true = median(y_trues)
        p_true = sorted(hxtt)[0]
        y_true = pid_override.index(p_true)

        for strat, y_hat in zip(strat_list, [hwgen_y_hat, step_y_hat, lin_y_hat, random_y_hat]):
            if strat not in delta_dict:
                delta_dict[strat] = []
            delta_dict[strat].append(((y_hat-y_true)))
            if(y_true == y_hat):
                exact_ct[strat]+=1

        ct += 1

    for strat in strat_list:
        delta_list = delta_dict[strat]
        sq_list = [ d*d for d in delta_list ]
        mu = numpy.mean(delta_list)
        medi = numpy.median(delta_list)
        stdev = numpy.std(delta_list)
        print("{}: mean={} med={} std={}".format(strat, mu, medi, stdev))
        print("Exact = {} of {} = {}".format(exact_ct[strat], ct, (exact_ct[strat]/ct)))
        print("MSE = {}\n".format(numpy.mean(sq_list)))
    input("tam")

def evaluate_by_bucket(tt,sxua, model, sc,fs, load_saved_data=False, group_data = None):
    bucket_step = 30
    bucket_width = 7

    # first_asst = {}
    all_assts = get_all_assignments()
    # for ts, grid in zip(list(all_assts["creation_date"]), list(all_assts["group_id"])):
    #     # print(ts,grid)
    #     students = get_student_list(grid)
    #     for psi in students:
    #         if psi not in first_asst:
    #             first_asst[psi] = ts


    buckets = [i for i in range(13)] #22 for bw 30
    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
            "tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(
            tt, sxua)
    # dops = [ s[1] for s in s_raw_list ]

    first_asst = {}
    for psi, aidts in zip(psi_list, ts_list):
        if psi not in first_asst:
            first_asst[psi] = aidts

    dops = [ (ts - first_asst[psi]).days for psi,ts in zip(psi_list, ts_list) ]

    s_list = sc.transform(s_list)
    exacts = 0
    stepexacts = 0
    linexacts = 0
    randexacts = 0
    mlookup = {}
    glookup = {}

    # dopdelta = min(max(dops),7+30*max(buckets)) - max(min(dops),30*min(buckets)-7)
    dopdelta = min(max(dops),(max(bucket_step,1)*max(buckets)))
    assert fs is not None
    x_list = x_list[:, fs]

    PLOT_MODE = "DELTA"
    collist = ["assignment", "group", "teacher", "student", "bucket", "actdop", "y_true", "random", "linear", "canonical", "hwgen"]
    if group_data:
        collist += ["group_sum","group_vote"]
    apc_df = pandas.DataFrame(columns=collist)
    report_ix = 0
    ct = 0
    bucket_counter = Counter()

    tid_pnt = 0
    psi_pnt = 0
    anon_tids = {}
    anon_psis = {}
    for aid,grid,psi,sl,actdop,xl,hxtt, hxtd, aidts in zip(aid_list,gr_id_list,psi_list,s_list, dops, x_list,hexes_to_try_list, hexes_tried_list, ts_list):
        # xl = top_n_of_X(xl,fs)
        predictions = model.predict([sl.reshape(1,-1), xl.reshape(1,-1)])

        # actdop = (aidts - first_asst[psi]).days

        teacher_id = (all_assts[all_assts["group_id"]==grid]["owner_user_id"]).iloc[0]
        # if teacher_id not in anon_tids:
        #     anon_tids[teacher_id] = tid_pnt
        #     tid_pnt +=1
        # teacher_id = anon_tids[teacher_id]

        # if psi not in anon_psis:
        #     anon_psis[psi] = psi_pnt
        #     psi_pnt +=1
        # psi = anon_psis[psi]

        for b in buckets:
            if abs((bucket_step*b) - actdop)<=bucket_width:
                bucket_counter[b] += 1
                ct += 1
                y_hats = []
                y_hats_raw = list(reversed(numpy.argsort(predictions)[0]))
                yix = 0
                while len(y_hats)<len(hxtt):
                    cand = pid_override[y_hats_raw[yix]]
                    yix+=1
                    if cand not in hxtd:
                        y_hats.append(pid_override.index(cand))

                # p_hat = pid_override[y_hat]

                #RANDOM SELECTION
                options = [p for p in pid_override if p not in hxtd]

                #N+1
                hts = [h for h in hxtd if (h in pid_override)]
                # n1_p_hat = [h for h in sorted(hxtt) if (h in pid_override)][0]
                # n1_y_hat = pid_override.index(n1_p_hat)

                ix = 0
                for p_true, y_hat in zip(sorted([hxtt[0]]), [y_hats[0]]):
                    random_p_hat = choice(options)
                    random_y_hat = pid_override.index(random_p_hat)
                    n1_y_hat = 0 if not hts else min(len(pid_override) - 1, pid_override.index(hts[-1]) + ix)
                    lin_y_hat = int((len(pid_override)-2)*(bucket_step*b / dopdelta))
                    ix+=1

                    # y_trues=[]
                    # for p_true in sorted(hxtt):
                    #     y_trues.append(pid_override.index(p_true))
                    # y_true = mean(y_trues)
                    y_true = pid_override.index(sorted(hxtt)[0])

                    # y_true = pid_override.index(p_true)
                    if (y_true == y_hat):
                        exacts += 1
                    if (y_true == n1_y_hat):
                        stepexacts += 1
                    if (y_true == lin_y_hat):
                        linexacts += 1
                    if (y_true == random_y_hat):
                        randexacts += 1

                    #print(y_hat, y_true, p_hat, p_true)delta = abs(y_hat - y_true)


                    if PLOT_MODE=="ABS":
                        delta = abs(y_hat - y_true)
                        signed_delta = abs(y_hat - y_true)
                        rand_delta = abs(random_y_hat - y_true)
                        n1_delta = abs(n1_y_hat - y_true)
                        lin_delta = abs(lin_y_hat - y_true)
                    elif PLOT_MODE=="DELTA":
                        delta = (y_hat - y_true)
                        signed_delta = (y_hat - y_true)
                        rand_delta = (random_y_hat - y_true)
                        n1_delta = (n1_y_hat - y_true)
                        lin_delta = (lin_y_hat - y_true)
                    elif PLOT_MODE=="RAW":
                        delta = y_hat
                        signed_delta = y_hat
                        rand_delta = random_y_hat
                        n1_delta = n1_y_hat
                        lin_delta = lin_y_hat
                    else:
                        raise ValueError("{} is an invalid plotting mode!".format(PLOT_MODE))

                    replist = [int(aid),grid,teacher_id, psi, bucket_step*b, actdop, y_true, random_y_hat, lin_y_hat, n1_y_hat, y_hat]
                    if group_data is not None:
                        (gr_sum_ix, gr_vote_ix) = group_data[aid]
                        if b not in glookup:
                            glookup[b] = [ (gr_sum_ix - y_true, gr_vote_ix - y_true) ]
                        else:
                            deltas = glookup[b]
                            deltas.append( (gr_sum_ix - y_true, gr_vote_ix - y_true) )
                        replist += [gr_sum_ix, gr_vote_ix]

                    if not b in mlookup:
                        mlookup[b] = [ (delta, signed_delta, rand_delta, n1_delta, lin_delta) ]

                    else:
                        deltas = mlookup[b]
                        deltas.append( (delta, signed_delta, rand_delta, n1_delta, lin_delta) )
                        mlookup[b] = deltas

                    apc_df.loc[report_ix] = replist
                    report_ix += 1

    for sm in [exacts,stepexacts,linexacts,randexacts]:
        print(sm, sm/ct)


    bucketx = []
    buckety = []
    for b in sorted(list(bucket_counter.keys())):
        bucketx.append((max(bucket_width, 1) * b))
        buckety.append(bucket_counter[b])
    plt.plot(bucketx, buckety)
    plt.show()

    # apc_df.index = apc_df["assignment"]
    # apc_df.drop("assignment", inplace=True)
    apc_df.to_csv("apc38_train_df.csv")
    y_del_vals = []
    y_randels = []
    y_n1dels = []
    y_lindels = []
    y_actuals = []
    for b in mlookup:
        dels, sdels, randels, n1dels, lin_dels = zip(*mlookup[b])
        n_samples = len(dels)
        totdels = sum(dels)
        totsdels = sum(sdels)
        # mu = totdels / n_samples
        # smu = totsdels / n_samples
        mu = mean(dels)
        smu = mean(sdels)
        smed = median(sdels)
        sig = stdev(sdels) if len(sdels)>1 else 0.0
        print(b, n_samples, mu, smu, smed, sig)
        y_del_vals.append(mean(sdels))
        y_randels.append(mean(randels))
        y_n1dels.append(mean(n1dels))
        y_lindels.append(mean(lin_dels))
        y_actuals.append(0)

    xvals = sorted(mlookup.keys())
    for yvals in y_actuals, y_del_vals, y_randels, y_n1dels, y_lindels:
        plt.scatter(xvals, yvals)
        new_buckets = numpy.linspace(xvals[0], xvals[-1], 50)
        # smooth = spline(xvals, yvals, new_buckets)
        # s = InterpolatedUnivariateSpline(xvals, yvals)
        # ynew = s(new_buckets)
        z = numpy.polyfit(xvals, yvals, 2)
        p = numpy.poly1d(z)
        plt.plot(new_buckets, p(new_buckets))
        # plt.plot(xvals,yvals)
    labels = ['actual', 'hwgen', 'random', 'canonical', 'linear']

    if group_data is not None:
        g_y_sumvals = []
        g_y_votevals = []
        for b in glookup:
            sumdels, votedels = zip(*glookup[b])
            g_y_sumvals.append(mean(sumdels))
            g_y_votevals.append(mean(votedels))
        xvals = sorted(glookup.keys())
        for yvals in g_y_sumvals, g_y_votevals:
            plt.scatter(xvals, yvals)
            new_buckets = numpy.linspace(xvals[0], xvals[-1], 50)
            z = numpy.polyfit(xvals, yvals, 2)
            p = numpy.poly1d(z)
            plt.plot(new_buckets, p(new_buckets))
        labels = labels + ["group (sum)", "group (vote)"]

    plt.legend(labels)
    plt.xlabel("Student time elapsed (months)")
    plt.ylabel("Relative position in syllabus (qns)")
    plt.show()


def evaluate_phybook_loss(tt,sxua, model, sc, load_saved_data=False):

    if load_saved_data:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load("tt.data")
    else:
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(tr, sxua)
        joblib.dump( (aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list),"tt.data")

    # hex_list = []
    # all_page_ids = pid_override
    # ailist = []

    for row in tt.iterrows():
        aid = row[1]["id"]
        # ts = row[1]["creation_date"]
        gr_id = row[1]["group_id"]
        gb_id = row[1]["gameboard_id"]
        student_ids = list(get_student_list(gr_id)["user_id"])
        print(student_ids)
        student_data = get_user_data(student_ids)
        hexes= list(gb_qmap[gb_id])
        print(hexes)

        for _ in student_ids:
            aid_list.append(aid)
            # hex_list.append(hexes)

    s_list = sc.transform(s_list)
    s_list = numpy.array(s_list)

    x_list = numpy.array(x_list)
    u_list = numpy.array(u_list)
    a_list = numpy.array(a_list)

    print(s_list.shape, x_list.shape, u_list.shape, a_list.shape)

    print("results")
    print(model.get_input_shape_at(0))
    # x_list = top_n_of_X(x_list,fs)
    predictions = model.predict([s_list, x_list])
    j_max = 0
    thresh_max = 0
    dir_hits_max = 0
    for j_thresh in [0.001, 0.005, 0.01, 0.025, .05, 0.075, .1,.2,0.3, 0.4, 0.5, 0.6, 0.7]:
    # for j_thresh in [0.4]:
        j_sum = 0
        # dir_sum = 0
        incl_sum = 0
        dir_hits = 0
        N = len(predictions)
        this_ai = None
        for ai, p, s,x,a,y in zip(aid_list, predictions, s_list,x_list,a_list,y_list):
            t = [pid_override[yix] for yix,yval in enumerate(y) if yval==1]
            if ai != this_ai:
                print("\n...new asst",ai)
                this_ai = ai
            phxs = []
            probs = []
            print("pshape",p.shape)
            maxpox = numpy.argmax(p)
            print(maxpox, len(pid_override))
            max_guess = pid_override[maxpox]
            phxs.append(max_guess)

            probs.append(p[maxpox])
            for ix, el in enumerate(p):
                if el>j_thresh and pid_override[ix] not in phxs:
                    phxs.append(pid_override[ix])
                    probs.append(p[ix])
            probs_shortlist = list(reversed(sorted(probs)))
            Z = list(reversed( [x for _, x in sorted(zip(probs, phxs))] ))
            # if Z:
            #     for t_el in t:
            #         if t_el in Z:#'direct hit'
            #             dir_sum += 1.0/len(t)
            print(t, Z)
            print(probs_shortlist)
            # print([all_page_ids[hx] for hx,el in enumerate(a) if el==1])
            if max_guess not in t:
                robot="BAD ROBOT"
            else:
                if max_guess == t[0]:
                    robot = "GREAT ROBOT"
                    dir_hits += 1
                else:
                    robot = "GOOD ROBOT"
            print("{} {}, XP={}".format(robot, sc.inverse_transform(s), numpy.sum(x)))
            t=set(t)
            phxs=set(phxs)
            if len(t.intersection(phxs)) > 0:
                incl_sum += 1
            j_sum  += len(t.intersection(phxs)) / len(t.union(phxs))
        j_score = j_sum/N
        # dir_score = dir_sum/N
        if dir_hits > dir_hits_max:
            j_max = j_score
            thresh_max = j_thresh
            dir_hits_max = dir_hits
            # dir_for_j_max = dir_score
        print("j_thresh =",j_thresh)
        print("Jaccard:", j_score)
        print("Incl:", incl_sum/N)
        print("D/H:", dir_hits/N)
        print("~ ~ ~ ~")
    print("max thresh/jacc:", thresh_max, j_max, dir_hits_max/N)
    print("num examples", N)

    #save_class_report_card(ailist,S,U,X,y,y_preds,awgt,slist,qlist,ylb)

    # print("max D/H:",dir_for_j_max)

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


concept_map = {}
topic_map = {}
concept_list = []
page_list = []
meta_df = pandas.DataFrame.from_csv(base + "book_question_meta.csv")
for thing in meta_df.iterrows():
    thing = thing[1]
    k = thing["URL:"].split("/")[-1]
    page_list.append(k)
    sft = "/".join((thing["Subject"], thing["Field"], thing["Topic"]))
    # concepts = thing["Related Concepts"].split(",")
    # concept_map[k] = concepts
    topic_map[k] = sft
    for c in concepts:
        if c not in concept_list:
            concept_list.append(c)

pid_override = list(topic_map.keys())

if __name__ == "__main__":
    # tracemalloc.start()
    print("Initialising deep learning HWGen....")
    os.nice(3)

    model = None
    sc = None
    fs = None
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
    assignments["creation_date"] = assignments["creation_date"].dt.floor("D")

    BUILD_SXUA = False
    if BUILD_SXUA:
        print("building SXUA")
        SXUA = {}
        student_static = {}
        last_ts = {}
        last_hexes = {}
        print("build dob cache")
        try:
            dob_cache = joblib.load(base+"dob_cache")
        except:
            dob_cache= build_dob_cache(assignments)
            joblib.dump(dob_cache, base+"dob_cache")
        print("done")

        group_ids = pandas.unique(assignments["group_id"])
        print(len(assignments))
        print(len(group_ids))
        print(group_ids[0:20])
        # exit()

        for gr_id in group_ids:
            gr_ass = assignments[assignments["group_id"]==gr_id]
            for row in gr_ass.iterrows():
                # for row in assignments.iterrows():
                aid = row[1]["id"]
                # print(row)
                ts = row[1]["creation_date"]
                # gr_id = row[1]["group_id"]
                gc.collect()
                gb_id = row[1]["gameboard_id"]
                student_ids = list(get_student_list(gr_id)["user_id"])
                # print(student_ids)
                student_data = get_user_data(student_ids)
                now_hexes= list(gb_qmap[gb_id])
                # print(now_hexes)
                # if 118651 not in student_ids:
                #     continue
                for psi in student_ids:
                    # if psi != 118651:
                    #     continue
                    # print(psi)
                    if psi not in SXUA:
                        S = numpy.zeros(6)
                        X = numpy.zeros(len(all_qids), dtype=numpy.int16)
                        U = numpy.zeros(len(all_qids), dtype=numpy.int8)
                        A = numpy.zeros(len(pid_override), dtype=numpy.int8)
                        SXUA[psi] = {}
                        print("+",psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                        psi_data = student_data[student_data["id"]==psi]
                        rd = pandas.to_datetime(psi_data.iloc[0]["registration_date"])
                        # print(rd)
                        student_static[psi] = (rd,)
                        l_ts = pandas.to_datetime("1970-01-01 00:00:00")
                        l_hexes = []
                    else:
                        l_ts = last_ts[psi]
                        l_hexes = last_hexes[psi]
                        S,X,U,A = pickle.loads(zlib.decompress(SXUA[psi][l_ts]))
                    # S,X,U,A = copy(S),copy(X),copy(U),copy(A)
                    #make updates

                    # if psi ==118651:
                    #     print("birdskoeping")

                    attempts = get_attempts_from_db(psi)
                    attempts = attempts[attempts["timestamp"] < ts]
                    all_wins = list(attempts[(attempts["correct"] == True)]["question_id"])

                    recent_attempts = attempts[attempts["timestamp"]>=l_ts]
                    # qids = list(set(recent_attempts["question_id"]))
                    qids = list(set(recent_attempts["question_id"]))
                    recent_wins = list(recent_attempts[(recent_attempts["correct"] == True)]["question_id"])

                    for qid in qids:
                        try:
                            qix = all_qids.index(qid)
                            attct = numpy.sum(recent_attempts["question_id"] == qid)
                            X[qix] += attct
                            if qid in recent_wins:
                                U[qix] = 1
                        except:
                            print("UNK Qn ", qid)
                            continue

                    print(l_hexes)
                    for hx in l_hexes:
                        hxix = pid_override.index(hx)
                        A[hxix] = 1

                    S[0] = (ts - dob_cache[psi]).days / 365.242 if dob_cache[psi] is not None else 0
                    # print(ts, l_ts)
                    day_delta = max(1, (ts-l_ts).seconds)/ 86400.0
                    att_delta = recent_attempts.shape[0]
                    all_atts = attempts.shape[0]
                    # print(day_delta, att_delta)
                    reg_date = student_static[psi][0]
                    # print(reg_date)
                    all_days = max(0, (ts - reg_date).days)
                    S[1] = all_days
                    S[2] = (att_delta/day_delta) #recent perseverence
                    S[3] = (len(recent_wins)/att_delta if att_delta else 0) # recent success rate
                    S[4] = (all_atts / all_days if all_days else 0) # all time perseverance
                    S[5] = (len(all_wins) / all_atts if all_atts else 0) # all time success rate

                    last_ts[psi] = ts
                    last_hexes[psi] = now_hexes
                    print("~",psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                    SXUA[psi][ts] = zlib.compress(pickle.dumps((S,X,U,A)))
                    # if str(aid) in ["47150", "49320", "53792"]:
                    #     input(">> {}".format(aid))

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
        print("loading SXUA")
        f = open(base + "SXUA.comp.pkl", 'rb')
        SXUA = pickle.load(f)
        f.close()
    print("loaded")

    gc.collect()
    print("gc'd")

    # assignments = assignments[assignments["owner_user_id"]==7062]
    # assignments = assignments[assignments["owner_user_id"]==6026]

    POST_FILTER=True
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
            if len(student_ids)<=75:
                assignments.iloc[rix,6] = True

        print(assignments.shape[0])
        assignments = assignments[assignments["include"]==True]
        print(assignments.shape[0])
        print("post filtered")


    COUNT_TEACHERS=True
    if COUNT_TEACHERS:
        ct = Counter()
        for t in numpy.unique(assignments["owner_user_id"]):
            t_assignments = assignments[assignments["owner_user_id"] == t]
            ct[t] = t_assignments.shape[0]
        print(ct.most_common(20))
        print("teachers counted")


    t_train = None
    t_test = None


    asslimit = 30000
    assct = 0
    teacherct = 0
    for (t, tct) in list(ct.most_common(len(ct))):
        t_assignments = assignments[assignments["owner_user_id"] == t]
        this_ass = t_assignments.shape[0]
        if this_ass < 10:
            print("not enough assignments",t,tct,this_ass)
            del ct[t]
        else:
            teacherct += 1
            assct += this_ass
        if assct > asslimit:
            break
    print(teacherct, assct)

    teacherN = teacherct
    test_insts = 0
    target_test_insts = 10 #teacherN // 10

    for_test = False
    for (t, tct) in ct.most_common(teacherN):
        t_assignments = assignments[assignments["owner_user_id"] == t]
        print(t_assignments.shape[0], "new training assts")
        # sel_n = min(t_assignments.shape[0], assct // teacherN)
        # t_assignments = t_assignments.iloc[0:sel_n, :]

        # nass = t_assignments.shape[0]
        # if nass <2:
        #     this_split = 0
        # else:
        # this_split = 1
        # print("this split", this_split)
        # temp_t_tr = t_assignments.iloc[0:-(split//teacherN),:]
        # temp_t_tt = t_assignments.iloc[-(split//teacherN):, :]
        # end = t_assignments.shape[0] - this_split
        # temp_t_tr = t_assignments.iloc[0:end,:]
        # temp_t_tt = t_assignments.iloc[end:, :]

        if for_test and test_insts<target_test_insts:
            for_test = False
            test_insts += 1
            if t_test is None:
                t_test = t_assignments
            else:
                t_test = pandas.concat([t_test, t_assignments])
        else:
            for_test = True
            if t_train is None:
                t_train = t_assignments
            else:
                t_train = pandas.concat([t_train, t_assignments])

        # print("training dates:", temp_t_tr["creation_date"].min(), temp_t_tr["creation_date"].max())
        # print("testing dates:", temp_t_tt["creation_date"].min(), temp_t_tt["creation_date"].max())
        # if t_train is None:
        #     t_train = temp_t_tr
        #     t_test = temp_t_tt
        #     print("created t_train {} and t_test {}".format(len(t_train),len(t_test)))
        # else:
        #     t_train = pandas.concat([t_train, temp_t_tr])
        #     t_test = pandas.concat([t_test, temp_t_tt])
        #     print("extended t_train {} and t_test {}".format(len(t_train),len(t_test)))
        # if len(t_train) + len(t_test) >= totass:
        #     print("exceeded totass")
        #     break
    tr = t_train
    tt = t_test


    gc.collect()
    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    n_macroepochs =100
    n_epochs = 100

    # aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = joblib.load(
    #     "tr.data")
    # print(len(numpy.unique(aid_list)))
    # print(len(numpy.unique(psi_list)))
    # print(len(aid_list))
    # exit()

    if do_train:
        print("training")
        model, fs, sc = train_deep_model(tr, SXUA, n_macroepochs, n_epochs, load_saved_tr=use_saved)
        print("...deleted original X,y")
        model.save(base + 'hwg_model.hd5')
        joblib.dump(fs, base + 'hwg_fs.pkl')
        joblib.dump(sc, base + 'hwg_mlb.pkl')
        # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')

    if model is None:
        model = load_model(base + "hwg_model.hd5")
        fs = joblib.load(base + 'hwg_fs.pkl')
        sc = joblib.load(base + 'hwg_mlb.pkl')

    numpy.set_printoptions(precision=4)
    if do_testing:
        print("testing")
        evaluate3(tt, SXUA, model, sc,fs, load_saved_data=use_saved)
        # input("now class")
        class_ev_lookup = class_evaluation(tt, SXUA, model, sc, fs, load_saved_data=use_saved)
        # evaluate_phybook_loss(tt, SXUA, model, sc, load_saved_data=use_saved)  # , sscaler,levscaler,volscaler)
        # input("DEEP testing done")
        print("m testing")
        evaluate_by_bucket(tt, SXUA, model, sc,fs, load_saved_data=use_saved, group_data=class_ev_lookup)

    if create_scorecards:
        create_student_scorecards(tt, SXUA, model, sc,fs, load_saved_data=use_saved)
