import gc
import math
import os
import pickle
import random
import zlib
from collections import Counter, defaultdict
from random import seed, shuffle
from statistics import mean
from typing import Union

import keras
import numpy
import pandas
import pylab
from keras import callbacks, Input, Model
from keras.layers import Dense, Dropout, concatenate, LSTM, Embedding
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelBinarizer
from sklearn.utils import compute_class_weight

from hwgen.common import init_objects, get_meta_data, get_n_hot, split_assts, get_page_concepts, jaccard, \
    get_all_assignments, get_student_list, make_gb_question_map
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.hwgengen2 import hwgengen2, gen_qhist
from hwgen.profiler import get_attempts_from_db

from matplotlib import pyplot as plt

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


def cluster_and_print(assts):
    xygen = hwgengen2(assts, batch_size=-1, FRESSSH=False)  # make generator object
    for S,X,U,y,ai,awgt in xygen:
            y_labs = numpy.array(y)
            if (X==[]):
                continue
            S = numpy.array(S) # strings (labels)
            X = numpy.array(X) #floats (fade)
            U = numpy.array(U) #signed ints (-1,0,1)
            print("S",S.shape)
            print("X",X.shape)
            print("U",U.shape)
            assert y_labs.shape[1]==1 # each line shd have just one hex assignment

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
            colors = numpy.array( [lab_set.index(l) for l in y_labs] )[0:n]

            # calc_entropies(X,y_labs)
            # exit()

            # pca = PCA(n_components=2)
            tsne = TSNE(n_components=2)
            # converted = pca.fit_transform(X) # convert experience matrix to points
            converted = tsne.fit_transform(X[0:n])

            plt.scatter(x=converted[:,0], y=converted[:,1], c=colors, cmap=pylab.cm.cool)
            plt.show()

def calc_entropies(X, y):
    d = defaultdict(list)
    for x,lab in zip(X,y):
        d[str(lab)].append(x)
    for l in d:
        print("calc for {}, len {}".format(l,len(d[l])))
        H = entropy(d[l])
        print("H({}) = {}".format(l, H))

def entropy(lizt):
    "Calculates the Shannon entropy of a list"
    # get probability of chars in string
    lizt = [tuple(e) for e in lizt]
    prob = [float(lizt.count(entry)) / len(lizt) for entry in lizt]
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy


def dummy_generator(assts,batch_size=512, pid_override=None):
    b = 0 # batch counter
    S = []
    X= []
    lenX = 1024 #len(assts)
    y = []
    y_cs = []
    y_ts = []
    y_lv = []
    y_v = []
    emcons = 0
    n=1
    for i in range(lenX):
        if i % 2 == 0:
            hx = "ch_j_p4"
            X.append([1,0])
            y.append([hx])
        else:
            hx = "ch_b_p5"
            X.append([0,1])
            y.append([hx])
        n+=1

        if (batch_size>0) and (n % batch_size == 0):
            print("b={}, n samples = {} ({}/{}={:.1f}%)".format(b,len(X), n, lenX, (100.0*n/lenX)))
            b+=1
            yield S,X,y
            S = []
            X = []
            y = []

    print("out of assts")
    print("empty concepts = {} of {}".format(emcons,lenX))
    yield S, X, y


def make_phybook_model(n_S, n_X, n_U, n_P):
    n_Voc = 10000
    n_Emb = 32
    # this is our input placeholder
    input_S = Input(shape=(n_S,), name="s_input")
    input_X = Input(shape=(n_X,), name="x_input")
    input_U = Input(shape=(n_U,), name="u_input")

    # e = Embedding(n_Voc, n_Emb, input_length=20)
    # lstm1 = LSTM(256, input_shape=(20,))

    # w = 1024 # best was 256
    # w = (n_S + n_X + n_P)//2
    w = 256

    # i_X = Dense(200, activation='relu')(input_X)
    # i_U = Dense(200, activation='relu')(input_U)

    hidden = Dense(w, activation='relu')(concatenate([input_S, input_X, input_U]))
    hidden = Dropout(.5)(hidden)

    # hidden = Dense((w+n_P)//2, activation='relu')(hidden)

    # decode_test = Dense(n_Q, activation="sigmoid", name="decode_test")(hidden)
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.5)(hidden)

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.5)(hidden)

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.5)(hidden)

    # hidden = Dense(1000, activation='relu')(hidden)


    #EXTRA DEEP CHICAGO STYLE!
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # next_concept = Dense(n_C, activation='sigmoid', name="next_concept")(hidden)
    next_pg = Dense(n_P, activation='softmax', name="next_pg")(hidden)

    # m = Model(inputs=input_X, outputs=[next_pg,next_concept] )
    # m.compile(optimizer="adam", loss=['categorical_crossentropy','binary_crossentropy'])

    o = Adam()

    m = Model(inputs=[input_S,input_X, input_U], outputs=next_pg )
    m.compile(optimizer=o, loss='categorical_crossentropy', metrics=['acc'])

    m.summary()
    return m


numpy.set_printoptions(threshold=numpy.nan)
def train_deep_model(tr, n_macroepochs=100, n_epochs=10, concept_map=None, pid_override=None, bake_fresh=False):
    model = None

    concept_list = list(set().union(*concept_map.values()))
    print(concept_list)

    clb = MultiLabelBinarizer(classes=concept_list)
    # clb.fit(concept_list) # initialise the concept binariser too

    yship = []
    qlist = pid_override
    hex_counter = Counter()
    tot = 0
    gb_qmap = make_gb_question_map()
    for i,ass in enumerate(tr.iterrows()):
        gb_id = ass[1]["gameboard_id"]
        gr_id = ass[1]["group_id"]

        hexagons = [gb_qmap[gb_id][0]]
        students = get_student_list(gr_id)
        for psi in students:
            for hx in hexagons:
                if hx not in qlist:
                    print(hx," not in qlist")
                    qlist.append(hx)
                yship.append(hx)
                hex_counter[hx]+=1
                tot += 1

    # yship = list(concept_map.keys()) +yship
    ylb = LabelBinarizer() #(classes=qlist)
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
    for clix, (cls, wgt) in enumerate(zip(ylb.classes_,class_wgt)):
        print(clix,cls,wgt)
        weights[clix] = wgt

    nb_epoch = n_macroepochs
    for e in range(nb_epoch):
        xygen = hwgengen2(tr, batch_size=-1, FRESSSH=bake_fresh, qid_override=qlist)  # make generator object
        print("macroepoch %d of %d" % (e, nb_epoch))
        for S,X,U,y,ai,awgt,_,_ in xygen:

            # if False:
            #     for aie, s, x, u, t in zip(ai, S, X, U, y):
            #         print_student_summary(aie, s, u, ylb, None, t, None, None)
            #     input("standing by")

            y_labs = numpy.array(y)
            if (X==[]):
                continue
            S = numpy.array(S) # strings (labels)
            X = numpy.array(X) #floats (fade)
            U = numpy.array(U) #signed ints (-1,0,1)
            # print("S",S.shape)
            # print("X",X.shape)
            # print("U",U.shape)
            assert y_labs.shape[1]==1 # each line shd have just one hex assignment

            # c_labs = [concept_map[label[0]] for label in y_labs]
            # c = clb.transform(c_labs)
            # print(c_labs[0:10])
            # print(c[0:10])

            try:
                y = ylb.transform(y_labs) #/ awgt
            except:
                y = ylb.fit_transform(y_labs) #/ awgt
            # print(y_labs)
            # input(y)
            # input(awgt)
            # assert numpy.sum(y[0]) == 1 # Each line should be one-hot

            # hex_int_wgts = {}
            # oov_wgt = min(hex_wghts.values())
            # for clix, cls in enumerate(ylb.classes_):
            #     if cls in hex_wghts:
            #         hex_int_wgts[clix] = hex_wghts[cls]
            #     else:
            #         hex_int_wgts[clix] = oov_wgt

            es = callbacks.EarlyStopping(monitor='acc',
                                         min_delta=0,
                                         patience=2,
                                         verbose=1, mode='auto')
            if model is None:
                print("making model")
                print(S.shape, X.shape, U.shape, y.shape)
                model = make_phybook_model(S.shape[1], X.shape[1], U.shape[1], y.shape[1])
                print("model made")

            # if len(y)<5:
            #     continue
            # for r in range(S.shape[0]):
            #     print(S[r]

            model.fit([S,X,U], y, epochs=n_epochs, shuffle=True, batch_size=8, callbacks=[es]) #, class_weight=weights)

            scores = model.evaluate([S,X,U], y)
            print(scores[0], scores[1])

            X=y=yc=lv = None
            gc.collect()

    return model, ylb, qlist #, sscaler, levscaler, volscaler


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


    assert agg_pred.shape[0] == y_preds.shape[1] # shd have same num of columns as y_preds
    sortargs = numpy.argsort(agg_pred)
    sortargs = list(reversed(sortargs))
    maxN = sortargs[0:N] # get the top N
    pred_qs = [ylb.classes_[ix] for ix in maxN]
    print(true_qs," vs ",pred_qs)
    score = len(true_qs.intersection(pred_qs))/N
    print("score = {}".format(score))
    return score


def get_top_subjs(x, ylb, n):
    top_topics = Counter()
    for ix in range(len(x)):
        v = x[ix]
        if v!=0:
            qid = all_qids[ix]
            # print(qid)
            cat = cat_lookup[qid]
            top_topics[cat] += 1
    return top_topics.most_common(n)

def print_student_summary(ai, psi, s,x, ylb, clz, t, p, pr):
    top_subjs = get_top_subjs(x, ylb, 5)
    pid = None if clz is None else ylb.classes_[clz]
    xlevs = []
    for ix, el in enumerate(x):
        if el == 1:
            xlevs.append(levels[all_qids[ix]])

    print("{}:{}\t{}|\t{}\t{}\t{}\t({})\t{}\t{}\t{}\t{}".format(ai, psi, pr, t, p, pid, clz, numpy.sum(x), s, numpy.mean(xlevs),
                                                             top_subjs))

def save_class_report_card(ailist, S, U, X, y, y_preds, awgt, slist, qhist, ylb):
    max_probs = y_preds.max(axis=1)
    max_labs = y_preds.argmax(axis=1)
    fn_ai = None
    for ai, s, u, x, t, p, pr, wgt, psi, qh in zip(ailist, S, U, X, y, max_labs, max_probs, awgt, slist, qhist):
        if fn_ai is None:
            fn_ai = str(ai)
            f = open("report_cards/"+fn_ai+".csv","w")
            f.write("student,age,months_on_isaac,qns_tried,successes,hexes_done,top_5_topics,last_5_qns\n")

        tabu = []
        for ix in range(u.shape[0]):
            if u[ix] > 0:
                label = all_qids[ix]
                page = label.split("|")[0]
                if page not in tabu:
                    tabu.append(page)
        if len(tabu)>0:
            tabu = "\n".join(map(str,tabu))
            tabu = '"{}"'.format(tabu)

        big5 = get_top_subjs(x, ylb, 5)
        if len(big5)>0:
            big5 = "\n".join(map(str,big5))
            big5 = '"{}"'.format(big5)
        if len(qh)>0:
            ql,tl = zip(*qh)
            last5 = [q for q in numpy.unique(ql)[-5:]]
            last5 = "\n".join(last5)
            last5 = '"{}"'.format(last5) #wrap in quotes
        else:
            last5 = []
        f.write("{},{},{:0.1f},{},{},{},{},{}\n".format(psi, int(10*s[0])/10.0, s[1]/30.44, numpy.sum(x), numpy.sum((u>0)), tabu, big5, last5))
    f.close()

def evaluate_phybook_loss(tt, model, ylb, clb, concept_map, topic_map, qid_override=None): #, sscaler,levscaler,volscaler): #, test_meta):
    print("ready to evaluate...")
    num_direct_hits =0
    errs = 0
    num_chapter_hits =0
    ass_tot = 0
    num_cases = 1
    num_students = 0
    test_gen = hwgengen2(tt, batch_size="assignment", FRESSSH=False, qid_override=qid_override) #batch_size = "group"
    for S,X,U,y,ailist,awgt,slist,qhist in test_gen:
        print("batch")

        psi_len = len(set(slist)) # <-- best direct hits we can get is one per student

        S = numpy.array(S)
        if X==[]:
            continue
        X = numpy.array(X) # Xperience is a float vector (due to fade)
        U = numpy.array(U) #success is -1,0,1
        y = numpy.array(y) # string vector
        y_preds = model.predict([S,X,U], verbose=True)
        #convert predictions to recommendations here
        print("Preds done")
        # print(y_preds)
        # print(c_preds)
        # c_preds[c_preds >= 0.1] = 1
        # c_preds[c_preds < 0.1] = 0
        # sugg_c_labs = clb.inverse_transform(c_preds)

        print("inverting")
        max_y = ylb.inverse_transform(y_preds) #shd give us a single output label for the best choice!
        print("concating")
        y = numpy.concatenate(y)

        ass_tot += evaluate_hits_against_asst(ailist,y, max_y, y_preds, ylb)

        num_students+=psi_len
        save_class_report_card(ailist, S, U, X, y, y_preds, awgt, slist, qhist, ylb)

        for ai, s,u,x, t,p, pr,clz,this_preds,w,psi in zip(ailist, S,U,X, y,max_y, y_preds.max(axis=1), y_preds.argmax(axis=1), y_preds, awgt, slist):
            # if numpy.sum(x) < 20:
            #     continue

            sortargs = numpy.argsort(this_preds)
            # print(sortargs)
            sortargs = list(reversed(numpy.argsort(this_preds)))
            #print(sortargs)
            pred_labels = [ylb.classes_[ix] for ix in sortargs]
            pred_probas = list(reversed(numpy.sort(this_preds)))
            tabus = [u[ix] for ix in sortargs]
            # print(t,p)
            # print(p10)
            # print(t10)

            print("STUDENT: {} (real is {})".format(psi,t))
            print(s)
            print(numpy.sum((x>0)), numpy.sum((u>0)), 1.0/numpy.mean(u[u>0]))
            for i in range(5):
                print("   {} {:.5f} {}".format(pred_labels[i], pred_probas[i], tabus[i]))


            # if(numpy.sum(t10)>0):
            #     input("")

            # i=0
            # while u[sortargs[i]] != 0: #knock out the tabu qns
            #     i += 1
            # p = ylb.classes_[sortargs[i]]

            # print(p10)
            # print(t10)
            # print(this_preds[0:10])
            # if(i > 0):
            #     input(("{}th arg".format(i),t,p))

            if t==p:
                num_direct_hits += 1
            #
            # if t != p:
            #     errs += 1.0/w[0]
            #
            # if t[0:4]==p[0:4]:
            #     num_chapter_hits+=1
            num_cases += 1
            #
            # print_student_summary(ai, psi, s, u, ylb, clz, t, p, pr)

            # for t,p,sg in zip(true_y, max_y, sugg_c_labs):
            #     true_c_labs = concept_map[t]
            #     pred_c_labs = concept_map[p]
            # print(true_c_labs)
            # print(pred_c_labs)
            # print(sg)
            # print("- - -@")

            # num_direct_hits += numpy.sum(max_y == y)
            # for el,el2 in zip(max_y,y):
            #     if el[0:4]==el2[0:4]:
            #         num_chapter_hits+=1

            # num_cases += len(max_y)
    asst_level_score = ass_tot / num_cases
    batch_score = num_direct_hits / num_cases
    print("direct hits: {} of {}: {}".format(num_direct_hits, num_cases, batch_score))
    print("chapter hits: {} of {}: {}".format(num_chapter_hits, num_cases, num_chapter_hits/num_cases))
    print("aggregated avg score = ",asst_level_score)
    print("wgted error rate = ", (errs/num_cases))
    print("student hits = ",(num_direct_hits/num_students))
    # X_sums = numpy.array(X_sums).ravel()
    # print(X_sums)
    # X_sums = (X_sums==0)
    # tot_zero = numpy.sum(X_sums)
    # print(tot_zero)


def filter_assignments(assignments, book_only):
    #query = "select id, gameboard_id, group_id, owner_user_id, creation_date from assignments order by creation_date asc"
    assignments["include"]=True
    print(assignments.shape)
    map = make_gb_question_map()
    meta = get_meta_data()
    for ix in range(assignments.shape[0]):
        include = True
        gr_id = assignments.loc[ix,"group_id"]

        if book_only:
            gb_id = assignments.loc[ix,"gameboard_id"]
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


if __name__ == "__main__":
    print("Initialising deep learning HWGen....")

    os.nice(3)

    concept_map = {}
    topic_map = {}
    concept_list = []
    meta_df = pandas.DataFrame.from_csv(base+"book_question_meta.csv")
    for thing in meta_df.iterrows():
        thing = thing[1]
        k = thing["URL:"].split("/")[-1]
        sft = "/".join((thing["Subject"],thing["Field"],thing["Topic"]))
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
    do_train = True
    do_testing = False
    frisch_backen = False
    ass_n = 10005
    split = 5
    n_macroepochs = 1
    n_epochs = 100

    USE_CACHED_ASSGTS=True
    SAVE_CACHED_ASSGTS=True
    cache_fname=base+"cached_assgts.csv"
    if USE_CACHED_ASSGTS:
        assignments = pandas.DataFrame.from_csv(cache_fname)
    else:
        assignments = get_all_assignments()
        assignments = filter_assignments(assignments, book_only=True)
        if SAVE_CACHED_ASSGTS:
            assignments.to_csv(cache_fname)
    #Now filter and split
    assignments = assignments[assignments["include"]==True]
    assignments["creation_date"] = pandas.to_datetime(assignments["creation_date"])
    # assignments = assignments[assignments["creation_date"] >=pandas.to_datetime("2016-01-01")]

    frac = 1 if ass_n <= 0 else (ass_n / assignments.shape[0])
    frac = min(1.0, frac)
    assignments = assignments.sample(frac=frac, random_state=666)
    tt = assignments[0:split]
    tr = assignments[split:]

    # del assignments

    # cluster_and_print(tr)
    # exit()

    # ass_list = []
    # xygen = hwgengen.xy_generator(assignments, batch_size=1)  # make generator object
    # for thing in xygen:
    #     ass_list.append(thing)

    # ass_list = random.shuffle(ass_list)
    # sc = len(ass_list//ass_n)
    # tr = assignments[0:-(sc*split)]
    # tt = assignments[-(sc*split):]

    gc.collect()
    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    if do_train:
        print("training")
        model, ylb, qlist = train_deep_model(tr, n_macroepochs, n_epochs, concept_map=concept_map, pid_override=pid_override, bake_fresh=frisch_backen)
        print("...deleted original X,y")
        model.save(base + 'hwg_model.hd5')
        joblib.dump((ylb, qlist), base + 'hwg_mlb.pkl')
        # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')

    if do_testing:
        print("testing")
        if model is None:
            model = load_model(base + "hwg_model.hd5")
            (ylb, qlist) = joblib.load(base + 'hwg_mlb.pkl')
            #(sscaler,levscaler,volscaler) = joblib.load(base + 'hwg_scaler.pkl')
        # evaluate_predictions(tt, model, scaler, sscaler)
        evaluate_phybook_loss(tt, model, ylb, None, concept_map, topic_map, qid_override=qlist) #, sscaler,levscaler,volscaler)
        print("DEEP testing done")