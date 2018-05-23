import gc
import os
import pickle
import random
import zlib
from collections import Counter
from random import seed, shuffle
from statistics import mean

import numpy
import pandas
from keras import callbacks, Input, Model
from keras.layers import Dense, Dropout, concatenate, LSTM, Embedding
from keras.models import load_model, Sequential
from numpy import argmax
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelBinarizer
from sklearn.utils import compute_class_weight

from hwgen.common import init_objects, get_meta_data, get_n_hot, split_assts, get_page_concepts, jaccard
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.model import make_mixed_loss_model, LSTM_model

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
# def convert_to_Xy(assts, split=None):
#     ret = []
#     ct = 0
#     no = len(assts)
#     if(split):
#         split = int(split * len(assts))
#         seed(666)
#         shuffle(assts)
#         tt = assts[0:split]
#         tr = assts[split:]
#     for s in [tr,tt]:
#         X = []
#         y = []
#         for a in s:
#             ct += 1
#             profiles = pickle.loads(zlib.decompress(a[5]))
#             print("{}/{}/{}".format(ct, len(s), no))
#             hexagons = a[3]
#             # qns_list = [s for s in all_qids if any(s.startswith(hx) for hx in hexagons)]
#
#             for psi in profiles:  # set up the training arrays here
#                 X.append(profiles[psi])
#                 y.append(hexagons)
#                 # y.append(qns_list)
#         ret.append(X)
#         ret.append(y)
#     ret.append(tt) # get the testing metadata too
#     return ret

def xy_generator(assts,batch_size=512, pid_override=None):
    b = 0 # batch counter
    c = 0 # cumulative counter
    S = []
    X= []
    lenX = len(assts)
    y = []
    y_cs = []
    y_ts = []
    y_lv = []
    y_v = []
    emcons = 0
    n = 0

    for a in assts:
        profiles = pickle.loads(zlib.decompress(a[7]))
        hexagons = a[3]

        if profiles is None or len(profiles)==0:
            print("skipping empty profiles object")
            continue

        topics = []
        # _levels = []
        # _concepts = []
        for hx in hexagons:
            if "|" in hx:
                hexagons.remove(hx)
                hx = hx.split("|")[0]
                hexagons.append(hx)
            topics.append(cat_page_lookup[hx])

        # if len(_concepts) == 0:
        #     emcons +=1
        #     continue

        for psi in profiles:  # set up the training arrays here
            s,x,elle = profiles[psi] #TODO add L vector here!

            #s = numpy.concatenate((s,elle))
            s = s[0:2]
            # s = numpy.concatenate((s,elle))
            if len(hexagons)==0:
                continue
            for hx in hexagons: #Split the assignment up by hexagon
                n += 1
                if pid_override is not None and hx not in pid_override:
                    continue

                S.append(s) #TODO use indices 0:2 to keep just student AGE and XP
                X.append(x)
                y.append([hx])
                y_ts.append(topics)
                tvolume = len(hexagons) #TODO wld prefer qn vol here but try hex volume for now
                # tvolume = 0
                # for tp in hexagons:
                #     tqids = [qid for qid in all_qids if qid.startswith(tp)]
                #     tvolume += len(tqids)

                y_v.append((tvolume,))

        c += 1

        n = len(X)
        # print("gen'd {} items".format(n))
        if (batch_size>0) and (n % batch_size == 0):
            print("b={}, n samples = {} ({}/{}={:.1f}%)".format(b,len(X), c, lenX, (100.0*c/lenX)))
            b+=1
            yield S,X,y
            S = []
            X = []
            y = []
            y_cs = []
            y_ts = []
            y_lv = []
            y_v = []
    print("out of assts")
    print("empty concepts = {} of {}".format(emcons,lenX))
    yield S, X, y


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


def make_phybook_model(n_S, n_Q, n_P, n_C):
    n_Voc = 10000
    n_Emb = 32
    # this is our input placeholder
    # input_S = Input(shape=(n_S,), name="s_input")
    input_Q = Input(shape=(n_Q,), name="wide_input")

    # e = Embedding(n_Voc, n_Emb, input_length=20)
    # lstm1 = LSTM(256, input_shape=(20,))

    # w = 1024 # best was 256
    w = 1024

    # hidden = Dense(5000, activation='relu')(input_Q)
    # hidden = Dropout(.2)(hidden)

    # hidden = Dense(1000, activation='relu')(input_Q)
    # hidden = Dropout(.2)(hidden)

    hidden = Dense(w, activation='relu')(input_Q)
    # hidden = Dropout(.33)(hidden)

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.33)(hidden)

    # hidden = Dense(w, activation='relu')(hidden)

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

    # m = Model(inputs=input_Q, outputs=[next_pg,next_concept] )
    # m.compile(optimizer="adam", loss=['categorical_crossentropy','binary_crossentropy'])

    m = Model(inputs=input_Q, outputs=next_pg )
    m.compile(optimizer="rmsprop", loss='categorical_crossentropy')

    m.summary()
    return m


numpy.set_printoptions(threshold=numpy.nan)
def train_deep_model(assts, n_macroepochs=100, n_epochs=10, concept_map=None):
    model = None
    mlb = None

    qn_list = list(concept_map.keys())
    concept_list = list(set().union(*concept_map.values()))
    print(concept_list)

    ylb = LabelBinarizer()
    ylb.fit(qn_list) # start by fitting the binariser to the shortlist of book qns

    clb = MultiLabelBinarizer(classes=concept_list)
    clb.fit(concept_list) # initialise the concept binariser too

    yship = []
    hex_counter = Counter()
    tot = 0
    for a in assts:
        hexagons = a[3]
        students = a[6]
        for hx in hexagons:
            for psi in students:
                yship.append(hx)
                hex_counter[hx]+=1
                tot += 1

    for hx in hex_counter.most_common():
        print(hx[0], hx[1])
    print(tot)

    weights = {}
    class_wgt = compute_class_weight('balanced', ylb.classes_, yship)
    for clix, (cls, wgt) in enumerate(zip(ylb.classes_,class_wgt)):
        print(clix,cls,wgt)
        weights[clix] = wgt

    nb_epoch = n_macroepochs
    for e in range(nb_epoch):
        xygen = xy_generator(assts, batch_size=-1)  # make generator object
        print("macroepoch %d of %d" % (e, nb_epoch))
        for S,X,y in xygen:
            y_labs = numpy.array(y)
            X = numpy.array(X, dtype=numpy.uint8)
            S = numpy.array(S)
            assert y_labs.shape[1]==1 # each line shd have just one hex assignment

            c_labs = [concept_map[label[0]] for label in y_labs]
            c = clb.transform(c_labs)
            print(c_labs[0:10])
            print(c[0:10])

            y = ylb.transform(y_labs)
            assert numpy.sum(y[0]) == 1 # Each line should be one-hot

            # hex_int_wgts = {}
            # oov_wgt = min(hex_wghts.values())
            # for clix, cls in enumerate(ylb.classes_):
            #     if cls in hex_wghts:
            #         hex_int_wgts[clix] = hex_wghts[cls]
            #     else:
            #         hex_int_wgts[clix] = oov_wgt

            es = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=2,
                                         verbose=1, mode='auto')
            if model is None:
                model = make_phybook_model(len(S[0]), len(X[0]), len(y[0]), len(c[0]))

            if len(y)<5:
                continue
            model.fit(X, y, epochs=n_epochs, shuffle=True, batch_size=128, callbacks=[es], validation_split=0.1, class_weight=weights)

            X=y=yc=lv = None
            gc.collect()

    return model, ylb, clb #, sscaler, levscaler, volscaler


def get_top_k_hot(raw, k):
    args = list(reversed(raw.argsort()))[0:k]
    print(args)
    k = numpy.zeros(len(raw))
    k[args] = 1.0
    return k, args

def evaluate_phybook_loss(tt, model, ylb, clb, concept_map, topic_map): #, sscaler,levscaler,volscaler): #, test_meta):
    print("ready to evaluate...")
    num_direct_hits =0
    num_cases = 0
    X_sums = []
    for S,X,y in xy_generator(tt, batch_size=10000):
        print("batch")
        S = numpy.array(S)
        X = numpy.array(X)
        X_sums.append(numpy.sum(X, axis=1))
        y = numpy.array(y)
        y_preds = model.predict(X, verbose=True)
        print(y_preds)
        # print(c_preds)
        # c_preds[c_preds >= 0.1] = 1
        # c_preds[c_preds < 0.1] = 0
        # sugg_c_labs = clb.inverse_transform(c_preds)

        print("inverting")
        max_y = ylb.inverse_transform(y_preds) #shd give us a single output label for the best choice!
        print("concating")
        y = numpy.concatenate(y)

        for x,t,p in zip(X, y, max_y):
            print("T/P {} {} ... {}".format(t,p, numpy.sum(x)))

        # for t,p,sg in zip(true_y, max_y, sugg_c_labs):
        #     true_c_labs = concept_map[t]
        #     pred_c_labs = concept_map[p]
            # print(true_c_labs)
            # print(pred_c_labs)
            # print(sg)
            # print("- - -@")

        num_direct_hits += numpy.sum(max_y == y)
        num_cases += len(max_y)
    batch_score = num_direct_hits / len(max_y)
    print("direct hits: {} of {}: {}".format(num_direct_hits, len(max_y), batch_score))
    X_sums = numpy.array(X_sums).ravel()
    print(X_sums)
    X_sums = (X_sums==0)
    tot_zero = numpy.sum(X_sums)
    print(tot_zero)

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
    print("loading...")
    with open(asst_fname, 'rb') as f:
        assignments = pickle.load(f)

    print("loaded {} assignments".format(len(assignments)))

    do_train = True
    do_testing = True
    ass_n = 2500
    split = 500
    n_macroepochs = 1
    n_epochs = 100
    assignment_keys = list(assignments.keys()) # train on a subset of the data
    random.seed(666)
    print("first 10 keys preshuff = ", assignment_keys[0:10])
    shuffle(assignment_keys)
    print("first 10 keys postshuff = ", assignment_keys[0:10])

    assert_profile_len = False
    if assert_profile_len:
        for k in assignment_keys:
            ass = assignments[k]
            students = ass[6]
            profiles = pickle.loads(zlib.decompress(ass[7]))
            assert len(students)==len(profiles)

    BOOK_ONLY=True
    print("Splitting...")
    tr,tt = split_assts(assignments, ass_n, split, BOOK_ONLY, pid_override)
    del assignments
    gc.collect()
    print("Split complete!")
    print("{} {}".format(len(tt), len(tr)))

    if do_train:
        print("training")
        model, ylb, clb = train_deep_model(tr, n_macroepochs, n_epochs, concept_map)
        print("...deleted original X,y")
        model.save(base + 'hwg_model.hd5')
        joblib.dump((ylb, clb), base + 'hwg_mlb.pkl')
        # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')

    if do_testing:
        print("testing")
        if model is None:
            model = load_model(base + "hwg_model.hd5")
            (ylb, clb) = joblib.load(base + 'hwg_mlb.pkl')
            #(sscaler,levscaler,volscaler) = joblib.load(base + 'hwg_scaler.pkl')
        # evaluate_predictions(tt, model, scaler, sscaler)
        evaluate_phybook_loss(tt, model, ylb, clb, concept_map, topic_map) #, sscaler,levscaler,volscaler)
        print("DEEP testing done")