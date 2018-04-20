import gzip
import json
import os
import _pickle
import pickle
import zlib
from ast import literal_eval
from random import seed, shuffle

import numpy
import pandas
from keras import callbacks
from keras.layers import Dropout, Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from hwgen.common import init_objects, get_meta_data
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.profiler import profile_student, get_attempts_from_db

base = "../../../isaac_data_files/"

#need to build a softmax classifier to recommend questions...
#this would have qn output nodes

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, _, _ , all_page_ids = init_objects(n_users)

hwdf = get_meta_data()
concepts_all = set()
hwdf.index = hwdf["question_id"]
hwdf["related_concepts"] = hwdf["related_concepts"].map(str)
for concepts_raw in hwdf["related_concepts"]:
    print(concepts_raw)
    concepts = eval(concepts_raw)
    if concepts is not None:
        concepts_all.update(concepts)
concepts_all = list(concepts_all)


asst_fname = base+"assignments.txt" #pkl"

con_page_lookup = page_to_concept_map()
def convert_to_Xy(assts, split=None):
    ret = []
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
            profiles = pickle.loads(zlib.decompress(a[4]))
            hexagons = a[3]
            # qns_list = [s for s in all_qids if any(s.startswith(hx) for hx in hexagons)]

            # this_cons = set()
            # for q in this_qns:
            #     if q in con_page_lookup:
            #         this_cons.update(con_page_lookup[q])


            for psi in profiles:  # set up the training arrays here
                X.append(profiles[psi])

                y.append(hexagons)
                # y.append(qns_list)
        ret.append(X)
        ret.append(y)
    ret.append(tt) # get the testing metadata too
    return ret

def make_model(n_in, n_out):
    print(n_in, n_out)
    mode="ML_SOFTMAX_WIDE"

    model = Sequential()

    if mode=="MLBIN":
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(4800, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(2400, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='sigmoid'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    elif mode=="MLBIN_SMALL":
        # model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(n_out, activation='relu', input_dim=n_in))
        # model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(8*n_out, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(8*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='sigmoid'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    elif mode=="ML_SOFTMAX":
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(8*n_out, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(4*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
    elif mode=="ML_SOFTMAX_WIDE":
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(2*n_in, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(n_in, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(int((n_in+n_out)/2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="categorical_crossentropy", optimizer='rmsprop')

    return model


# concepts_all = list(concept_extract())
def train_model(X,y):
    mlb = MultiLabelBinarizer(classes=all_page_ids)
    y = mlb.fit_transform(y)
    print("binarised, shape is ", y.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("fitted scaler, scaled")

    model = make_model(len(X[0]), len(y[0]))
    print("made model")
    es = callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=1,
                                  verbose=0, mode='auto')
    print("fitting")
    model.fit(X,y, validation_split=0.2, batch_size=256, epochs=10, shuffle=True, callbacks=[es])
    print("fitted")
    return model, mlb, scaler

def evaluate_predictions(preds, true_y_qns, ml_binariser, test_meta):
    fout = open(base + "hw_gobbler.out", "w")
    print(len(preds[0]))

    #first thing to do is convert the qn lists into n-hot encoding
    y = ml_binariser.transform(true_y_qns)

    scores = []
    answers = zip(preds, y)
    for asst_ix, meta in enumerate(test_meta):
        # print(meta[0],meta[1],meta[2],meta[3])
        ts = meta[0]
        gameboard = meta[1]
        group = meta[2]

        true_questions = meta[3]
        profiles = pickle.loads(zlib.decompress(meta[4]))
        class_size = len(profiles)
        no_items = len(true_questions)

        fout.write("-----{} : {} on {}\n".format(ts,group,gameboard))
        fout.write("TRUTH: {}\n".format(true_questions))
        class_scores = []
        guesses = numpy.zeros((len(all_page_ids),))
        for offset in range(class_size):
            pred_ix = asst_ix + offset
            guesses = numpy.add(guesses, preds[pred_ix] ) # sum up the scores across all guesses

        top = list(numpy.argsort(guesses))
        top.reverse()
        gix_list = [ix for ix in top[0: no_items]]
        guess_list = [all_page_ids[ix] for ix in gix_list]
        fout.write("GUESS: {}\n".format(guess_list))

        # top = list(numpy.argsort(sooth))
        # top.reverse()

        score_ab = 0
        for q in true_questions:
            if q in guess_list:
                score_ab += 1

        score = score_ab / no_items
        fout.write("{} of {} = {}\n".format(score_ab, no_items, score))
        scores.append(score)
        class_scores.append(score)  # hit rate for this prediction
    avgscore = numpy.mean(scores)
    fout.write("***\nAvg Score {}\n".format(avgscore))
    print("Avg Score {}".format(avgscore))

os.nice(3)

#ass_n = 10000 # the number of SUCCESSFUL (i.e. complete) assignments to process # incomplete assts are skipped and do not count
do_train = False
do_testing = True

model = None
print("loading...")
from pandas import Timestamp
# asses = []
# for a in open(asst_fname,"r"):
#     print("one")
#     asses.append(eval(a))

# L = open(asst_fname).readlines()
# print("eval..")
# asses = [eval(a) for a in L]

# with gzip.open(asst_fname, 'r') as f:
#     asses = _pickle.loads(f.read())
with open(asst_fname, 'rb') as f:
    asses = pickle.load(f)

print("loaded {} assignments".format(len(asses)))
X, y, test_X, test_y, test_meta = convert_to_Xy(asses, split=0.10)
del(asses) # TODO helps?

if do_train:
    model, mlb, scaler = train_model(X, y)
    model.save(base + 'hwg_model.hd5')
    joblib.dump(mlb, base + 'hwg_mlb.pkl')
    joblib.dump(scaler, base + 'hwg_scaler.pkl')

if do_testing:
    if model is None:
        model = load_model(base + "hwg_model.hd5")
        mlb = joblib.load(base + 'hwg_mlb.pkl')
        scaler = joblib.load(base + 'hwg_scaler.pkl')
    predictions = model.predict(test_X)
    print("eval'g")
    evaluate_predictions(predictions, test_y, mlb, test_meta)
    print("done")