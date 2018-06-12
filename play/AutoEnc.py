import gc
import gzip
import json
import os
import _pickle
import pickle
import random
import zlib
from ast import literal_eval
from copy import copy
from random import seed, shuffle

import numpy
import pandas
from keras import callbacks, Input, Model
from keras.layers import Dropout, Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam, Adadelta
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from hwgen.common import init_objects, get_meta_data
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.model import make_model
from hwgen.profiler import profile_student, get_attempts_from_db

base = "../../../isaac_data_files/"

#need to build a softmax classifier to recommend questions...
#this would have qn output nodes

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


asst_fname = base+"assignments.txt" #pkl"
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

def xy_generator(assts,batch_size=512):
    n = 0 # this-batch size
    b = 0 # batch counter
    c = 0 # cumulative counter
    L = len(assts)
    X= []
    y = []
    for a in assts:
        profiles = pickle.loads(zlib.decompress(a[5]))
        hexagons = a[3]
        # qns_list = [s for s in all_qids if any(s.startswith(hx) for hx in hexagons)]

        if profiles is None or len(profiles)==0:
            print("skipping empty profiles object")
            continue

        n += 1
        c += 1

        for psi in profiles:  # set up the training arrays here
            X.append(profiles[psi])
            y.append(hexagons)
            # y.append(qns_list)

        if n >= batch_size > 0:
            print("b={}, n samples = {} ({}/{}={:.1f}%)".format(b, n, c, L, (100.0*c/L)))
            b+=1
            yield X,y
            X = []
            y = []
            n = 0
    print("out of assts")
    yield X,y

def train_autoencoder(assts):


    scaler = StandardScaler()
    sgen = xy_generator(assts, batch_size=1000)
    for X,_ in sgen:
        scaler.partial_fit(X)
    sgen = None

    pca = PCA(n_components=256)
    pcagen = xy_generator(assts, batch_size=-1)
    for X,_ in pcagen:
        pca.fit(X)

    datagen = xy_generator(assts, batch_size=512)  # make generator object

    model = None
    mlb = None


    nb_epoch = 1
    gc.collect()
    es = callbacks.EarlyStopping(monitor='val_loss',
                                 min_delta=0,
                                 patience=2,
                                 verbose=0, mode='auto')

    # for e in range(nb_epoch):
    #     print("epoch %d" % e)
    for X, y in datagen:
        io_dim = 9090
        if model is None:
            print("a/e io dim:",io_dim)
            encoding_dim = 1000  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

            # this is our input placeholder
            input_layer = Input(shape=(io_dim,))
            # "encoded" is the encoded representation of the input
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            # "decoded" is the lossy reconstruction of the input
            do = Dropout(0.5)(encoded)
            encoded2 = Dense(encoding_dim, activation='relu')(do)
            decoded = Dense(io_dim, activation='sigmoid')(encoded2)

            # this model maps an input to its reconstruction
            model = Model(input_layer, decoded)
            model.compile(optimizer='adadelta', loss='mse')

        # print("fit here")
        X = scaler.transform(X)
        model.fit(X,X, validation_split=0.2, batch_size=128, epochs=10, callbacks=[es])
        X = None
        y = None
        gc.collect()
    return model, scaler

# concepts_all = list(concept_extract())
def train_model2(assts):
    xygen = xy_generator(assts, batch_size=256) # make generator object
    model = None
    mlb = None
    scaler = None

    nb_epoch = 1
    gc.collect()
    for e in range(nb_epoch):
        print("epoch %d" % e)
        for X, y in xygen:
            if mlb is None:
                mlb = MultiLabelBinarizer(classes=all_page_ids)
                y = mlb.fit_transform(y)
                print("binarised, shape is ", y.shape)
                # scaler = StandardScaler()
            else:
                y = mlb.transform(y)
            # scaler.partial_fit(X)
            # X = scaler.transform(X)
            # es = callbacks.EarlyStopping(monitor='val_loss',
            #                              min_delta=0,
            #                              patience=1,
            #                              verbose=0, mode='auto')
            if model is None:
                model = make_model(len(X[0]), len(y[0]))

            # print("fit here")
            model.train_on_batch(X, y)
            X = None
            y = None
            gc.collect()

    return model, mlb, scaler

# concepts_all = list(concept_extract())
# def train_model(X,y):
#     mlb = MultiLabelBinarizer(classes=all_page_ids)
#     y = mlb.fit_transform(y)
#     print("binarised, shape is ", y.shape)
#
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     print("fitted scaler, scaled")
#
#     model = make_model(len(X[0]), len(y[0]))
#     print("made model")
#     es = callbacks.EarlyStopping(monitor='val_loss',
#                                   min_delta=0,
#                                   patience=1,
#                                   verbose=0, mode='auto')
#     print("fitting")
#     model.fit(X,y, validation_split=0.2, batch_size=256, epochs=10, shuffle=True, callbacks=[es])
#     print("fitted")
#     return model, mlb, scaler

def evaluate_predictions(tt, mlb): #, test_meta):
    fout = open(base + "hw_gobbler.out", "w")
    X,y = next(xy_generator(tt, batch_size=-1))
    y_preds = model.predict(X, verbose=True)
    X = None

    for ct, tt_a in enumerate(tt): # for each test assigment
        #get the student-level profiles
        ts = tt_a[0]
        board = tt_a[1]
        grp = tt_a[2]
        psi_list = tt_a[4]
        true_hexes = tt_a[3]
        # true_binsd = mlb.transform(true_hexes)

        ensemble_sum = None
        for offset, student in enumerate(psi_list):
            y_pred = y_preds[ct+offset]
            # y_pred = (y_preds[ct+offset] >= 0.5).astype(int)
            ensemble_sum = y_pred if ensemble_sum is None else ensemble_sum + y_pred # keep running total

        yixs = numpy.argsort(ensemble_sum)
        reversed(yixs)
        pred_hexes = []
        pred_levs = []
        pred_concepts = set()
        true_levs = []
        true_concepts = set()
        for ix, t_hx_name in enumerate(true_hexes):
            hx_ix = yixs[ix]
            hx_name = all_page_ids[hx_ix]
            pred_hexes.append(hx_name)
            pred_levs.append( lev_page_lookup[hx_name] if hx_name in lev_page_lookup else "UNK" )
            pred_concepts.update( con_page_lookup[hx_name] if hx_name in con_page_lookup else "UNK" )
            true_levs.append( lev_page_lookup[t_hx_name] if t_hx_name in lev_page_lookup else "UNK")
            true_concepts.update( con_page_lookup[t_hx_name] if t_hx_name in con_page_lookup else "UNK" )

        fout.write("\n# # # # # # EL NEXTO\n")
        # fout.write("{}\n".format(true_binsd))
        # fout.write("- - - -- \n")
        numpy.set_printoptions(threshold=numpy.nan)
        # fout.write("{}\n".format(ensemble_sum))
        fout.write("{}, sums to ={}\n".format(ensemble_sum.shape, ensemble_sum.sum()))
        fout.write("{} {} {}\n".format(ts,grp,board))
        fout.write("{} :{}\n".format(len(psi_list), psi_list))
        fout.write("...\n")
        fout.write("{}\n".format(true_hexes))
        fout.write("{}\n".format(true_levs))
        fout.write("{}\n".format(true_concepts))
        fout.write("v v v v v\n")
        fout.write("{}\n".format(pred_hexes))
        fout.write("{}\n".format(pred_levs))
        fout.write("{}\n".format(pred_concepts))

        score = 0
        for hx in pred_hexes:
            if hx in true_hexes:
                score +=1
        fout.write("Score = {}\n".format(score))
        fout.flush()
        # pred_hexes = [all_page_ids[yixs[ix]] for ix in range(len(true_hexes))]
        # print(true_hexes)
        # print("- vs -")
        # print(pred_hexes)
        # print("\n~ ~ ~EL NEXTO~ ~ ~\n")
    fout.close()

os.nice(3)


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
print("Splitting...")
# X, y, test_X, test_y, test_meta = convert_to_Xy(asses, split=0.10)
# del(asses) # TODO helps?

ass_n = 1000
shuffle(asses)
asses = asses[0:ass_n] # train on a subset of the data
#split = int(0.2 * len(asses))

split = 100
random.seed(666)
tt = asses[0:split]
tr = asses[split:]
print("SPlat!")

do_train = True
do_testing = True

model, xscaler = train_autoencoder(tr)
print("evaluating")
for tX, ty in xy_generator(tt, batch_size=-1):
    tX = xscaler.transform(tX)
    eval = model.evaluate(tX,tX)
    predX = model.predict(tX)

    for ttX, pX in zip(tX, predX):
        diff = ttX - pX
        print(xscaler.inverse_transform(ttX))
        print(xscaler.inverse_transform(pX))
        print((diff**2).mean())
    print(eval)
exit()

if do_train:
    model, mlb, scaler = train_model2(tr)
    print("...deleted original X,y")
    model.save(base + 'hwg_model.hd5')
    joblib.dump(mlb, base + 'hwg_mlb.pkl')
    joblib.dump(scaler, base + 'hwg_scaler.pkl')

if do_testing:
    if model is None:
        model = load_model(base + "hwg_model.hd5")
        mlb = joblib.load(base + 'hwg_mlb.pkl')
        scaler = joblib.load(base + 'hwg_scaler.pkl')
    evaluate_predictions(tt, mlb)
    print("done")