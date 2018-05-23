import gc
import os
import pickle
import random
import zlib
from random import seed, shuffle
from statistics import mean

import numpy
from keras import callbacks, Input, Model
from keras.layers import Dense, Dropout
from keras.models import load_model, Sequential
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from hwgen.common import init_objects, get_meta_data, get_n_hot, split_assts, jaccard, get_page_concepts
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.model import make_mixed_loss_model, LSTM_model

base = "../../../isaac_data_files/"

print("Initialising deep wmeta learning HWGen....")

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

def xy_generator(assts,batch_size=512):
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
    for a in assts:
        profiles = pickle.loads(zlib.decompress(a[7]))
        hexagons = a[3]
        _concepts = set()
        _levels = []
        # qns_list = [s for s in all_qids if any(s.startswith(hx) for hx in hexagons)]

        if profiles is None or len(profiles)==0:
            print("skipping empty profiles object")
            continue

        # topics = []
        # _levels = []
        # _concepts = []
        # for hx in hexagons:
        #     if "|" in hx:
        #         hexagons.remove(hx)
        #         hx = hx.split("|")[0]
        #         hexagons.append(hx)
        #     topics.append(cat_page_lookup[hx])
        #     _levels.append(lev_page_lookup[hx])
        #     _concepts.update(con_page_lookup[hx])

        # if len(_concepts) == 0:
        #     emcons +=1
        #     continue

        for psi in profiles:  # set up the training arrays here
            s,x,elle = profiles[psi] #TODO add L vector here!
            #s = numpy.concatenate((s,elle))
            s = [0]
            # s = numpy.concatenate((s,elle))
            for hx in hexagons: #Split the assignment up by hexagon
                hx = hx.split("|")[0]
                S.append(s) #TODO use indices 0:2 to keep just student AGE and XP
                X.append(x)
                y.append([hx])
                y_cs.append(con_page_lookup[hx])
                y_ts.append(cat_page_lookup[hx])
                tvolume = len(hexagons) #TODO wld prefer qn vol here but try hex volume for now
                # tvolume = 0
                # for tp in hexagons:
                #     tqids = [qid for qid in all_qids if qid.startswith(tp)]
                #     tvolume += len(tqids)

                y_lv.append([lev_page_lookup[hx]])
                y_v.append((tvolume,))

        c += 1

        n = len(X)
        # print("gen'd {} items".format(n))
        if (batch_size>0) and (n >= batch_size):
            print("b={}, n samples = {} ({}/{}={:.1f}%)".format(b,len(X), c, lenX, (100.0*c/lenX)))
            b+=1
            yield S,X,y,y_cs, y_ts, y_lv, y_v
            S = []
            X = []
            y = []
            y_cs = []
            y_ts = []
            y_lv = []
            y_v = []
    print("out of assts")
    print("empty concepts = {} of {}".format(emcons,lenX))
    yield S,X,y,y_cs, y_ts, y_lv, y_v


def make_wmeta_model(n_Q,n_P, n_C, n_T, n_L):
    # this is our input placeholder
    input_Q = Input(shape=(n_Q,), name="wide_input")

    w = 256

    # hidden = Dense(5000, activation='relu')(input_Q)
    # hidden = Dropout(.2)(hidden)

    hidden = Dense(w, activation='relu')(input_Q)
    hidden = Dropout(.2)(hidden)

    hidden = Dense(w, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)

    hidden = Dense(w, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)

    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)

    # decode_test = Dense(n_Q, activation="sigmoid", name="decode_test")(hidden)
    # hidden = Dense(w, activation='relu')(concatenate([input_S, hidden]))
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # hidden = Dense(1000, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
    # hidden = Dense(w, activation='relu')(hidden)
    # hidden = Dropout(.2)(hidden)
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
    next_page = Dense(n_P, activation='softmax', name="next_page")(hidden)
    next_concepts = Dense(n_C, activation='sigmoid', name="next_concepts")(hidden)
    next_topic = Dense(n_T, activation='softmax', name="next_topic")(hidden)
    next_level = Dense(n_L, activation='relu', name="next_level")(hidden)
    m = Model(inputs=input_Q, outputs=[next_page, next_concepts, next_topic, next_level])
    m.compile(optimizer="rmsprop", loss=['categorical_crossentropy','binary_crossentropy','categorical_crossentropy','mse'], loss_weights=[1,1,.1,.1])
    m.summary()
    return m


# concepts_all = list(concept_extract())
numpy.set_printoptions(threshold=numpy.nan)
def train_wmeta_model(assts, n_macroepochs=100, n_epochs=10):
    model = None
    mlb = None
    mlbc = None
    mlbt = None
    # sscaler = None
    # levscaler = None
    # volscaler = None
    # scaler = StandardScaler()
    # sscaler = StandardScaler()
    # levscaler = StandardScaler()
    # volscaler = StandardScaler()

    # print("fitting scaler...")
    # sgen = xy_generator(assts, batch_size=5000)
    # for S,X,_,_,_,ylv,yv in sgen:
        # print("S=",S)
        # sscaler.partial_fit(S)
        # scaler.partial_fit(X)
        # levscaler.partial_fit(ylv)
        # volscaler.partial_fit(yv)
    # sgen = None
    # print("fat")

    nb_epoch = n_macroepochs
    for e in range(nb_epoch):
        xygen = xy_generator(assts, batch_size=60000)  # make generator object
        print("macroepoch %d of %d" % (e, nb_epoch))
        for S,X,y,y_cs, y_ts, y_lv, y_v in xygen:
            X = numpy.array(X, dtype=numpy.int8)
            # y = numpy.array(y)

            # c2 = []
            # for cel in y_cs
            #     c2.append(get_n_hot(cel, concepts_all))
            # y_cs = numpy.array(c2, dtype=numpy.int8)

            c2=[]
            for hx in y:
                assert len(hx)==1
                hx = hx[0] # get first/only item
                concept_list = con_page_lookup[hx] #get_page_concepts(hx)
                nhot=get_n_hot(concept_list, concepts_all)
                assert numpy.sum(nhot) == len(concept_list)
                c2.append(nhot)
            y_cs = numpy.array(c2, dtype=numpy.int8)

            t2 = []
            for tel in y_ts:
                # print(tel)
                nhot = get_n_hot(tel, cats)
                assert sum(nhot) == 1

                t2.append(nhot)
            y_ts = numpy.array(t2, dtype=numpy.int8)

            y_lv = numpy.array(y_lv)
            # y_v = numpy.array(y_v)

            print(y[0])
            assert len(y[0])==1 # each line shd have just one hex assignment
            y2 = []
            print("binarising y")
            for yel in y:
                y2.append( get_n_hot(yel, all_page_ids) )
            y = numpy.array(y2, dtype=numpy.int8)
            print("binarised, shape is ", y.shape)
            # assert numpy.sum(y, axis=1) == 1 # Each line should be one-hot

            assert numpy.all((numpy.sum(y,axis=1) == 1), axis=0)

            es = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=2,
                                         verbose=1, mode='auto')
            if model is None:
                print("dims>>>")
                print(len(X[0]), len(y[0]), len(y_cs[0]), len(y_ts[0]), len(y_lv[0]))
                model = make_wmeta_model(len(X[0]), len(y[0]), len(concepts_all), len(cats), len(y_lv[0]))

            model.fit(X,[y, y_cs, y_ts, y_lv], epochs=n_epochs, shuffle=True, batch_size=64, callbacks=[es], validation_split=0.2)

            X=y=yc=lv = None
            gc.collect()

    return model, mlb, None, None #, sscaler, levscaler, volscaler


def get_top_k_hot(raw, k):
    args = list(reversed(raw.argsort()))[0:k]
    print(args)
    k = numpy.zeros(len(raw))
    k[args] = 1.0
    return k, args

def evaluate_wmeta_loss(tt, model, ylb): #, sscaler,levscaler,volscaler): #, test_meta):
    print("ready to evaluate...")
    num_direct_hits =0
    num_cases = 0
    X_sums = []
    for _,X,y in xy_generator(tt, batch_size=10000):
        print("batch")
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


os.nice(3)

model = None
print("loading...")
with open(asst_fname, 'rb') as f:
    asses = pickle.load(f)

print("loaded {} assignments".format(len(asses)))

do_train = True
do_testing = True
ass_n = 1500
split = 500
n_macroepochs = 1
n_epochs = 10
ass_keys = list(asses.keys()) # train on a subset of the data
random.seed(666)
print("first 10 keys preshuff = ",ass_keys[0:10])
shuffle(ass_keys)
print("first 10 keys postshuff = ",ass_keys[0:10])

assert_profile_len = False
if assert_profile_len:
    for k in ass_keys:
        ass = asses[k]
        students = ass[6]
        profiles = pickle.loads(zlib.decompress(ass[7]))
        assert len(students)==len(profiles)

BOOK_ONLY=False
print("Splitting...")
tr,tt = split_assts(asses, ass_n, split, BOOK_ONLY)
del asses
gc.collect()
print("Split complete!")
print("{} {}".format(len(tt), len(tr)))

if do_train:
    print("training")
    model, mlb, mlbc, mlbt = train_wmeta_model(tr, n_macroepochs, n_epochs)
    print("...deleted original X,y")
    model.save(base + 'hwg_model.hd5')
    joblib.dump((mlb, mlbc, mlbt), base + 'hwg_mlb.pkl')
    # joblib.dump((sscaler,levscaler,volscaler), base + 'hwg_scaler.pkl')

if do_testing:
    print("testing")
    if model is None:
        model = load_model(base + "hwg_model.hd5")
        (mlb, mlbc, mlbt) = joblib.load(base + 'hwg_mlb.pkl')
        #(sscaler,levscaler,volscaler) = joblib.load(base + 'hwg_scaler.pkl')
    # evaluate_predictions(tt, model, scaler, sscaler)
    evaluate_wmeta_loss(tt, model, mlb) #, sscaler,levscaler,volscaler)
    print("DEEP wmeta testing done")