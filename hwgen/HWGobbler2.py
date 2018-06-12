import os
from collections import Counter
from datetime import datetime
from math import isnan
from pickle import load
from pprint import pprint
from random import shuffle, seed

import numpy
import pandas
from keras import Sequential, Input, callbacks
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from backfit.BackfitUtils import init_objects
from hwgen.concept_extract import concept_extract, page_to_concept_map
from hwgen.profiler import profile_student
from utils.utils import jaccard_score

base = "../../../isaac_data_files/"

#need to build a softmax classifier to recommend questions...
#this would have qn output nodes

n_users = -1
cats, cat_lookup, _, users, diffs, levels, cat_ixs = init_objects(n_users)


def make_gb_question_map():
    gbd_df = pandas.read_csv(base + "gameboards.txt", sep="~")
    map = {}
    for gb_id, item in zip(gbd_df["id"], gbd_df["questions"]):
        if str is not type(item):
            continue
        # print(gb_id)
        # print(item)
        item = item[1:-1]
        item = item.split(",")
        map[gb_id] = item
    return map

grp_df = pandas.read_csv(base + "group_memberships.csv")
def get_students_in_group(gr_id):
    return list(grp_df[grp_df["group_id"] == gr_id]["user_id"])


hwdf = pandas.read_csv(base + "hwgen1.csv", index_col=0, header=0)
def profile_students(student_list, profile_df, up_to_ts, user_cache):
    profiles = {}
    if not student_list:
        return profiles

    genesis = pandas.to_datetime("1970-01-01")
    sys_mean_dob_del =  (profile_df[(profile_df.role=="STUDENT")]["date_of_birth"] - genesis).mean()

    profile_df = profile_df[(profile_df.id.isin(student_list))]
    dobs = profile_df["date_of_birth"]

    dob_dels = dobs - genesis
    mean_del = dob_dels.mean()
    if (pandas.isnull(mean_del)):
        print("nan mean dob, use system mean")
        mean_dob = sys_mean_dob_del + genesis
    else:
        mean_dob = mean_del + genesis

    for psi in student_list:
        if(profile_df[profile_df.id==psi]["date_of_birth"]).shape[0]==0:
            continue
        dob = profile_df[profile_df.id==psi]["date_of_birth"].iloc[0]
        if not isinstance(dob, pandas.Timestamp):
            dob = mean_dob
        age = (up_to_ts - dob).days / 365.242
        if numpy.isnan(age):
            age = -1
        # profile student at time of assignment
        pf = profile_student(psi, age, up_to_ts, cats, cat_lookup, cat_ixs, levels, concepts_all, hwdf, user_cache)
        if (pf):  # ie. if not empty ... TODO why do we get empty ones??
            profiles[psi] = pf
            print(psi, pf)
        # train softmax classifier with student profile and assignment profile
    return profiles

asst_fname = base+"assignments.pkl"
all_qids = list(pandas.read_csv(base+"all_pids.csv")["page_id"])
def make_data(ass_n):
    asses = []
    ass_df = pandas.read_csv(base + "gb_assignments.csv")
    # ass_df = ass_df.iloc[27000:, :]
    sprofs = pandas.read_csv(base + "student_profiling/users_all.csv")
    sprofs["date_of_birth"] = pandas.to_datetime(sprofs["date_of_birth"])
    gb_qmap = make_gb_question_map()
    ass_ct =0

    ass_df["timestamp"] = pandas.to_datetime(ass_df["timestamp"])
    ass_df = ass_df[ass_df.event_details!="{}"]
    ass_df["event_details"] = ass_df["event_details"].replace("0L,", "0,")
    user_cache = {}
    for ass in ass_df.iterrows():
        if 0 < ass_n < ass_ct:
            break
        print("assct {} of {} ({} users cached)".format(ass_ct, ass_n, len(user_cache)))
        ts = ass[1]['timestamp']
        # print(ts)
        event_details = eval(ass[1]['event_details'])
        gb_id = event_details["gameboardId"]
        if gb_id not in gb_qmap:
            print("gb id unknown")
            continue
        this_qns = gb_qmap[gb_id]

        ass_ct += 1

        gr_id = event_details["groupId"]
        students = get_students_in_group(gr_id)
        profiles = profile_students(students, sprofs, ts, user_cache)

        qarray = [0]*len(all_qids)
        for qid in this_qns:
            if qid in all_qids:
                ix = all_qids.index(qid)
                qarray[ix] = 1

        ass_entry = (ts, gb_id, gr_id, this_qns, profiles, qarray)
        asses.append(ass_entry)
        print("...{} students".format(len(profiles)))
    print("dumping")
    joblib.dump(asses, asst_fname)
    print("dumped")
    return asses

def make_model(n_in, n_out):
    print(n_in, n_out)
    mode="ML_SOFTMAX"

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
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(8*n_out, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(8*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8*n_out, activation='relu'))
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
        model.add(Dense(8*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
    return model

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
            profiles = a[4]
            this_qns = a[3]
            this_cons = set()
            for q in this_qns:
                if q in con_page_lookup:
                    this_cons.update(con_page_lookup[q])
            for psi in profiles:  # set up the training arrays here
                X.append(profiles[psi])
                y.append(this_cons)
        ret.append(X)
        ret.append(y)
    return ret

concepts_all = list(concept_extract())
def train_model(X,y):
    mlb = MultiLabelBinarizer(classes=concepts_all)
    y = mlb.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = make_model(len(X[0]), len(y[0]))
    es = callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=1,
                                  verbose=0, mode='auto')
    model.fit(X,y, validation_split=0.2, batch_size=256, epochs=10, shuffle=True, callbacks=[es])
    return model, mlb, scaler

def evaluate_predictions(preds, y):
    fout = open(base + "hw_gobbler.out", "w")
    print(len(preds[0]))

    scores = []
    for spoof, sooth in zip(preds, y):
        if len(list(sooth))==0: #TODO skip empty concept-lists for now, but will need a secondary way to deal with these!
            continue

        fout.write("{}\n".format(str(list(sooth))))
        ixs = (-spoof).argsort()[:len(sooth)] # get the top suggestions
        print(ixs)
        recomm_concepts = [concepts_all[ix] for ix in ixs]
        print("*",sooth)
        print(">",recomm_concepts)
        fout.write("{}\n".format( str(recomm_concepts)))

        score = 0
        for c in sooth:
            if c in recomm_concepts:
                score += 1
        score = score / len(sooth)
        fout.write("Score {}\n".format(score))
        scores.append(score)  # hit rate for this prediction
    avgscore = numpy.mean(scores)
    fout.write("***\nAvg Score {}\n".format(avgscore))
    print("Avg Score {}".format(avgscore))


    # catsj = jaccard_score(cats, topcats)
    # consj = jaccard_score(cons, topcons)
    # # levsj = jaccard_score(levs, toplevs)
    # bag_levs = Counter(levs)
    # bag_toplevs = Counter(toplevs)
    # fout.write("{}, {} {}".format(bag_levs, bag_toplevs, bag_levs & bag_toplevs))
    # fout.write("{}, {} {}".format(bag_levs, bag_toplevs, bag_levs | bag_toplevs))
    # levsj = sum((bag_levs & bag_toplevs).values()) / sum((bag_levs | bag_toplevs).values())
    #
    # fout.write("\nJaccard indices: categories {}  concepts {}  levels{}\n\n".format(catsj, consj, levsj))
    fout.close()


cat_page_lookup = {}
for k in cat_lookup.keys():
    qpage = k.split("~")[0]
    qsft = cat_lookup[k]
    if qpage not in cat_page_lookup:
        cat_page_lookup[qpage] = qsft


lev_page_lookup = {}
for k in levels.keys():
    qpage = k.split("~")[0]
    L = levels[k]
    if qpage not in lev_page_lookup:
        lev_page_lookup[qpage] = L

#get assignments ....




os.nice(3)

ass_n = 10000 # the number of SUCCESSFUL (i.e. complete) assignments to process # incomplete assts are skipped and do not count
data_gen = False
do_train = False
do_testing = True

asses=None
model=None
if __name__=="__main__":
    if data_gen:
        asses = make_data(ass_n)

    if asses is None:
        asses = load(open(asst_fname, "rb"))

    X,y, test_X,test_y = convert_to_Xy(asses, split=0.10)

    if do_train:
        model, mlb, scaler = train_model(X,y)
        model.save(base + 'hwg_model.hd5')
        joblib.dump(mlb, base + 'hwg_mlb.pkl')
        joblib.dump(scaler, base + 'hwg_scaler.pkl')

    if do_testing:
        if model is None:
            model = load_model(base + "hwg_model.hd5")
            mlb = joblib.load(base + 'hwg_mlb.pkl')
            scaler = joblib.load(base + 'hwg_scaler.pkl')
        predictions = model.predict(test_X)
        evaluate_predictions(predictions, test_y)

