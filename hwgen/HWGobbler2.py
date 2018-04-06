import os
from collections import Counter
from pickle import load
from random import shuffle

import numpy
import pandas
from keras import Sequential
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
ass_n = 10000
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users)

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
    for psi in student_list:
        age = 0
        if psi in profile_df.index:
            dob = profile_df.loc[psi, "date_of_birth"]
            # print("dob", dob, type(dob))
            dob = pandas.to_datetime(dob)
            if isinstance(dob, pandas.Timestamp):
                age = (up_to_ts - dob).days / 365.242
            else:
                age = 0
        # profile student at time of assignment
        pf = profile_student(psi, age, up_to_ts, cats, cat_lookup, cat_ixs, levels, concepts_all, hwdf, user_cache)
        if (pf):  # ie. if not empty ... TODO why do we get empty ones??
            profiles[psi] = pf
            print(psi, pf)
        # train softmax classifier with student profile and assignment profile
    return profiles

asst_fname = base+"assignments.pkl"
def make_data():
    ass_df = pandas.read_csv(base + "gb_assignments.csv")
    ass_df = ass_df.iloc[27000:, :]
    sprofs = pandas.read_csv(base + "student_profiling/users_all.csv")
    sprofs["date_of_birth"] = pandas.to_datetime(sprofs["date_of_birth"])
    gb_qmap = make_gb_question_map()
    asses = []
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
    joblib.dump(asses, asst_fname)
    return asses

def make_model(n_in, n_out):
    Sequential()
    model.add(Dense(n_out, activation='relu', input_dim=n_in))
    model.add(Dropout(0.5))
    model.add(Dense(n_out, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(n_in, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(n_out, activation='sigmoid'))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # optimiser = Adam(0.0002, 0.5)
    optimiser = Adam()
    model.compile(loss="binary_crossentropy", optimizer=optimiser, metrics=["mse", "binary_crossentropy"])

def convert_to_Xy(assts, split=None):
    if(split):
        shuffle(assts)
        tr = assts[0:split]
        tt = assts[split:]
    for set in [tr,tt]:
        for a in set:
            profiles = a[4]
            qarray = a[5]
            this_qns = a[3]
            this_cons = set()
            for q in this_qns:
                if q in con_page_lookup:
                    this_cons.update(con_page_lookup[q])
            for psi in profiles:  # set up the training arrays here
                X.append(profiles[psi])
                y.append(this_cons)
        yield X,y


# model = None
# if do_train:
#     tr = load(open(trdump_fname,"rb"))
#     X = []
#     y = []
#     qs = []
#     print("profiling")
#
#     n_in= len(X[0])
#     n_out= len(all_qids)
#     tr=None #don't need this anymore
#     model = make_model(n_in, n_out)
#
#     print("scaling")
#     mlb = MultiLabelBinarizer(classes=concepts_all)
#     y = mlb.fit_transform(y)
#     print(y.shape)
#     # input("prompt")
#     scaler = StandardScaler()
#     training_X = scaler.fit_transform(X)
#     print("done")
#
#     print("fitting")
#     model.fit(X, y)
#     print("done")
#     print("saving model and scaler")
#     model.save(base+"hwg_model.hd5")
#     #joblib.dump(model, base+'filename.pkl')
#     joblib.dump(scaler, base+'hwg_scaler.pkl')
#     joblib.dump(mlb, base+'hwg_mlb.pkl')
#     print("done")



def train_model(X,y):
    model = make_model(len(X[0]), len(y[0]))
    model.fit(X,y)

def evaluate_predictions(pred_y, y):
    fout = open(base + "hw_gobbler.out", "w")
    preds = model.predict(X)
    topcats = set()
    topcons = set()
    cats = set()
    cons = set()
    levs = []
    fout.write("- - - - -\n")

    for v in y:
        if v==1:
            qid=all_qids[v]
            cons.update(con_page_lookup[qid] if qid in con_page_lookup else "UNK_CONCEPT")
            cats.add(cat_page_lookup[qid] if qid in cat_page_lookup else "UNK_TOPIC")
            levs.append(lev_page_lookup[qid] if qid in cat_page_lookup else "UNK_LEV")

    vote_ct = Counter()
    for predix, pred in enumerate(preds):
        print("pred", pred)
        for ix, i in enumerate(pred):
            qid = all_qids[ix]
            vote_ct[qid] += pred[ix]

    fout.write(str(cats) + "\n")
    fout.write(str(cons) + "\n")
    fout.write(str(levs) + "\n")

    fout.write("> > > > >\n")
    topcats = set()
    topcons = set()
    toplevs = []
    top = vote_ct.most_common(10)
    for qid, cnt in top:
        topcons.update(con_page_lookup[qid] if qid in con_page_lookup else "UNK_CONCEPT")
        topcats.add(cat_page_lookup[qid] if qid in cat_page_lookup else "UNK_TOPIC")
        toplevs.append(lev_page_lookup[qid] if qid in cat_page_lookup else "UNK_LEV")
    fout.write(str(top) + "\n")
    fout.write(str(topcats) + "\n")
    fout.write(str(topcons) + "\n")
    fout.write(str(toplevs) + "\n")

    catsj = jaccard_score(cats, topcats)
    consj = jaccard_score(cons, topcons)
    # levsj = jaccard_score(levs, toplevs)
    bag_levs = Counter(levs)
    bag_toplevs = Counter(toplevs)
    fout.write("{}, {} {}".format(bag_levs, bag_toplevs, bag_levs & bag_toplevs))
    fout.write("{}, {} {}".format(bag_levs, bag_toplevs, bag_levs | bag_toplevs))
    levsj = sum((bag_levs & bag_toplevs).values()) / sum((bag_levs | bag_toplevs).values())

    fout.write("\nJaccard indices: categories {}  concepts {}  levels{}\n\n".format(catsj, consj, levsj))



cat_page_lookup = {}
for k in cat_lookup.keys():
    qpage = k.split("~")[0]
    qsft = cat_lookup[k]
    if qpage not in cat_page_lookup:
        cat_page_lookup[qpage] = qsft

con_page_lookup = page_to_concept_map()

lev_page_lookup = {}
for k in levels.keys():
    qpage = k.split("~")[0]
    L = levels[k]
    if qpage not in lev_page_lookup:
        lev_page_lookup[qpage] = L

#get assignments ....
all_qids = list(pandas.read_csv(base+"all_pids.csv")["page_id"])

gbd_map = {}
gbd_name_map = {}

concepts_all = list(concept_extract())
trdump_fname = base + "hw_trfile.pkl"
ttdump_fname = base + "hw_ttfile.pkl"

ct_tried = 0
ct_no_ev_dets = 0
ct_missing_gbid = 0
# if data_gen:
#     # atypes = pandas.read_csv(base + "atypes.csv", header=None)
#     training_X = []
#     training_y = []
#     # for each assignment
#
#
#
#     print("no assts tried=",ct_tried)
#     print("no event details=", ct_no_ev_dets)
#     print("missing gb id=", ct_missing_gbid)
#     # input("prompt")

os.nice(10)



data_gen = True
do_train = True
do_testing = False

asses=None
if __name__=="__main__":
    if data_gen:
        asses = make_data()

    if asses is None:
        asses = load(open(asst_fname, "rb"))

    X,y, test_X,test_y = convert_to_Xy(asses)

    if do_train:
        model, mlb, scaler = train_model(X,y)

    if do_testing:
        if model is None:
            model = load_model(base + "hwg_model.hd5")
            mlb = joblib.load(base + 'hwg_mlb.pkl')
            scaler = joblib.load(base + 'hwg_scaler.pkl')
        predictions = model.predict(test_X)
        evaluate_predictions(predictions, test_y)

