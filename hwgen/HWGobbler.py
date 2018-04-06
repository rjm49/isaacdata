import os
from collections import Counter
from pickle import load

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


data_gen = True
do_train = True
do_testing = True

n_users = -1
cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users)

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
if data_gen:
    ass_df = pandas.read_csv(base + "gb_assignments.csv")
    gbd_df = pandas.read_csv(base + "gameboards.txt", sep="~")
    grp_df = pandas.read_csv(base + "group_memberships.csv")
    atypes = pandas.read_csv(base + "atypes.csv", header=None)
    sprofs = pandas.read_csv(base + "student_profiling/users_all.csv")
    sprofs["date_of_birth"] = pandas.to_datetime(sprofs["date_of_birth"])

    ass_df = ass_df.iloc[27000:, :]
    print(ass_df.shape[0])
    ass_n = 10000
    ass_ct = 0

    questions = {}
    for gb_id, item in zip(gbd_df["id"], gbd_df["questions"]):
        if str is not type(item):
            continue
        # print(gb_id)
        # print(item)
        item = item[1:-1]
        item = item.split(",")
        questions[gb_id] = item

    hwdf = pandas.read_csv(base + "hwgen1.csv", index_col=0, header=0)
    user_cache = {}
    training_X = []
    training_y = []
    # for each assignment
    asses = []
    for assix, ass in enumerate(ass_df.iterrows()):
        if 0 < ass_n < ass_ct:
            break
        ct_tried+=1
        print("assix {} of {}".format(ass_ct, ass_n))
        ts = pandas.to_datetime(ass[1]['timestamp'] )
        # print(ts)
        event_details = eval(ass[1]['event_details'].replace("0L,","0,"))
        # print(type(event_details))
        if not event_details:
            ct_no_ev_dets+=1
            continue
        # print(event_details.keys())
        gb_id = event_details["gameboardId"]
        if gb_id not in questions:
            ct_missing_gbid+=1
            continue

        ass_ct += 1
        this_qns = questions[gb_id]

        gr_id = event_details["groupId"]
        students = list(grp_df[grp_df["group_id"]==gr_id]["user_id"])

        # for each student in class
        profiles = {}
        dobs = []

        # student_df = sprofs[sprofs["id"].isin(students)]
        # print(student_df)
        # # notnull_df = student_df[pandas.notnull(student_df["date_of_birth"])]
        # student_df["age"] = ts - student_df["date_of_birth"]
        # print(student_df['age'])

        # print(notnull_df)
        # dob_series = notnull_df["date_of_birth"]
        # min_dob = dob_series.min()
        # del_series = dob_series - min_dob
        # mean_del = (del_series).mean()
        # mean_dob = min_dob + mean_del
        # print(dob_series)

        for psi in students:
            age = 0
            if psi in sprofs.index:
                dob = sprofs.loc[psi,"date_of_birth"]
                # print("dob", dob, type(dob))
                dob = pandas.to_datetime(dob)\

                if isinstance(dob, pandas.Timestamp):
                    age = (ts-dob).days / 365.242
                else:
                    age=0

            #profile student at time of assignment

            pf = profile_student(psi, age, ts, cats, cat_lookup, cat_ixs, levels, concepts_all, hwdf, user_cache)
            if(pf): #ie. if not empty ... TODO why do we get empty ones??
                profiles[psi] = pf
                print(psi, pf)
            #train softmax classifier with student profile and assignment profile

        qarray = [0]*len(all_qids)
        for qid in this_qns:
            if qid in all_qids:
                ix = all_qids.index(qid)
                qarray[ix] = 1

        ass_entry = (ts, gb_id, gr_id, this_qns, profiles, qarray)
        asses.append(ass_entry)
        print("...{} students".format(len(profiles)))


    tixes = range(len(asses))
    trixes, ttixes = train_test_split(tixes, train_size=0.90)

    # tr,tt = train_test_split(asses, train_size=0.90)
    joblib.dump([asses[i] for i in ttixes], ttdump_fname)
    joblib.dump([asses[i] for i in trixes], trdump_fname)

    print("no assts tried=",ct_tried)
    print("no assts selects=", len(asses))
    print("no event details=", ct_no_ev_dets)
    print("missing gb id=", ct_missing_gbid)
    # input("prompt")

os.nice(10)

model = None
if do_train:
    # estr = LogisticRegression(verbose=1)
    # model = OneVsRestClassifier(estr, n_jobs=4)
    # model = RandomForestClassifier(n_jobs=4 )

    print("loading data")
    tr = load(open(trdump_fname,"rb"))
    # tr = tr[0:5000]
    print("done")

    X = []
    y = []
    qs = []
    print("profiling")
    for a in tr:
        profiles = a[4]
        qarray = a[5]
        this_qns = a[3]
        this_cons = set()
        for q in this_qns:
            if q in con_page_lookup:
                this_cons.update(con_page_lookup[q])

        for psi in profiles:  # set up the training arrays here
            # training_X.append(numpy.array(profiles[psi]).reshape(-1,nin))
            # training_y.append(numpy.array(qarray).reshape(-1,nout))
            X.append(profiles[psi])
            y.append(this_cons)
    print("done")

    n_in= len(X[0])
    n_out= len(all_qids)
    tr=None #don't need this anymore
    model = Sequential()
    model.add(Dense(n_out, activation='relu', input_dim=n_in))
    model.add(Dropout(0.5))
    model.add(Dense(n_out, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(n_in, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(n_out, activation='sigmoid'))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # optimiser = Adam(0.0002, 0.5)
    optimiser = Adam()
    model.compile(loss="binary_crossentropy", optimizer=optimiser, metrics=["mse","binary_crossentropy"])


    print("scaling")
    mlb = MultiLabelBinarizer(classes=concepts_all)
    y = mlb.fit_transform(y)
    print(y.shape)
    # input("prompt")
    scaler = StandardScaler()
    training_X = scaler.fit_transform(X)
    print("done")

    print("fitting")
    model.fit(X, y)
    print("done")
    print("saving model and scaler")
    model.save(base+"hwg_model.hd5")
    #joblib.dump(model, base+'filename.pkl')
    joblib.dump(scaler, base+'hwg_scaler.pkl')
    joblib.dump(mlb, base+'hwg_mlb.pkl')
    print("done")

if not do_testing:
    exit()
#Testing!
if model is None:
    print("loading hd5 model")
    model = load_model(base+"hwg_model.hd5")
    #model = joblib.load(base+'filename.pkl')
    mlb = joblib.load(base + 'hwg_mlb.pkl')
    scaler = joblib.load(base + 'hwg_scaler.pkl')
tt = load(open(ttdump_fname,"rb"))
# tt = tt[0:20]

fout = open(base+"hw_gobbler.out", "w")
print(len(tt))
# input("prompt")
for a in tt:
    print(a)
    X = []
    y = []
    qns = a[3]
    this_cons = set()
    for q in this_qns:
        if q in con_page_lookup:
            this_cons.update(con_page_lookup[q])

    profiles = a[4]
    if not profiles:
        print("no profiles")
        continue

    for psi in profiles:  # set up the training arrays here
        # training_X.append(numpy.array(profiles[psi]).reshape(-1,nin))
        # training_y.append(numpy.array(qarray).reshape(-1,nout))
        X.append(profiles[psi])
        y.append(this_cons)

    y = mlb.transform(y) # convert to indicators here TODO tidy up custom binarisation throughout
    X = scaler.transform(X)
    X = numpy.asarray(X)
    preds = model.predict(X)
    topcats = set()
    topcons = set()
    cats = set()
    cons = set()
    levs = []
    fout.write("- - - - -\n")

    fout.write(str(qns)+"\n")
    for qid in qns:
        cons.update(con_page_lookup[qid] if qid in con_page_lookup else "UNK_CONCEPT")
        cats.add(cat_page_lookup[qid] if qid in cat_page_lookup else "UNK_TOPIC")
        levs.append(lev_page_lookup[qid] if qid in cat_page_lookup else "UNK_LEV")

    vote_ct = Counter()
    for predix,pred in enumerate(preds):
        print("pred", pred)
        for ix,i in enumerate(pred):
            qid = all_qids[ix]
            vote_ct[qid]+= pred[ix]

    fout.write(str(cats)+"\n")
    fout.write(str(cons)+"\n")
    fout.write(str(levs)+"\n")

    fout.write("> > > > >\n")
    topcats = set()
    topcons = set()
    toplevs = []
    top = vote_ct.most_common(10)
    for qid,cnt in top:
        topcons.update(con_page_lookup[qid] if qid in con_page_lookup else "UNK_CONCEPT")
        topcats.add(cat_page_lookup[qid] if qid in cat_page_lookup else "UNK_TOPIC")
        toplevs.append(lev_page_lookup[qid] if qid in cat_page_lookup else "UNK_LEV")
    fout.write(str(top)+"\n")
    fout.write(str(topcats)+"\n")
    fout.write(str(topcons)+"\n")
    fout.write(str(toplevs) + "\n")

    catsj = jaccard_score(cats, topcats)
    consj = jaccard_score(cons, topcons)
    # levsj = jaccard_score(levs, toplevs)
    bag_levs = Counter(levs)
    bag_toplevs = Counter(toplevs)
    fout.write("{}, {} {}".format( bag_levs, bag_toplevs, bag_levs & bag_toplevs))
    fout.write("{}, {} {}".format(bag_levs, bag_toplevs, bag_levs | bag_toplevs))
    levsj = sum((bag_levs & bag_toplevs).values()) / sum((bag_levs | bag_toplevs).values())


    fout.write("\nJaccard indices: categories {}  concepts {}  levels{}\n\n".format(catsj, consj, levsj))