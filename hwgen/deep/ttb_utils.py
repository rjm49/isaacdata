import gc
import pickle
import zlib
import numpy

import pandas
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, concatenate, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from hwgen.common import get_student_list, get_user_data, make_gb_question_map
from hwgen.hwgengen2 import build_dob_cache
from hwgen.profiler import get_attempts_from_db
from matplotlib import pyplot

gb_qmap = make_gb_question_map()


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


def augment_data(tr, sxua, filter_length=True, all_page_ids=None, pid_override=None):
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
                if S[0]<16 or S[0]>18: #i.e. if student has no valid age TODO filter for AAAI19
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


def train_deep_model(tr, sxua, n_macroepochs=100, n_epochs=10, use_linear=False, load_saved_tr=False, all_page_ids=None, pid_override=None):
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
        aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list = augment_data(tr, sxua,  all_page_ids=all_page_ids, pid_override=pid_override)
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

        es = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
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


def build_SXUA(base, assignments, all_qids, all_page_ids, pid_override):
    global SXUA, gr_id, row, aid, ts, gb_id, student_ids
    print("building SXUA")
    SXUA = {}
    student_static = {}
    last_ts = {}
    last_hexes = {}
    print("build dob cache")
    try:
        dob_cache = joblib.load(base + "dob_cache")
    except:
        dob_cache = build_dob_cache(assignments)
        joblib.dump(dob_cache, base + "dob_cache")
    print("done")
    group_ids = pandas.unique(assignments["group_id"])
    print(len(assignments))
    print(len(group_ids))
    print(group_ids[0:20])
    # exit()
    for gr_id in group_ids:
        gr_ass = assignments[assignments["group_id"] == gr_id]
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
            now_hexes = list(gb_qmap[gb_id])
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
                    print("+", psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                    psi_data = student_data[student_data["id"] == psi]
                    rd = pandas.to_datetime(psi_data.iloc[0]["registration_date"])
                    # print(rd)
                    student_static[psi] = (rd,)
                    l_ts = pandas.to_datetime("1970-01-01 00:00:00")
                    l_hexes = []
                else:
                    l_ts = last_ts[psi]
                    l_hexes = last_hexes[psi]
                    S, X, U, A = pickle.loads(zlib.decompress(SXUA[psi][l_ts]))
                # S,X,U,A = copy(S),copy(X),copy(U),copy(A)
                # make updates

                # if psi ==118651:
                #     print("birdskoeping")

                attempts = get_attempts_from_db(psi)
                attempts = attempts[attempts["timestamp"] < ts]
                all_wins = list(attempts[(attempts["correct"] == True)]["question_id"])

                recent_attempts = attempts[attempts["timestamp"] >= l_ts]
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
                day_delta = max(1, (ts - l_ts).seconds) / 86400.0
                att_delta = recent_attempts.shape[0]
                all_atts = attempts.shape[0]
                # print(day_delta, att_delta)
                reg_date = student_static[psi][0]
                # print(reg_date)
                all_days = max(0, (ts - reg_date).days)
                S[1] = all_days
                S[2] = (att_delta / day_delta)  # recent perseverence
                S[3] = (len(recent_wins) / att_delta if att_delta else 0)  # recent success rate
                S[4] = (all_atts / all_days if all_days else 0)  # all time perseverance
                S[5] = (len(all_wins) / all_atts if all_atts else 0)  # all time success rate

                last_ts[psi] = ts
                last_hexes[psi] = now_hexes
                print("~", psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                SXUA[psi][ts] = zlib.compress(pickle.dumps((S, X, U, A)))
                # if str(aid) in ["47150", "49320", "53792"]:
                #     input(">> {}".format(aid))
    return SXUA