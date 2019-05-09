import copy
import gc
import math
import pickle
import zlib
from collections import defaultdict

import numpy

import pandas
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, concatenate, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from hwgen.common import get_student_list, get_user_data, make_gb_question_map, get_all_attempts
from hwgen.hwgengen2 import build_dob_cache
from hwgen.profiler import get_attempts_from_db
from matplotlib import pyplot

gb_qmap = make_gb_question_map()

def create_assignment_summary(assignments : pandas.DataFrame):
    print(assignments.shape)
    print(assignments[0:56])
    print(len(assignments))

    qmap = make_gb_question_map()

    assignments.loc[:, "creation_date"] = pandas.to_datetime(assignments["creation_date"]).dt.floor("D")

    print("ix loop to ", max(assignments.index))
    cache = {}
    for ix in assignments.index:
        if ix % 100 ==0:
            print(ix)


        aid, ts, gr_id, gb_id, t = assignments.loc[ix, ["id","creation_date","group_id","gameboard_id","owner_user_id"]]
        hxs_from_map = copy.copy(qmap[gb_id])
        print(gb_id,"->",len(hxs_from_map))
        student_df = get_student_list([gr_id])
        students = list(student_df["user_id"])

        if (ts,gr_id,t) not in cache:
            cache[(ts,gr_id,t)] = (aid, students, hxs_from_map)
        else:
            print("adding to xisitn cache entry")
            aid, xstudents, xhexes = cache[(ts,gr_id,t)]
            for s in students:
                if s not in xstudents:
                    xstudents.append(s)
            for h in hxs_from_map:
                if h not in xhexes:
                    xhexes.append(h)
            print(ts,gr_id,t)
            # print("hexes is now",len(xhexes),"long")#: {}".format(len(xhexes),xhexes))
            cache[(ts,gr_id,t)] = (aid,xstudents, xhexes)

    df = pandas.DataFrame(index=range(len(cache)))
    print("building df... #entries:", len(cache))
    for ix,k in enumerate(cache):
        if ix % 100 == 0:
            print(ix)
        ts,gr_id,t = k
        aid, students, hxs = cache[k]
        df.loc[ix, "creation_date"] = ts
        df.loc[ix, "group_id"] = int(gr_id)
        df.loc[ix, "owner_user_id"] = int(t)
        df.loc[ix, "id"] = int(aid)
        df.loc[ix, "students"] = str(students)
        df.loc[ix, "hexes"] = str(hxs)
        df.loc[ix, "num_hexes"] = len(hxs)
        hex_in_book = [(h.startswith("ch_") or h.startswith("ch-i")) for h in hxs]
        df.loc[ix, "has_book_hexes"] = (True in hex_in_book)
        df.loc[ix,"include"] = (len(students)>0) and (len(students) <= 20)
    print(df.shape)
    return df



def make_phybook_model(n_S, n_X, n_U, n_A, n_P, lr):
    # this is our input placeholder
    if n_S is not None:
        input_S = Input(shape=(n_S,), name="s_input")
        inner_S = Dense(10, activation="relu")(input_S)

    w=200
    #w100: 17% to beat
    do=.2

    input_X = Input(shape=(n_X,), name="x_input")
    input_U = Input(shape=(n_U,), name="u_input")
    input_A = Input(shape=(n_A,), name="a_input")

    inner_X = Dense(300, activation="relu")(input_X)
    inner_U = Dense(300, activation="relu")(input_U)
    inner_A = Dense(300, activation="relu")(input_A)

#     inner_X = Dense(100, activation="relu")(inner_X)
#     inner_U = Dense(100, activation="relu")(inner_U)
#     inner_A = Dense(100, activation="relu")(inner_A)

    
    hidden = concatenate([inner_S, inner_X, inner_U , inner_A ])

#     hidden = Dense(w, activation='relu')(hidden)
#     hidden = Dropout(.2)(hidden)
#     hidden = Dense(w, activation='relu')(hidden) #2 layer no DO: 221
#     hidden = Dropout(.2)(hidden)
    ########
    hidden = Dense(512, activation='relu')(hidden)
    hidden = Dropout(.2)(hidden)
    hidden = Dense(256, activation='relu')(hidden) #2 layer no DO: 221
    hidden = Dropout(.2)(hidden)
#     hidden = Dense(128, activation='relu')(hidden) #2 layer .2 DO: 207
#     hidden = Dropout(.2)(hidden)

    next_pg = Dense(n_P, activation='softmax', name="next_pg")(hidden)

    o = Adam(lr= lr)

    # m = Model(inputs=[input_S, input_U], outputs=[next_pg, output_U])
    # m.compile(optimizer=o, loss=["binary_crossentropy","binary_crossentropy"], metrics={'next_pg':["binary_accuracy", "top_k_categorical_accuracy"], 'outAuto':['binary_accuracy']})
    if n_S is not None:
        ins = [input_S, input_X, input_U, input_A]
    else:
        ins = input_U

    m = Model(inputs=ins, outputs=[next_pg])
    m.compile(optimizer=o, loss='categorical_crossentropy', metrics={'next_pg':['acc', 'top_k_categorical_accuracy']})
    # plot_model(m, to_file='hwgen_model.png')
    m.summary()
    # input(",,,")
    return m


def cnt_ass(df):
    cnt = 0
    for ix in df.index:
        cnt+= len(eval(df.loc[ix,"students"]))
    return cnt

def augment_data(ass_summ, sxua, pid_override, all_qids=None, all_page_ids=None, filter=False):
    # print("ass_summ len = ",len(ass_summ))
    print("augmenting")
    psi_ass_cnt = 0
    out_age = 0
    out_dop = 0
    not_in_sxua = 0
    psi_collaps_ts = 0

    max_dop = 0
    inverse_all_page_ids = {}
    for pix,pid in enumerate(pid_override):
        inverse_all_page_ids[pid] = pix

    psi_atts_cache = {}
    group_ids = pandas.unique(ass_summ["group_id"])

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
    ass_summ.loc[:, "creation_date"] = pandas.to_datetime(ass_summ["creation_date"]).dt.floor("D")
    
    j=0
    for aix in ass_summ.index:
        if j % len(ass_summ)==0:
            print(j)
        j+=1
        them_hexes = sorted([h.replace("-","_") for h in eval(ass_summ.loc[aix, "hexes"]) if h in pid_override])
        student_ids = eval(ass_summ.loc[aix, "students"])
        gr_id = ass_summ.loc[aix, "group_id"]
        aid = ass_summ.loc[aix, "id"]
        ts = ass_summ.loc[aix,"creation_date"]

        if them_hexes == []:
            continue
        # else:
        #     hx = them_hexes
        #
        # hix = pid_override.index(hx)
        dop_limit = 365
        for psi in student_ids:
            try:
                S, X, U, A = pickle.loads(zlib.decompress(sxua[psi][ts]))
            except:
                print(psi,ts,"not in sxua, skip")
                continue

            # print("remapping from q to p")
            if all_page_ids:
                #get the weights (in qns) of all the pages here...
#                 page_wgts = defaultdict(int)
#                 for q in all_qids:
#                     pid = q.split("|")[0]
#                     page_wgts[pid] += 1
                
                plabs = [all_qids[i].split("|")[0] for i in numpy.nonzero(X)[0]]
                pixes = [all_page_ids.index(lab) for lab in plabs]
                X = numpy.zeros(len(all_page_ids))
                # for pix in pixes:
                X[pixes]=1.0
                
#                 U = numpy.zeros(len(all_page_ids))
#                 for plab,pix in zip(plabs,pixes):
#                     U[pix] += 1.0/page_wgts[plab]
#                 print("sum mapped X:",sum(X))
            # print("remapping from q to p DOME")

#             U = numpy.zeros(1)
#             A = numpy.zeros(1)
    
            dsreg = max(0,S[1])
            dsass = max(0,S[2])
            # dop = min(dsass,dsreg)
            dop = min(dsass,dsreg)

#             print("filter dops =",filter)
            if filter:
#                 print("cf {} vs {}".format(dop, dop_limit))
                if dop > dop_limit:
#                     print("dropping")
                    out_dop += 1
                    continue

            if (S[0]<16 or S[0]>18): #i.e. if student has no valid age TODO filter for EDM'19
                # print(psi,S[0],"not in age range, skipping")
                out_age += 1
                continue

            if dop > max_dop:
                max_dop = dop

            # if S[1]==0: #no time in platform
            #     continue

            hexes_tried = []
            hexes_to_try = them_hexes

            if psi in psi_atts_cache:
                atts = psi_atts_cache[psi]
            else:
                atts = get_attempts_from_db(psi)
            psi_atts_cache[psi]=atts

            fatts = atts[atts["timestamp"] < ts]
            for qid in fatts["question_id"]:
                pid = qid.split("|")[0]
                if pid not in hexes_tried:
                    if pid in pid_override:
                        hexes_tried.append(pid)

            natts = fatts.shape[0]
            nsucc = len(set(fatts[fatts["correct"] == True]["question_id"]))
            ndist = len(set(fatts["question_id"]))
            # passrate = nsucc/dop if dop>0 else -1
            # passprob = nsucc/natts if natts>0 else 0
            # passprob_perday = passprob / dop if dop>0 else -1

            crapness = dop * natts / (nsucc if nsucc > 0 else 0.01)

            del fatts

            # for hx in hexes:
            #     if hx not in hexes_tried:
            #         hexes_to_try.append(hx)

            # if hexes_to_try==[]:
            #     print("no hexes to try")
            #     continue
            # hexes_to_try = hexes

            y_true = numpy.zeros(len(pid_override))  # numpy.zeros(len(all_page_ids))
            # hexes_to_try = sorted(hexes_to_try)
            #for hx in sorted(hexes_to_try):
            # hx = sorted(hexes_to_try)[ (len(hexes_to_try)-1)//2 ]

            # y_true[hix] = 1.0

            TARGET_MODE = "no_weight"
            if TARGET_MODE=="decision_weighted": #230
                for hx in hexes_to_try:
                    hxix = pid_override.index(hx)
                    y_true[hxix] = 1.0 / len(hexes_to_try)
            elif TARGET_MODE=="no_weight": #224
                for hx in hexes_to_try:
                    hxix = pid_override.index(hx)
                    y_true[hxix] = 1.0
            elif TARGET_MODE=="first": #221
                hx = sorted(hexes_to_try)[0]
                hxix = pid_override.index(hx)
                y_true[hxix] = 1.0
            # elif TARGET_MODE=="middle":
            #     hx = sorted(hexes_to_try)[(len(hexes_to_try) - 1) // 2]
            #     hxix = pid_override.index(hx)
            #     y_true[hxix] = 1.0
            # else:
            #     raise ValueError("'{}' is not a valid target mode!".format(TARGET_MODE))


            # X=Xa
            # print("hexes tried: {}".format(hexes_tried))
            # print("hexes t try: {}".format(hexes_to_try))
            # print("hexes      : {}".format(hexes))

            # aid_list.append(aid)
            s_raw_list.append(S)

            # nsucc = int(10.0 *nsucc / age_1dp)/10.0
            # s_list.append([(int(10*S[0])/10.0), S[1], natts, ndist, nsucc])
            # s_list.append([natts, ndist, nsucc])
            # Sa = [round(S[0], 1), S[2], natts, ndist, nsucc] #214. 267
            Sa = [round(S[0],1), dop, natts, ndist, nsucc] # S1=
            # Sa = [ round(S[0],1), S[2], crapness] #sumMSE 202, voteMSE 242
            # Sa = [ round(S[0],1) ] # 203, 245
            # Sa = [ S[2] ] #209, 250
#             Sa = [ round(S[0],1), dop, crapness ] #214 230
            # Sa = [0]
            s_list.append(Sa)# (nsucc/natts if natts>0 else 0)])

            # print("appended SA")
            x_list.append(X)
            u_list.append(U)
            a_list.append(A)
            y_list.append(y_true)
            psi_list.append(psi)
            hexes_to_try_list.append(hexes_to_try)
            hexes_tried_list.append(hexes_tried)
            # gr_id = 0 #TODO should separate out based on (psi,group) pair
            gr_id_list.append(gr_id)
            aid_list.append(aid)
            ts_list.append(ts)

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
    print("Total of {} (student,assts)".format( len(s_list)))
    print("From an original {}".format(psi_ass_cnt))
    print("After merging timestamps: {}".format((psi_ass_cnt - psi_collaps_ts)))
    print("#not found in SXUA: {}".format(not_in_sxua))
    print("#outside age range: {}".format(out_age))
    print("#outside date range: {}".format(out_dop))

    return aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list, max_dop


def train_deep_model(aug):
    aid_list, s_list, x_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list, maxdops = aug

    print((s_list.shape, x_list.shape, u_list.shape, a_list.shape, y_list.shape))

    #get min and max dops
    print("min/max age", min(s_list[:,0]), max(s_list[:,0]))
    print("min/max days since first assignment", min(s_list[:,1]), max(s_list[:,1]))
    # print("min/max flab", min(s_list[:,2]), max(s_list[:,2]))


    SCALE = False
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

    print("s,x list shapes:",s_list.shape, x_list.shape)
    # x_list = numpy.concatenate((s_list, x_list), axis=1)
    # print(x_list.shape)
    gc.collect()

    # if fs is None:
    #     fs = feature_check(s_list, x_list, y_list)

    #OK, we now have the four student profile vectors, and the true y vector, so we can fit the model
    max_mod = None

    # x_mask = numpy.nonzero(numpy.any(x_list != 0, axis=0))[0]
    # print(x_list.shape)
    # x_list = x_list[:, x_mask]

    min_loss = math.inf
    model = None
    if model is None:
        lrs = []
        accs = []
        BSs = []
        max_acc = -1
        max_BS = -1

        print("making model")
        # x_list = top_n_of_X(x_list, fs)
        # print(s_list)
        # print(x_list)
        # print(u_list)
        # print(a_list)

        Sw = s_list.shape[1]
        Xw = x_list.shape[1]
        Uw = u_list.shape[1]
        Aw = a_list.shape[1]
        print((Sw, Xw, Uw, Aw, y_list.shape[1]))

        print(len(s_list))

#         es = EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')#, restore_best_weights=True)
        es = EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')
        # for BS in [50, 64, 100]:
        #     for LR in [0.003, 0.0025, 0.002]:
        # for BS in [40,50,60,70,80]:
        for BS in [32]: #80
            # for LR in [0.0015, 0.002, 0.0025, 0.003, 0.0035]:
            for LR in [0.001]: #0.0015
                model = make_phybook_model(Sw,Xw,Uw,Aw, y_list.shape[1], lr=LR)
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

#                 history = model.fit([s_list, x_list,u_list,a_list], y_list, verbose=1, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)
                history = model.fit([s_list, x_list,u_list,a_list], y_list, verbose=1, validation_split=.2, epochs=100, callbacks=[es], shuffle=True, batch_size=BS)

                scores = model.evaluate([s_list,x_list,u_list,a_list], y_list)
                print(scores)
                lrs.append(LR)
                accs.append( scores[1])
                BSs.append(BS)
                if scores[0] < min_loss:
                    max_mod = model
                    max_acc = scores[1]
                    min_loss = scores[0]
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
        print((max(accs), lrs[max_acc_ix], BSs[max_acc_ix]))
    return max_mod  # , sscaler, levscaler, volscaler


def build_start_dates(ass_summ):
    start_dates = {}
    ss = set()
    first_assign = {}
    for ts, aix, grid in zip(list(ass_summ["creation_date"]), list(ass_summ.index), list(ass_summ["group_id"])):
        students = list(get_student_list(grid)["user_id"])
        ss.update(students)
        for s in students:
            if s not in first_assign:
                first_assign[s] = copy.copy(ts)
    psi_df = get_user_data(list(ss))
    for ix in psi_df.index:
        psi, rd = psi_df.loc[ix, ["id","registration_date"]]
        if psi not in start_dates:
            ts = first_assign[psi]
            reg_date = pandas.to_datetime(psi_df.iloc[0]["registration_date"])
            start_dates[psi] = (ts, copy.copy(reg_date))
    return start_dates

def build_SXUA(base, ass_summ, all_qids, pid_override, start_dates):
    all_attempts = get_all_attempts()
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
    # if True:
        dob_cache = build_dob_cache(ass_summ)
        joblib.dump(dob_cache, base + "dob_cache")
    print("done")
    group_ids = pandas.unique(ass_summ["group_id"])
    print(len(ass_summ))
    print(len(group_ids))
    print(group_ids[0:20])
    # exit()

    psi_aixes = defaultdict(list)

    for aix in ass_summ.index:
        ts, grid, students = ass_summ.loc[aix, ["creation_date","group_id","students"]]
        students = eval(students)
        for psi in students:
            psi_aixes[psi].append(aix)

    psix=0
    for psi in psi_aixes:
        try:
            attempts = get_attempts_from_db(psi)
        except:
            print("backup attempts for",psi)
            attempts = all_attempts[all_attempts["user_id"]==psi]
            if attempts.empty:
                print("no attempts available for {}".format(psi))
                continue
        if psi not in dob_cache:
            print(psi, "not in dob cache")
            continue

        print("Psi", psix, "/", len(psi_aixes))
        psix+=1
        if psi not in start_dates:
            continue
        first_asst = start_dates[psi][0]
        reg_date = start_dates[psi][1]

        aixes = psi_aixes[psi] # list of aixes in chrono order
        S = numpy.zeros(8)
        X = numpy.zeros(len(all_qids), dtype=numpy.int16)
        U = numpy.zeros(len(all_qids), dtype=numpy.int8)
        A = numpy.zeros(len(pid_override), dtype=numpy.int8)
        SXUA[psi] = {}
        print("+", psi)  # , S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
        # print(rd)
        l_ts = pandas.to_datetime("1970-01-01 00:00:00")
        l_hexes = []

        aixix = 0
        for aix in sorted(aixes):
            print("Asst", aixix, "/", len(aixes))
            aixix += 1
            # print("pretend processing of", psi, aix, "here...")
            # continue

            ts, gr_id, gb_id, now_hexes = ass_summ.loc[
                aix, ["creation_date", "group_id", "gameboard_id", "hexes"]]
            

            print("calc day counts for:",psi, aix)
            all_days = int((ts - reg_date).days)
            print(all_days, " := ", ts, reg_date)
            days_active = int((ts - first_asst).days)
            print(days_active, " := ", ts, first_asst)

            now_hexes = eval(now_hexes)

            attempts = attempts[attempts["timestamp"] < ts]
            all_wins = list(attempts[(attempts["correct"] == True)]["question_id"])

            recent_attempts = attempts[attempts["timestamp"] >= l_ts]
            recent_qids = list(set(recent_attempts["question_id"]))
            recent_wins = list(recent_attempts[(recent_attempts["correct"] == True)]["question_id"])

            for qid in recent_qids:
                # try:
                qix = all_qids.index(qid)
                attct = numpy.sum(recent_attempts["question_id"] == qid)
                X[qix] += attct
                if qid in recent_wins:
                    U[qix] = 1
                # except:
                #     print("UNK Qn ", qid)
                #     continue

            # print("last hexes>>>", l_hexes)
            #ASSIGNMENTS
            print("last hexes is {} long".format(len(l_hexes)))
            for hx in l_hexes:
                hx = hx.split("|")[0].replace("-","_")
                if hx in pid_override:
                    hxix = pid_override.index(hx)
                    A[hxix] = 1
                else:
                    print("ASS {} NOT IN PID OVERRIDE".format(hx))

            S[0] = (ts - dob_cache[psi]).days / 365.242 if dob_cache[psi] is not None else 0
            # print(ts, l_ts)
            day_delta = max(1, (ts - l_ts).seconds) / 86400.0
            att_delta = recent_attempts.shape[0]
            all_atts = attempts.shape[0]
            # print(day_delta, att_delta)
            # print(reg_date)

            S[1] = all_days
            S[2] = days_active

            S[3] = (att_delta / day_delta)  # recent perseverence
            S[4] = (len(recent_wins) / att_delta if att_delta else 0)  # recent success rate
            S[5] = (all_atts / all_days if all_days else 0)  # all time perseverance
            S[6] = (len(all_wins) / all_atts if all_atts else 0)  # all time success rate
            S[7] = (all_atts / days_active if days_active else 0)  # all time active perseverance

            print("~", psi,ts, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
            SXUA[psi][ts] = zlib.compress(pickle.dumps((S, X, U, A)))
            l_ts = ts
            l_hexes = now_hexes

    return SXUA