import gc
import pickle
import zlib
from copy import copy
from random import choice

import numpy

import pandas
from scipy import sparse
from sklearn.externals import joblib

from hwgen.common import get_student_list, get_user_data, make_gb_question_map, get_all_assignments
from hwgen.hwgengen2 import build_dob_cache
from hwgen.profiler import get_attempts_from_db


def filter_assignments(assignments, mode, max_n=1000):
    # assignments = assignments[["id","group_id","gameboard_id"]]
    assignments.loc[:,"include"] = False
    assignments.index = assignments["id"]
    print(assignments.shape)
    map = make_gb_question_map()
    gr_ids = numpy.unique(assignments["group_id"])
    ct=0

    print("checking for empty groups")
    gr_ids_keep = {}
    for grix, gr_id in enumerate(gr_ids):
        if gr_id in gr_ids_keep:
            continue
        gr_ids_keep[gr_id] = False
        students = get_student_list([gr_id])
        if not students.empty:
            for psi in list(students["user_id"]):
                atts = get_attempts_from_db(psi)
                if not atts.empty:
                    gr_ids_keep[gr_id]=True
                    break
    print("...done")

    incs = []
    for ix in assignments["id"]:
        include=True
        if not gr_ids_keep[assignments.loc[ix,"group_id"]]:
            include = False
        if mode == "all":
            include = True
        elif mode == "book_only":
            gb_id = assignments.loc[ix, "gameboard_id"]
            hexes = map[gb_id]
            for hx in hexes:
                hx = hx.split("|")[0]
                if not (hx.startswith("ch_") or hx.startswith("ch-i")):
                    include = False
                    break
        elif mode == "non_book_only":
            gb_id = assignments.loc[ix, "gameboard_id"]
            hexes = map[gb_id]
            for hx in hexes:
                hx = hx.split("|")[0]
                if (hx.startswith("ch_") or hx.startswith("ch-i") or hx.startswith("gcse_")):
                    include = False
                    break
        elif mode == "no_book_gcse_or_chem":
            gb_id = assignments.loc[ix, "gameboard_id"]
            hexes = map[gb_id]
            for hx in hexes:
                hx = hx.split("|")[0]
                if (hx.startswith("ch_") or hx.startswith("ch-i") or hx.startswith("gcse_") or hx.startswith("chem_")):
                    include = False
                    break
        else:
            pass
            # raise ValueError("Unknown book filtering mode")

        incs.append(include)

        if include:
            ct+=1
            print(ct)
        if max_n and ct >= max_n:
            break

    # padding = [False] * (assignments.shape[0] - len(incs))
    # incs = incs + padding
    # assignments[incs,"include"] = True
    incixes = []
    for aid,inc in zip(list(assignments["id"]), incs):
        if inc : incixes.append(aid)

    assignments = assignments[assignments["id"].isin(incixes)]

    print(assignments.shape)
    return assignments



def augment_data(tr, sxua, filter_by_length=False, pid_map=None, sugg_map=None):
    gb_qmap = make_gb_question_map()
    inverse_all_page_ids = {}
    for pix,pid in enumerate(pid_map):
        inverse_all_page_ids[pid] = pix

    psi_atts_cache = {}

    aid_list = []
    s_list = []
    x_list = []
    c_list = []
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
    for aid in tr.loc[:,"id"]:
        azz = tr.loc[aid,:]
        gr_id = azz["group_id"]
        ts = azz["creation_date"]
        student_ids = list(get_student_list(gr_id)["user_id"])
        print(student_ids)
        gb_id = azz["gameboard_id"]
        hexes = set()
        hexes.update(list(gb_qmap[gb_id]))

        for psi in student_ids:
            sxua_psi = sxua[psi]
            S, _, _, A = pickle.loads(zlib.decompress(sxua_psi[ts]))
            if S[0] < 16 or S[0] > 18:
                continue
            hexes_tried = []
            hexes_to_try = []


            Xa = numpy.zeros(shape=len(pid_map), dtype=numpy.bool)
            Xc = numpy.zeros(shape=len(pid_map), dtype=numpy.uint16)
            Xm = numpy.zeros(shape=len(pid_map), dtype=numpy.bool)
            if psi in psi_atts_cache:
                atts = pickle.loads(zlib.decompress(psi_atts_cache[psi]))
            else:
                atts = get_attempts_from_db(psi)
                psi_atts_cache[psi] = zlib.compress(pickle.dumps(atts))

            fatts = atts[atts["timestamp"] < ts]
            for qid in fatts["question_id"]:
                pid = qid.split("|")[0]
                if pid not in hexes_tried:
                    if pid in inverse_all_page_ids:
                        hexes_tried.append(pid)
                        Xa[inverse_all_page_ids[pid]] = 1

            for qid in fatts["question_id"]:
                pid = qid.split("|")[0]
                if pid in inverse_all_page_ids:
                    Xc[inverse_all_page_ids[pid]] += 1

            natts = fatts.shape[0]
            ndist = len(set(fatts["question_id"]))

            catts = fatts[fatts["correct"] == True]
            for qid in fatts["question_id"]:
                pid = qid.split("|")[0]
                if pid in inverse_all_page_ids:
                    Xm[inverse_all_page_ids[pid]] = 1

            nsucc = len(set(catts["question_id"]))
            dop = S[1]
            # passrate = nsucc/dop if dop>0 else -1
            # passprob = nsucc/natts if natts>0 else 0
            # passprob_perday = passprob / dop if dop>0 else -1

            crapness = dop * natts / (nsucc if nsucc > 0 else 0.1)

            del atts
            del fatts
            del catts


            for hx in hexes:
                if hx not in hexes_tried:
                    hexes_to_try.append(hx)

            y_true = numpy.zeros(len(sugg_map))  # numpy.zeros(len(all_page_ids))
            hexes_to_try = sorted(hexes_to_try)
            # for hx in sorted(hexes_to_try):
            # hx = sorted(hexes_to_try)[ (len(hexes_to_try)-1)//2 ]

            hexes_to_try = [hx for hx in hexes_to_try if hx in sugg_map]
            if hexes_to_try == []:
                print("no hexes to try")
                continue

            # y_trues = None
            TARGET_MODE = "random"
            # if TARGET_MODE=="repeat":
            #     hx_ct = 0
            #     y_trues = []
            #     for hx in hexes_to_try:
            #         hxix = sugg_map.index(hx)
            #         y_true[hxix] = 1.0
            #         y_trues.append(copy(y_true))
            #         hx_ct += 1
            if TARGET_MODE == "random":
                hx = choice(hexes_to_try)
                hxix = sugg_map.index(hx)
                y_true[hxix] = 1.0
            elif TARGET_MODE == "decision_weighted":
                for hx in hexes_to_try:
                    hxix = sugg_map.index(hx)
                    y_true[hxix] = 1.0 / len(hexes_to_try)
            elif TARGET_MODE == "no_weight":
                for hx in hexes_to_try:
                    hxix = sugg_map.index(hx)
                    y_true[hxix] = 1.0
            elif TARGET_MODE == "first":
                hx = sorted(hexes_to_try)[0]
                hxix = sugg_map.index(hx)
                y_true[hxix] = 1.0
            elif TARGET_MODE == "middle":
                hx = sorted(hexes_to_try)[(len(hexes_to_try) - 1) // 2]
                hxix = sugg_map.index(hx)
                y_true[hxix] = 1.0
            else:
                raise ValueError("'{}' is not a valid target mode!".format(TARGET_MODE))

            print("hexes t try: {}".format(hexes_to_try))
            print("hexes      : {}".format(hexes))

            aid_list.append(aid)
            s_raw_list.append(S)

            # age_1dp = int(10.0*S[0])/10.0
            onedp = lambda z: int(10.0 * z) / 10.0
            # nsucc = int(10.0 *nsucc / age_1dp)/10.0
            # s_list.append([(int(10*S[0])/10.0), S[1], natts, ndist, nsucc])
            # s_list.append([natts, ndist, nsucc])
            Sa = [onedp(S[0]), dop, natts, ndist, nsucc]
            # Sa = [0]

            # y_trues = [y_true] if (y_trues is None) else y_trues
            # for y_true in y_trues:
            s_list.append(Sa)  # (nsucc/natts if natts>0 else 0)])
            x_list.append(Xa)
            # c_list.append(Xc)
            # u_list.append(Xm)
            c_list.append([0])
            u_list.append([0])
            # a_list.append(A)
            a_list.append([0])
            y_list.append(y_true)
            psi_list.append(psi)
            hexes_to_try_list.append(hexes_to_try)
            hexes_tried_list.append(hexes_tried)
            gr_id_list.append(gr_id)
            ts_list.append(ts)
            fout.write("{},{},{},{},{},{},\"{}\",\"{}\"\n".format(ts, gr_id, psi, ",".join(map(str, Sa)), sum(Xa), sum(Xa > 0), "\n".join(hexes_tried), "\n".join(hexes_to_try)))

            # ,\"{}\",\"{}\"\n".format(ts, gr_id, psi, ",".join(map(str, Sa)), Xa.sum(), numpy.sum(Xa > 0),
            #                                        "\n".join(hexes_tried), "\n".join(hexes_to_try)))
    fout.close()
    gc.collect()

    s_list = numpy.array(s_list)
    x_list = numpy.array(x_list, dtype=numpy.bool)
    # c_list = numpy.array(c_list, dtype=numpy.uint16)
    # u_list = numpy.array(u_list, dtype=numpy.bool)
    # a_list = numpy.array(a_list, dtype=numpy.bool)
    y_list = numpy.array(y_list)
    # psi_list = numpy.array(psi_list)
    # s_list = sparse.csr_matrix(s_list)
    # x_list = sparse.csr_matrix(x_list, dtype=numpy.bool)
    # c_list = sparse.csr_matrix(c_list, dtype=numpy.uint16)
    # u_list = sparse.csr_matrix(u_list, dtype=numpy.bool)
    # a_list = sparse.csr_matrix(a_list, dtype=numpy.bool)
    # y_list = sparse.csr_matrix(y_list)
    psi_list = sparse.csr_matrix(psi_list)

    return aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list


def augment_data_old(tr, sxua, filter_by_length=True, pid_map=None, sugg_map=None):
    bermuda_hexes = set()
    gb_qmap = make_gb_question_map()

    inverse_all_page_ids = {}
    for pix,pid in enumerate(pid_map):
        inverse_all_page_ids[pid] = pix

    psi_atts_cache = {}
    group_ids = pandas.unique(tr["group_id"])

    aid_list = []
    s_list = []
    x_list = []
    c_list = []
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

    psis_not_to_include = set()
    if filter_by_length:
        for gr_id in group_ids:
            gr_ass = tr[tr["group_id"] == gr_id]
            student_ids = list(get_student_list(gr_id)["user_id"])
            tss = sorted(list(set(gr_ass["creation_date"])))
            if len(tss)<5:
                pass
            else:
                psis_not_to_include.update(student_ids)

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
                if psi not in psis_not_to_include:
                    continue

                if psi in group_track and group_track[psi]!=gr_id:
                    print("skipping n-th group")
                    continue
                else:
                    group_track[psi]=gr_id

                sxua_psi = sxua[psi]
                S,_,_,A = pickle.loads(zlib.decompress(sxua_psi[ts]))
                if S[0]<16 or S[0]>18: #i.e. if student has no valid age TODO honolulu
                    continue
                # if S[1]==0: #no time in platform
                #     continue

                # print(aid, psi, ts)

                hexes_tried = []
                hexes_to_try = []
                # if len(hexes)==1:
                #     hexes_to_try = hexes
                # else:

                Xa = numpy.zeros(shape=len(pid_map), dtype=numpy.bool)
                Xc = numpy.zeros(shape=len(pid_map), dtype=numpy.uint16)
                Xm = numpy.zeros(shape=len(pid_map), dtype=numpy.bool)
                if psi in psi_atts_cache:
                    atts = pickle.loads(zlib.decompress(psi_atts_cache[psi]))
                else:
                    atts = get_attempts_from_db(psi)
                    psi_atts_cache[psi]=zlib.compress(pickle.dumps(atts))

                fatts = atts[atts["timestamp"] < ts]
                for qid in fatts["question_id"]:
                    pid = qid.split("|")[0]
                    if pid not in hexes_tried:
                        if pid in inverse_all_page_ids:
                            hexes_tried.append(pid)
                            Xa[inverse_all_page_ids[pid]]=1

                for qid in fatts["question_id"]:
                    pid = qid.split("|")[0]
                    if pid in inverse_all_page_ids:
                        hexes_tried.append(pid)
                        Xc[inverse_all_page_ids[pid]]+=1

                natts = fatts.shape[0]
                ndist = len(set(fatts["question_id"]))

                catts = fatts[fatts["correct"]==True]
                for qid in fatts["question_id"]:
                    pid = qid.split("|")[0]
                    if pid in inverse_all_page_ids:
                        hexes_tried.append(pid)
                        Xm[inverse_all_page_ids[pid]]=1

                nsucc = len(set(catts["question_id"]))
                dop = S[1]
                # passrate = nsucc/dop if dop>0 else -1
                # passprob = nsucc/natts if natts>0 else 0
                # passprob_perday = passprob / dop if dop>0 else -1

                crapness = dop * natts / (nsucc if nsucc > 0 else 0.1)

                del atts
                del fatts
                del catts

                for hx in hexes:
                    if hx not in hexes_tried:
                        hexes_to_try.append(hx)

                y_true = numpy.zeros(len(sugg_map))  # numpy.zeros(len(all_page_ids))
                hexes_to_try = sorted(hexes_to_try)
                #for hx in sorted(hexes_to_try):
                # hx = sorted(hexes_to_try)[ (len(hexes_to_try)-1)//2 ]

                for hx in hexes_to_try:
                    if hx not in sugg_map:
                        bermuda_hexes.add(hx)

                hexes_to_try = [hx for hx in hexes_to_try if hx in sugg_map]
                if hexes_to_try==[]:
                    print("no hexes to try")
                    continue

                # y_trues = None
                TARGET_MODE = "random"
                # if TARGET_MODE=="repeat":
                #     hx_ct = 0
                #     y_trues = []
                #     for hx in hexes_to_try:
                #         hxix = sugg_map.index(hx)
                #         y_true[hxix] = 1.0
                #         y_trues.append(copy(y_true))
                #         hx_ct += 1
                if TARGET_MODE=="random":
                    hx = choice(hexes_to_try)
                    hxix = sugg_map.index(hx)
                    y_true[hxix] = 1.0
                elif TARGET_MODE=="decision_weighted":
                    for hx in hexes_to_try:
                        hxix = sugg_map.index(hx)
                        y_true[hxix] = 1.0 / len(hexes_to_try)
                elif TARGET_MODE=="no_weight":
                    for hx in hexes_to_try:
                        hxix = sugg_map.index(hx)
                        y_true[hxix] = 1.0
                elif TARGET_MODE=="first":
                    hx = sorted(hexes_to_try)[0]
                    hxix = sugg_map.index(hx)
                    y_true[hxix] = 1.0
                elif TARGET_MODE=="middle":
                    hx = sorted(hexes_to_try)[(len(hexes_to_try) - 1) // 2]
                    hxix = sugg_map.index(hx)
                    y_true[hxix] = 1.0
                else:
                    raise ValueError("'{}' is not a valid target mode!".format(TARGET_MODE))

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

                # y_trues = [y_true] if (y_trues is None) else y_trues
                # for y_true in y_trues:
                s_list.append(Sa)# (nsucc/natts if natts>0 else 0)])
                x_list.append(Xa)
                c_list.append(Xc)
                u_list.append(Xm)
                a_list.append(A)
                y_list.append(y_true)
                psi_list.append(psi)
                hexes_to_try_list.append(hexes_to_try)
                hexes_tried_list.append(hexes_tried)
                gr_id_list.append(gr_id)
                ts_list.append(ts)

                fout.write("{},{},{},{},{},{},\"{}\",\"{}\"\n".format(ts,gr_id,psi,",".join(map(str,Sa)), Xa.sum(), numpy.sum(Xa>0), "\n".join(hexes_tried), "\n".join(hexes_to_try)))
        gc.collect()
    fout.close()
    # exit()
    # input("nibit")
    gc.collect()

    s_list = numpy.array(s_list)
    x_list = numpy.array(x_list, dtype=numpy.bool)
    c_list = numpy.array(c_list, dtype=numpy.uint16)
    # print(x_list.shape)
    # x_mask = numpy.nonzero(numpy.any(x_list != 0, axis=0))[0]
    # x_list = x_list[:, x_mask]
    # print(x_list.shape)
    u_list = numpy.array(u_list, dtype=numpy.bool)
    # a_list = numpy.array(a_list, dtype=numpy.int8)
    y_list = numpy.array(y_list)
    psi_list = numpy.array(psi_list)
    print("bermudean hexes are:")
    print(bermuda_hexes)

    return aid_list, s_list, x_list, c_list, u_list, a_list, y_list, psi_list, hexes_to_try_list, hexes_tried_list, s_raw_list, gr_id_list, ts_list

def populate_student_cache(assignments):
    group_ids = pandas.unique(assignments["group_id"])
    students_seen = set()
    for gr_id in group_ids:
        student_ids = list(get_student_list(gr_id)["user_id"])
        if student_ids:
            student_data = get_user_data(student_ids)
        student_ids = list(get_student_list(gr_id)["user_id"])
        for psi in student_ids:
            if psi not in students_seen:
                attempts = get_attempts_from_db(psi)
                students_seen.add(psi)
                print("pop'd cache for {}".format(psi))
        print("pop'd cache for {}".format(gr_id))
    return list(students_seen)

def build_group_lookup():
    g_lookup = {}
    az = get_all_assignments()
    group_ids = pandas.unique(az["group_id"])
    for gr_id in group_ids:
        student_ids = list(get_student_list(gr_id)["user_id"])
        for psi in student_ids:
            if psi not in g_lookup:
                g_lookup[psi] = []
            g_lookup[psi].append(gr_id)
    return g_lookup

# def lazy_get_from_SXUA(_SXUA, g_lookup, psi, ts, qid_map, pid_map):
#     if psi not in _SXUA:
#         S = numpy.zeros(6)
#         X = numpy.zeros(len(qid_map), dtype=numpy.int16)
#         U = numpy.zeros(len(qid_map), dtype=numpy.int8)
#         A = numpy.zeros(len(pid_map), dtype=numpy.int8)
#         _SXUA[psi] = {}
#     else:
#         az = get_all_assignments()
#         groups = g_lookup[psi]
#         gr_ass = az[az["group_id"].isin(groups) & ]
#         gb_ids = list(numpy.unique(az["gameboard_id"]))
#
#         ts_list = _SXUA[psi]
#         l_ts = max(ts_list)
#
#         S, X, U, A = pickle.loads(zlib.decompress(_SXUA[psi][ts]))


def build_SXUA(assignments, base, qid_map, pid_map):
    print("building SXUA")

    gb_qmap = make_gb_question_map()

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
            if not student_ids:
                continue

            student_data = get_user_data(student_ids)
            now_hexes = list(gb_qmap[gb_id])
            # print(now_hexes)
            # if 118651 not in student_ids:
            #     continue
            for psi in student_ids:
                # if psi != 118651:
                #     continue
                # print(psi)
                neu = False
                if psi not in SXUA:
                    neu = True
                    S = numpy.zeros(6)
                    X = numpy.zeros(len(qid_map), dtype=numpy.int16)
                    U = numpy.zeros(len(qid_map), dtype=numpy.int8)
                    A = numpy.zeros(len(pid_map), dtype=numpy.int8)
                    SXUA[psi] = {}
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

                attempts = get_attempts_from_db(psi)
                attempts = attempts[attempts["timestamp"] < ts]
                all_wins = list(attempts[(attempts["correct"] == True)]["question_id"])

                recent_attempts = attempts[attempts["timestamp"] >= l_ts]
                # qids = list(set(recent_attempts["question_id"]))
                qids = list(set(recent_attempts["question_id"]))
                recent_wins = list(recent_attempts[(recent_attempts["correct"] == True)]["question_id"])

                for qid in qids:
                    try:
                        qix = qid_map.index(qid)
                        attct = numpy.sum(recent_attempts["question_id"] == qid)
                        X[qix] += attct
                        if qid in recent_wins:
                            U[qix] = 1
                    except:
                        print("UNK Qn ", qid)
                        continue

                print(l_hexes)
                for hx in l_hexes:
                    if hx == "angles_and_projection|angles_and_projection":
                        hx = hx.split("|")[0]
                    hxix = pid_map.index(hx)
                    A[hxix] = 1

                if psi not in dob_cache:
                    S[0] = 0
                else:
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
                print(("+" if neu else "~"), psi, S, numpy.sum(X), numpy.sum(U), numpy.sum(A))
                SXUA[psi][ts] = zlib.compress(pickle.dumps((S, X, U, A)))
    return SXUA