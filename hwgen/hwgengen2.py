import gc
import pickle
import zlib
from collections import OrderedDict

import numpy
import pandas
from pandas._libs.tslib import Timestamp

from hwgen.common import init_objects, get_user_data, get_all_assignments, get_student_list, make_gb_question_map

cats, cat_lookup, all_qids, _, _, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(-1)

reverse_qid_dict = {}
for ix,q in enumerate(all_qids):
    reverse_qid_dict[q]=ix

from hwgen.profiler import get_attempts_from_db, get_age_df

profile_cache="../../../isaac_data_files/gengen_cache/"
base = "../../../isaac_data_files/"
prof_fname = base+"gengen_profiles.pkl"

dob_cache = base+"dob_cache.pkl"

LOAD_FROM_PROF_CACHE = True
SAVE_TO_PROF_CACHE = True

def ass_extract(ass):
    id = ass[1]["id"]
    ts = ass[1]['creation_date']
    gb_id = ass[1]["gameboard_id"]
    gr_id = ass[1]["group_id"]
    return id,ts,gb_id,gr_id


def build_dob_cache(dob_cache, assts):
    for ix, ass in enumerate(assts.iterrows()):
        id, ts, gb_id, gr_id = ass_extract(ass)
        students = list(get_student_list(gr_id)["user_id"])
        # print("#{}: PREP: grp {} at {}".format(ix, gr_id, ts))
        group_df = get_user_data(students)
        for psi in students:
            dob = None
            if psi not in dob_cache:
                # print("age gen...")
                age_df = get_age_df(ts, group_df)
                # age_df["dob"] = pandas.to_datetime(age_df["dob"])
                # age = age_df.loc[psi, "age"]
                for psi_inner in students:
                    dob = age_df.loc[psi_inner,"dob"]
                    # print(type(dob))
                    if isinstance(dob, Timestamp):
                        dob_cache[psi_inner] = dob
                    else:
                        dob_cache[psi_inner] = None
    return dob_cache


def build_oa_cache(asst_df, gb_q_map):
    #TODO assumes asst_df is complete, and ordered
    oa_cache = {} # hashtable to hold (timestamp -> student) mappings
    psi_last_A ={} # hashtable to hold (student -> last_assignment_timestamp) mappings, to retrieve previous state
    psi_last_hexes = {}
    for ix, ass in enumerate(asst_df.iterrows()):
        id, ts, gb_id, gr_id = ass_extract(ass)
        hexes = list(gb_q_map[gb_id])
        students = list(get_student_list(gr_id)["user_id"])
        # print("#{}: PREP: grp {} at {}".format(ix, gr_id, ts))
        if ts not in oa_cache:
            oa_cache[ts] = {}

        for psi in students:
            if psi not in psi_last_A:
                A = numpy.zeros(len(all_page_ids)) # create new vector and stash it
                d = oa_cache[ts]
                d[psi] = A
                oa_cache[ts] = d
            else:
                A = numpy.copy(psi_last_A[psi])
                updates = psi_last_hexes[psi]
                for hx in updates:
                    ix = all_page_ids.index(hx)
                    A[ix] = 1 # this homework has been set!
                oa_cache[ts][psi] = A
            psi_last_A[psi] = A
            psi_last_hexes[psi] = hexes
    return oa_cache


def gen_experience(psi, ts_list, clip=True):
    raw_attempts = get_attempts_from_db(psi)
    X_list = []
    # if raw_attempts.empty:
    #     print("student {} has no X attempts".format(psi))
    #     return X_list
    first_non_empty = None
    for ix,ts in enumerate(sorted(ts_list)):
        X = numpy.zeros(len(all_qids))
        attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
        if not attempts.empty:
            if first_non_empty is None:
                first_non_empty = ix
        hits = attempts["question_id"]
        for qid in list(hits):
            try:
                qix = reverse_qid_dict[qid]
            except:
                print("UNK Qn ", qid)
                continue
            # X = numpy.max(X-.1,0)
            # X -= 0.02  # reduce to zero in 50 moves
            # X[X < 0] = 0.0 #bottom out at zero
            # X[X > 0] += 0.01
            X[qix] = 1
            # print("birdvs iirdvs", numpy.median(X), numpy.sum(X))
        # X_list.append(numpy.copy(X))
        X_list.append(X)
        # raw_attempts = raw_attempts[(raw_attempts["timestamp"] >= ts)]
    if clip:
        X_list = X_list[max(first_non_empty - 1, 0):] if (not first_non_empty is None) else X_list[-1:]
    return X_list

class hwgengen2:
    def __init__(self, assts, batch_size=512, pid_override=None, FRESSSH=False, qid_override=all_qids, return_qhist=False, oac=None):
        self.assts: pandas.DataFrame = assts
        self.assts.loc[:,'creation_date'] = pandas.to_datetime(assts['creation_date'])

        self.gb_qmap = make_gb_question_map()

        self.batch_size=batch_size if batch_size!="assignment" else 0
        self.pid_override=pid_override
        self.qid_override = qid_override
        self.return_qhist = return_qhist
        if oac:
            self.open_assignment_cache = oac
        else:
            self.open_assignment_cache = build_oa_cache(assts, self.gb_qmap)

        if not FRESSSH:
            print("APPEND mode")
            #recycle old pap
            try:
                f = open(prof_fname, 'rb')
                self.profiles = pickle.load(f)
                print("got this many profiles:",len(self.profiles))
                # print(list(profiles.keys())[0:10])
                f.close()
            except:
                self.profiles = {}
            # d = open(dob_cache, 'rb')
            # self.dob_cache = pickle.load(d)
            # print("loaded dob cache with {} entries".format(self.dob_cache))
            # d.close()
        else:
            print("Baking FRESH, like cinnamon!")
            self.profiles = {}
            # self.dob_cache = {}

        self.ts_cache = {}
        self.assid_list = []
        self.ts_master_list = []
        self.gb_id_list = []
        self.gr_id_list = []
        self.students_list = []

        print("building dob_cache")
        empty_cache = {}
        self.dob_cache = build_dob_cache(empty_cache, assts)
        print(len(empty_cache))
        print("done")

        for ix, ass in enumerate(self.assts.iterrows()):
            id, ts, gb_id, gr_id = ass_extract(ass)
            self.assid_list.append(id)
            self.ts_master_list.append(ts)
            self.gb_id_list.append(gb_id)
            self.gr_id_list.append(gr_id)
            students = list(get_student_list(gr_id)["user_id"])
            self.students_list.append(students)
            # print("#{}: PREP: grp {} at {}".format(ix, gr_id, ts))
            for psi in students:
                if psi in self.ts_cache.keys():
                    # print("try add ts {}".ts)
                    # temp = self.ts_cache[psi]
                    # print(temp)
                    # temp.append(ts)
                    # self.ts_cache[psi] = temp
                    t = self.ts_cache[psi]
                    t.append(ts)
                    self.ts_cache[psi] = t
                else:
                    self.ts_cache[psi] = [ts]

        c=-1
        for i,ts,gb_id,gr_id in zip(self.assid_list, self.ts_master_list, self.gb_id_list, self.gr_id_list):
            c += 1
            has_changed = False
            students = list(get_student_list(gr_id)["user_id"])
            for psi in students:  # set up the training arrays here
                fn = "prof_{}_{}".format(psi, ts)
                if fn not in self.profiles:
                    print("{}- - - -   profile for {} .. not found .. will create all ={}".format(c,psi, SAVE_TO_PROF_CACHE))
                    has_changed = True
                    group_df = get_user_data(students)
                    ts_list = self.ts_cache[psi]
                    print("ts_list", ts_list)
                    print("s..")
                    s_psi_list = gen_semi_static(psi, self.dob_cache, ts_list)
                    print("x..")
                    x_psi_list = gen_experience(psi, ts_list)
                    print("u..")
                    u_psi_list = gen_success(psi, ts_list)

                    a_psi_list =[]
                    for ts in ts_list:
                        a_psi_list.append(self.open_assignment_cache[ts][psi])

                    print("done")
                    for ts,s_psi,x_psi,u_psi,a_psi in zip(sorted(ts_list[-len(s_psi_list):]),s_psi_list,x_psi_list, u_psi_list, a_psi_list):
                        loopvar = "prof_{}_{}".format(psi, ts)
                        self.profiles[fn] = zlib.compress(pickle.dumps((s_psi, x_psi, u_psi, a_psi)))
                        print("created profile for ",loopvar, "xp=",numpy.sum(x_psi),"sxp=",numpy.sum(u_psi),"S=",s_psi,"Ass/d=",numpy.sum(a_psi))
                else:
                    print(".. {} f/cache".format(fn))
            if has_changed:
                f = open(prof_fname, 'wb')
                pickle.dump(self.profiles, f)
                f.close()
                print("*** *** *** SAVED")

    def __iter__(self):
        b = 0  # batch counter
        c = 0  # cumulative counter
        S = []
        X = []
        U = []
        A=[]
        len_assts = len(self.assts)
        y = []
        awgt = []
        assids = []
        psi_list = []
        qhist_list = []

        last_i = None
        for i, ts, gb_id, gr_id in zip(self.assid_list, self.ts_master_list, self.gb_id_list, self.gr_id_list):
            c += 1
            hexagons = [self.gb_qmap[gb_id][0]]
            students = get_student_list(gr_id)
            students = list(students["user_id"])

            print("...", ts, students, hexagons)
            for psi in students:  # set up the training arrays here
                hexagons = [hx.split("|")[0] for hx in hexagons]

                fn = "prof_{}_{}".format(psi, ts)
                if fn not in self.profiles:
                    print(fn, "not in profiles, why??")
                    continue

                quartet = pickle.loads(zlib.decompress(self.profiles[fn]))
                if quartet is None:
                    print(fn, "non das cacas.")
                else:
                    (s_psi, x_psi, u_psi, a_psi) = quartet
                    for hx in hexagons:
                        if self.pid_override is not None and hx not in self.pid_override:
                            print("pid problem", hx)
                            continue

                        print(">>>", ts, psi, hx, s_psi, numpy.sum(x_psi), numpy.sum(u_psi), numpy.sum(a_psi))

                        S.append(s_psi)
                        X.append(x_psi)
                        U.append(u_psi)
                        A.append(a_psi)
                        y.append([hx])
                        assids.append(i)
                        awgt.append([len(hexagons)])
                        psi_list.append(psi)
                        if(self.return_qhist):
                            qhist_list.append(gen_qhist(psi,ts))
                        else:
                            qhist_list.append(None)

            print(len(X), "in the pipe...")
            bs = self.batch_size
            if (bs == 0 and i != last_i) or ((bs > 0) and (len(X) >= bs)):
                if last_i is None:
                    last_i = i
                    continue  # special frist nop case
                print("b={}, n samples = {} ({}/{}={:.1f}%)".format(b, len(X), c, len_assts, (100.0 * c / len_assts)))
                b += 1
                yield S, X, U, A, y, assids, awgt, psi_list, qhist_list
                last_i = i
                S = []
                X = []
                U = []
                A = []

                y = []
                assids = []
                awgt = []
                psi_list = []
                qhist_list = []
                gc.collect()
        print("out of assts")
        yield S, X, U, A, y, assids, awgt, psi_list, qhist_list

def gen_semi_static(psi, dob_cache, ts_list):
    S_list = []
    raw_attempts = get_attempts_from_db(psi)
    first_non_empty = None
    # if raw_attempts.empty:
    #     print("student {} has no S attempts".format(psi))
    #     return []
    dob = None
    for ix,ts in enumerate(sorted(ts_list)):
        age=0
        xp_atts = 0
        sx = 0
        days = 1.0
        attempts = raw_attempts[raw_attempts["timestamp"] < ts]
        dob = dob_cache[psi]
        if dob is not None:
            age = (ts - dob).days / 365.242
        # if (not isinstance(age,float)) or (age>=100) or (age<10):
        #     age = 16.9

        if not attempts.empty:
            if first_non_empty is None:
                first_non_empty = ix
            # print("chex...")
            mindate = raw_attempts["timestamp"].min()
            days = (ts - mindate).days
            if numpy.isnan(days):
                input(days)
            #print("&&&& {}-{} = {}?".format(ts,mindate,days))
            days = 1.0 if days<1.0 else days

            #xp_runs = len(numpy.unique(attempts["question_id"]))
            xp_atts = attempts.shape[0]
            correct = attempts[attempts["correct"] == True]
            sx = correct.shape[0]
            # rat = sx/xp if xp>0 else 0
            # print("done...")
            # S_list.append( numpy.array([age, days, xp_runs, xp_atts, (xp_atts/days), (xp_runs/days), (sx/xp_atts if xp_atts else 0)]) ) #,rat,xp/days,sx/days]))
        S_list.append(numpy.array([age, days, (xp_atts/days), (sx/xp_atts if xp_atts else 0)]))
    S_list = S_list[max(first_non_empty - 1, 0):] if (not first_non_empty is None) else S_list[-1:]
    return S_list

def gen_experience(psi, ts_list, clip=True):
    raw_attempts = get_attempts_from_db(psi)
    X_list = []
    # if raw_attempts.empty:
    #     print("student {} has no X attempts".format(psi))
    #     return X_list
    first_non_empty = None
    for ix,ts in enumerate(sorted(ts_list)):
        X = numpy.zeros(len(all_qids))
        attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
        if not attempts.empty:
            if first_non_empty is None:
                first_non_empty = ix
        hits = attempts["question_id"]
        for qid in list(hits):
            try:
                qix = reverse_qid_dict[qid]
            except:
                print("UNK Qn ", qid)
                continue
            # X = numpy.max(X-.1,0)
            # X -= 0.02  # reduce to zero in 50 moves
            # X[X < 0] = 0.0 #bottom out at zero
            # X[X > 0] += 0.01
            X[qix] = 1
            # print("birdvs iirdvs", numpy.median(X), numpy.sum(X))
        # X_list.append(numpy.copy(X))
        X_list.append(X)
        # raw_attempts = raw_attempts[(raw_attempts["timestamp"] >= ts)]
    if clip:
        X_list = X_list[max(first_non_empty - 1, 0):] if (not first_non_empty is None) else X_list[-1:]
    return X_list

def gen_qhist(psi, ts):
    raw_attempts = get_attempts_from_db(psi)
    attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
    l1 = list(attempts["question_id"])
    l2 = list(attempts["timestamp"])
    qhist = list( zip(l1,l2) )
    return qhist

def gen_success(psi,ts_list, clip=True):
    raw_attempts = get_attempts_from_db(psi)
    U_list = []
    # if raw_attempts.empty:
    #     print("student {} has no X attempts".format(psi))
    #     return U_list

    first_non_empty = None
    for ix,ts in enumerate(sorted(ts_list)):
        U = numpy.zeros(len(all_qids))
        attempts = raw_attempts[(raw_attempts["timestamp"] < ts)]
        if not attempts.empty:
            if first_non_empty is None:
                first_non_empty = ix
        hits = attempts[(attempts["correct"] == True)]
        hits = hits["question_id"]
        for qid in list(hits):
            try:
                qix = reverse_qid_dict[qid]
            except:
                print("UNK Qn ", qid)
                continue
            attct = len(attempts[attempts["question_id"]==qid])
            U[qix] = (1.0/attct)
        U_list.append(U)
        #U_list.append(numpy.copy(U))
        #raw_attempts = raw_attempts[(raw_attempts["timestamp"] >= ts)]
    if clip:
        U_list = U_list[max(first_non_empty - 1, 0):] if (not first_non_empty is None) else U_list[-1:]
    return U_list