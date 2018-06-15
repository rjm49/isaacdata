import gc
import pickle
import zlib

import numpy
import pandas

from hwgen.common import init_objects, get_user_data, get_all_assignments, get_student_list, make_gb_question_map

cats, cat_lookup, all_qids, _, _, levels, cat_ixs, cat_page_lookup, lev_page_lookup, all_page_ids = init_objects(-1)

reverse_qid_dict = {}
for ix,q in enumerate(all_qids):
    reverse_qid_dict[q]=ix

from hwgen.profiler import get_attempts_from_db, get_age_df

profile_cache="../../../isaac_data_files/gengen_cache/"
base = "../../../isaac_data_files/"
prof_fname = base+"gengen_profiles.pkl"

LOAD_FROM_PROF_CACHE = True
SAVE_TO_PROF_CACHE = True

def ass_extract(ass):
    id = ass[1]["id"]
    ts = ass[1]['creation_date']
    gb_id = ass[1]["gameboard_id"]
    gr_id = ass[1]["group_id"]
    return id,ts,gb_id,gr_id


class hwgengen2:
    def __init__(self, assts, batch_size=512, pid_override=None, FRESSSH=False, qid_override=all_qids, return_qhist=False):
        self.assts: pandas.DataFrame = assts
        self.assts.loc[:,'creation_date'] = pandas.to_datetime(assts['creation_date'])

        self.gb_qmap = make_gb_question_map()

        self.batch_size=batch_size if batch_size!="assignment" else 0
        self.pid_override=pid_override
        self.qid_override = qid_override
        self.return_qhist = return_qhist

        if not FRESSSH:
            print("APPEND mode")
            #recycle old pap
            f = open(prof_fname, 'rb')
            self.profiles = pickle.load(f)
            print("got this many profiles:",len(self.profiles))
            # print(list(profiles.keys())[0:10])
            f.close()
        else:
            print("Baking FRESH, like cinnamon!")
            self.profiles = {}

        self.ts_cache = {}
        self.assid_list = []
        self.ts_master_list = []
        self.gb_id_list = []
        self.gr_id_list = []
        self.students_list = []
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

        for i,ts,gb_id,gr_id in zip(self.assid_list, self.ts_master_list, self.gb_id_list, self.gr_id_list):
            has_changed = False
            students = list(get_student_list(gr_id)["user_id"])
            for psi in students:  # set up the training arrays here
                fn = "prof_{}_{}".format(psi, ts)
                if fn not in self.profiles:
                    print("- - - -   profile for {} .. not found .. will create all ={}".format(psi, SAVE_TO_PROF_CACHE))
                    has_changed = True
                    group_df = get_user_data(students)
                    ts_list = self.ts_cache[psi]
                    print("ts_list", ts_list)
                    print("s..")
                    s_psi_list = gen_semi_static(psi, group_df, ts_list)
                    print("done")
                    print("x..")
                    x_psi_list = gen_experience(psi, ts_list)
                    print("done")
                    print("u..")
                    u_psi_list = gen_success(psi, ts_list)
                    print("done")
                    for ts,s_psi,x_psi,u_psi in zip(sorted(ts_list),s_psi_list,x_psi_list, u_psi_list):
                        loopvar = "prof_{}_{}".format(psi, ts)
                        self.profiles[fn] = zlib.compress(pickle.dumps((s_psi, x_psi, u_psi)))
                        print("created profile for ",loopvar, "xp=",numpy.sum(x_psi),"sxp=",numpy.sum(u_psi),"S=",s_psi)
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

                tripat = pickle.loads(zlib.decompress(self.profiles[fn]))
                if tripat is None:
                    print(fn, "gives none")
                else:
                    (s_psi, x_psi, u_psi) = tripat
                    for hx in hexagons:
                        if self.pid_override is not None and hx not in self.pid_override:
                            print("pid problem", hx)
                            continue

                        print(">>>", ts, psi, hx, s_psi, numpy.sum(x_psi), numpy.sum(u_psi))

                        S.append(s_psi)
                        X.append(x_psi)
                        U.append(u_psi)
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
                yield S, X, U, y, assids, awgt, psi_list, qhist_list
                last_i = i
                S = []
                X = []
                U = []
                y = []
                assids = []
                awgt = []
                psi_list = []
                qhist_list = []
                gc.collect()
        print("out of assts")
        yield S, X, U, y, assids, awgt, psi_list, qhist_list

def gen_semi_static(psi, group_df, ts_list):
    S_list = []
    raw_attempts = get_attempts_from_db(psi)
    # if raw_attempts.empty:
    #     print("student {} has no S attempts".format(psi))
    #     return []
    for ts in sorted(ts_list):
        age=None
        xp_atts = 0
        sx = 0
        days = 1.0
        attempts = raw_attempts[raw_attempts["timestamp"] <= ts]
        if not attempts.empty:
            # print("age gen...")
            age_df = get_age_df(ts, group_df)
            age = age_df.loc[psi, "age"]
            # print("done")

            # print("chex...")
            maxdate = (attempts["timestamp"]).max()
            mindate = (attempts["timestamp"]).min()
            days = (maxdate - mindate).days
            if numpy.isnan(days):
                input(days)
            days = 1.0 if days<1.0 else days

            #xp_runs = len(numpy.unique(attempts["question_id"]))
            xp_atts = attempts.shape[0]
            correct = attempts[attempts["correct"] == True]
            sx = correct.shape[0]
        if (type(age) is not numpy.float64) or (age>=100) or (age<10):
            age = 16.9
            # rat = sx/xp if xp>0 else 0
            # print("done...")
        # S_list.append( numpy.array([age, days, xp_runs, xp_atts, (xp_atts/days), (xp_runs/days), (sx/xp_atts if xp_atts else 0)]) ) #,rat,xp/days,sx/days]))
        S_list.append(numpy.array([age, days, (xp_atts/days), (sx/xp_atts if xp_atts else 0)]))
    return S_list

def gen_experience(psi, ts_list):
    raw_attempts = get_attempts_from_db(psi)
    X_list = []
    # if raw_attempts.empty:
    #     print("student {} has no X attempts".format(psi))
    #     return X_list
    X = numpy.zeros(len(all_qids))
    for ts in sorted(ts_list):
        attempts = raw_attempts[(raw_attempts["timestamp"] <= ts)]
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
        X_list.append(numpy.copy(X))
        raw_attempts = raw_attempts[(raw_attempts["timestamp"] > ts)]
    return X_list

def gen_qhist(psi, ts):
    raw_attempts = get_attempts_from_db(psi)
    attempts = raw_attempts[(raw_attempts["timestamp"] <= ts)]
    l1 = list(attempts["question_id"])
    l2 = list(attempts["timestamp"])
    qhist = list( zip(l1,l2) )
    return qhist

def gen_success(psi,ts_list):
    raw_attempts = get_attempts_from_db(psi)
    U_list = []
    # if raw_attempts.empty:
    #     print("student {} has no X attempts".format(psi))
    #     return U_list
    U = numpy.zeros(len(all_qids))
    for ts in sorted(ts_list):
        attempts = raw_attempts[(raw_attempts["timestamp"] <= ts)]
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
        U_list.append(numpy.copy(U))
        raw_attempts = raw_attempts[(raw_attempts["timestamp"] > ts)]
    return U_list