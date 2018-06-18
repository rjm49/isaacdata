import datetime
import pickle
from math import nan

import numpy
import pandas as pd
from pandas._libs.tslib import NaTType

from hwgen.common import extract_runs_w_timestamp_df2, n_components, n_concepts, init_objects, make_db_call

profile_cache="../../../isaac_data_files/profile_cache/"
LOAD_FROM_CACHE = False
SAVE_TO_CACHE = True

cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs, cat_page_lookup, level_page_lookup, all_page_ids = init_objects(-1)

def profile_student(psi, age, ts, cats, cat_lookup, cat_ixs, levels, concepts_all, df, cache, attempts_df=None):
    pf = profile_student_enc(psi, age, ts, cats, cat_lookup, cat_ixs, levels, concepts_all, df, cache, attempts_df)
    return pf

# def profile_student_irt(u, ass_ts, cats, cat_lookup, cat_ixs, levels, concepts_all):
#     #load student's files
#     base = "../../../isaac_data_files/"
#     #cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(-1, path=base, seed=666)
#     df = pd.read_csv(base + "hwgen1.csv", index_col=0, header=0)
#     #runs = open(base + "by_runs/{}.txt".format(u)).readlines()
#     fname = base+"by_user/{}.txt".format(u)
#     try:
#         attempts = pd.read_csv(fname, header=None)
#     except FileNotFoundError:
#         return []
#     runs = extract_runs_w_timestamp(attempts)
#
#     u_run_ct = len(runs)
#     all_zero_level = True
#     run_ct=0
#
#     irts = {}
#     subj_irts = {}
#     print("num of runs", len(runs))
#     cat_levels = [0] * len(cats)
#     concept_levels = [0] * 100
#
#     for run_ix, run in enumerate(runs):
#         run_ct += 1
#         ts, q, n_atts, n_pass = run
#         ts = pd.to_datetime(ts)
#         # print("rum", run_ct, ts)
#
#         if ts > ass_ts:
#             break
#         # print(ts, "<=", ass_ts)
#         qt = q.replace("|","~")
#         cat = cat_lookup[qt]
#         lev = levels[qt]
#
#         if q not in df.index:
#             continue
#         concepts_raw = df.loc[q, "related_concepts"]
#
#         concepts = eval(concepts_raw) if not pd.isna(concepts_raw) else []
#         for c in concepts:
#             if c not in irts:
#                 print("at run", run_ct, "new irt engine for", u,c)
#                 irts[c] = IRTEngine()
#             else:
#                 irt = irts[c]
#                 irt.curr_theta = irt.update(lev, (n_pass > 0))
#                 irts[c] = irt
#                 print(u, c,"history =", irt.history)
#                 print("at run", run_ct, "irt update", u, c, irt.curr_theta)
#
#         if cat not in subj_irts:
#             subj_irts[cat] = IRTEngine()
#         else:
#             subj_irts[cat].curr_theta = subj_irts[cat].update(lev, (n_pass > 0))
#             print("at run", run_ct, "irt cat update", u, cat, subj_irts[cat].curr_theta)
#
#         for s in subj_irts:
#             catix = cat_ixs[cat]
#             theta = subj_irts[s].curr_theta
#             print(s,"=",theta)
#             cat_levels[catix]=theta
#         if irts:
#             print("\nConcept level abilities:")
#         for c in irts:
#             theta = irts[c].curr_theta
#             print(c,"=",theta)
#             conix = concepts_all.index(c)
#             concept_levels[conix]=theta
#
# #    concatd = cat_levels + concept_levels
#     concatd = concept_levels
#     print("Profile for user {}: {}".format(u, concatd))
#     if concatd==[]:
#         print("empty")
#     return concatd


def get_attempts_from_db(u):
    query = "select user_id, event_details->>'questionId' AS question_id, event_details->>'correct' AS correct, timestamp from logged_events where user_id in ({}) and event_type='ANSWER_QUESTION'"
    wrap = lambda w : "'{0}'".format(w)
    name = "attempts_{}.csv".format(u)
    u = wrap(u)
    query = query.format(u)
    raw_df =  make_db_call(query, name)
    # TODO Maybe do stuff to raw_df ??? Profit!
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    return raw_df

from_db = True


def get_age_df(ts, gr_df):
    #Start by setting the system default age
    default_age = 16.9
    DPY = 365.242

    genesis = pd.to_datetime("1970-01-01")
    dobseries = gr_df[(gr_df.role == "STUDENT")]["date_of_birth"]
    dobseries.dropna(inplace=True)

    class_avg_del = (dobseries - genesis).median()
    if class_avg_del is not pd.NaT:
        class_avg_dob = class_avg_del + genesis
        class_avg_age = (ts - class_avg_dob).days / DPY
    else:
        class_avg_dob = (ts - pd.DateOffset(days=int(DPY*default_age)))
        class_avg_age = default_age

    age_df = pd.DataFrame(index=gr_df["id"], columns=["dob","delta","age"])
    dobs = gr_df["date_of_birth"]
    age_df["dob"] = list(dobs)
    age_df["delta"] = list((ts - age_df["dob"]).astype('timedelta64[D]'))
    age_df["age"] = list(age_df["delta"] / DPY)
    age_df.loc[(age_df["age"]>100), "age"] = class_avg_age
    age_df.loc[(age_df["age"]<0) , "age"] = class_avg_age
    # age_df[numpy.isnan(age_df["age"])]["age"] = class_avg_age


    age_df["dob"].replace(pd.NaT, class_avg_dob, inplace=True)

    age_df["age"].replace(numpy.NaN, class_avg_age, inplace=True)
    age_df["age"].replace(nan, class_avg_age, inplace=True)
    return age_df


def profile_students(student_list, profile_df, up_to_ts, concepts_all, hwdf, user_cache, attempts_df):
    if student_list == []:
        return {}

    profiles = {}
    age_df = get_age_df(up_to_ts, profile_df)

    for psi in student_list:
        fn = profile_cache+"prof_{}_{}".format(psi, up_to_ts)
        loaded_from_cache = False
        if LOAD_FROM_CACHE:
            try:
                with open(fn, "rb") as c:
                    pf = pickle.load(c)
                    #print("cached profile {} loaded!".format(fn))
                    print("c", end="")
                    loaded_from_cache = True
            except:
                print("Looked for file {} .. not found .. will create={}".format(fn, SAVE_TO_CACHE))

        if not loaded_from_cache:
            age = age_df.loc[psi,"age"]
            if type(age) is not float:
                age = 16.9
            assert age < 100
            assert age > 0
            pf = profile_student(psi, age, up_to_ts, cats, cat_lookup, cat_ixs, levels, concepts_all, hwdf, user_cache, attempts_df)
            # print("*{} {}".format(psi, pf))

            if SAVE_TO_CACHE:
                with open(fn, "wb") as c:
                    pickle.dump(pf, c)

        assert pf is not None
        if (pf is not None):  # ie. if not empty ... TODO why do we get empty ones??
            profiles[psi] = pf
        # train softmax classifier with student profile and assignment profile
    return profiles

def profile_student_enc(u, age, ass_ts, cats, cat_lookup, cat_ixs, levels, concepts_all, df, cache, attempts_df=None):
    #load student's files
    base = "../../../isaac_data_files/"

    if u not in cache:
        # print("Neŧ!")
        S = numpy.zeros(shape=6)
        Q = numpy.zeros(shape=len(all_qids))
        T = numpy.zeros(shape=len(all_qids))
        L = numpy.zeros(shape=7) # level tracking
        S[:] = 0
        Q[:] = 0
        L[:] = 0
        atts_ct = 0
        pazz_ct = 0

        attempts = get_attempts_from_db(u)
        attempts.loc[:,"timestamp"] = pd.to_datetime(attempts.loc[:,"timestamp"])
        pv_ts = pd.to_datetime("1970-01-01")
        runs = extract_runs_w_timestamp_df2(attempts)
    else:
        runs, pv_ts, S,Q,T,L, atts_ct, pazz_ct = cache[u]

    if runs is None:
        runs = []

    fade = 0.999
    run_ct=0
    SS_AGE_IX = 0
    SS_XP_IX = 1
    SS_PASS_IX = 2
    SS_PRATE_IX = 3
    SS_ATT_LV_IX = 4
    SS_SUCC_LV_IX = 5
    S[SS_AGE_IX]=age

    for run_ix, run in enumerate(runs):
        run_ct += 1
        ts, q, n_atts, n_pass = run
        if ts <= pv_ts:
            runs.remove(run)
            continue #skip over stuff we've seen
        if ts > ass_ts:
            break
        qt = q
        page_id = q.split("|")[0]
        if qt not in cat_lookup:
            continue
        cat = cat_lookup[qt]
        catix = cat_ixs[cat]
        lev = int(levels[qt])
        if q not in df.index:
            continue
        else:
            qix = df.index.get_loc(q)

        S[SS_ATT_LV_IX] = max(lev,S[SS_ATT_LV_IX])
        # S[SS_RAVG_ATT_LV_IX]=0.66*S[SS_ATT_LV_IX] + 0.33*lev
        S[SS_XP_IX] = S[SS_XP_IX]+1
        # T *= fade
        # T[qix] = 1 # time since q att
        atts_ct += 1
        if (n_pass > 0):
            Q[qix] = 1
            S[SS_SUCC_LV_IX] = max(S[SS_SUCC_LV_IX],lev) #0.66*S[SS_SUCC_LV_IX] + 0.33*lev
            pazz_ct += 1
            S[SS_PASS_IX] = S[SS_PASS_IX]+1
            S[SS_PRATE_IX] = (pazz_ct / atts_ct) if atts_ct>0 else 1.0
            L[lev]+=1
        else:
            Q[qix] = -1

    # concatd = list(S.flatten()) + list(Q.flatten()) #+ list(L.flatten())
    cache[u] = runs, ass_ts, S,Q,T,L, atts_ct, pazz_ct #update the cache wuth the new values
    return S.flatten(),Q.flatten(),L.flatten()

