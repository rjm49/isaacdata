import datetime

import numpy
import pandas as pd

from backfit.BackfitUtils import init_objects
from hwgen.concept_extract import concept_extract
from irt.irt_engine import IRTEngine
from isaac.itemencoding import gen_X_primed, gen_qenc, s_features, n_components, k_features, n_concepts
from utils.utils import extract_runs_w_timestamp, extract_runs_w_timestamp_df


def profile_student(psi, age, ts, cats, cat_lookup, cat_ixs, levels, concepts_all, df, cache):
    return profile_student_enc(psi, age, ts, cats, cat_lookup, cat_ixs, levels, concepts_all, df, cache)

def profile_student_irt(u, ass_ts, cats, cat_lookup, cat_ixs, levels, concepts_all):
    #load student's files
    base = "../../../isaac_data_files/"
    #cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(-1, path=base, seed=666)
    df = pd.read_csv(base + "hwgen1.csv", index_col=0, header=0)
    #runs = open(base + "by_runs/{}.txt".format(u)).readlines()
    fname = base+"by_user/{}.txt".format(u)
    try:
        attempts = pd.read_csv(fname, header=None)
    except FileNotFoundError:
        return []
    runs = extract_runs_w_timestamp(attempts)

    u_run_ct = len(runs)
    all_zero_level = True
    run_ct=0

    irts = {}
    subj_irts = {}
    print("num of runs", len(runs))
    cat_levels = [0] * len(cats)
    concept_levels = [0] * 100

    for run_ix, run in enumerate(runs):
        run_ct += 1
        ts, q, n_atts, n_pass = run
        ts = pd.to_datetime(ts)
        # print("rum", run_ct, ts)

        if ts > ass_ts:
            break
        # print(ts, "<=", ass_ts)
        qt = q.replace("|","~")
        cat = cat_lookup[qt]
        lev = levels[qt]

        if q not in df.index:
            continue
        concepts_raw = df.loc[q, "related_concepts"]

        concepts = eval(concepts_raw) if not pd.isna(concepts_raw) else []
        for c in concepts:
            if c not in irts:
                print("at run", run_ct, "new irt engine for", u,c)
                irts[c] = IRTEngine()
            else:
                irt = irts[c]
                irt.curr_theta = irt.update(lev, (n_pass > 0))
                irts[c] = irt
                print(u, c,"history =", irt.history)
                print("at run", run_ct, "irt update", u, c, irt.curr_theta)

        if cat not in subj_irts:
            subj_irts[cat] = IRTEngine()
        else:
            subj_irts[cat].curr_theta = subj_irts[cat].update(lev, (n_pass > 0))
            print("at run", run_ct, "irt cat update", u, cat, subj_irts[cat].curr_theta)

        for s in subj_irts:
            catix = cat_ixs[cat]
            theta = subj_irts[s].curr_theta
            print(s,"=",theta)
            cat_levels[catix]=theta
        if irts:
            print("\nConcept level abilities:")
        for c in irts:
            theta = irts[c].curr_theta
            print(c,"=",theta)
            conix = concepts_all.index(c)
            concept_levels[conix]=theta

#    concatd = cat_levels + concept_levels
    concatd = concept_levels
    print("Profile for user {}: {}".format(u, concatd))
    if concatd==[]:
        print("empty")
    return concatd

def profile_student_enc(u, age, ass_ts, cats, cat_lookup, cat_ixs, levels, concepts_all, df, cache):
    #load student's files
    base = "../../../isaac_data_files/"
    #cats, cat_lookup, all_qids, users, _stretches_, levels, cat_ixs = init_objects(-1, path=base, seed=666)

    if u not in cache:
        print("Neŧ!")
        fname = base+"by_user_df/{}.csv".format(u)
        S = numpy.zeros(shape=4)
        X = numpy.zeros(shape=(n_components, 2))  # init'se a new feature vector w same width as all_X
        C = numpy.zeros(shape=(n_concepts,2))
        C[:] = -1
        X[:] = -1

        try:
            attempts = pd.read_csv(fname, header=0)
        except FileNotFoundError:
            print("File not found for student", u)
            return []

        attempts["timestamp"] = pd.to_datetime(attempts["timestamp"])
        attempts.drop(attempts.columns[0], axis=1, inplace=True)
        pv_ts = pd.to_datetime("1970-01-01")
        runs = extract_runs_w_timestamp_df(attempts)
    else:
        runs, pv_ts, S,X,C = cache[u]

    # attempts = attempts[(attempts.timestamp > pv_ts)]
    if runs is None:
        return []

    u_run_ct = len(runs)
    all_zero_level = True
    run_ct=0

    phi = 1.0
    SS_XP_IX = 0
    SS_SUCCESS_IX = 1
    SS_FAILED_IX = 2
    SS_AGE_IX = 3

    LEVEL_IX=0
    TIME_IX=1
    X[:, TIME_IX] = -1

    S[SS_AGE_IX]=age

    fade = 0.99

    for run_ix, run in enumerate(runs):
        run_ct += 1

        ts, q, n_atts, n_pass = run
        # print("rum", run_ct, ts)

        # print(ts, "<=", ass_ts)
        if ts <= pv_ts:
            runs.remove(run)
            continue #skip over stuff we've seen

        if ts > ass_ts:
            break

        qt = q.replace("|","~")
        if qt not in cat_lookup:
            continue

        cat = cat_lookup[qt]
        catix = cat_ixs[cat]
        lev = levels[qt]

        if q not in df.index:
            continue
        concepts_raw = df.loc[q, "related_concepts"]
        concepts = eval(concepts_raw) if not pd.isna(concepts_raw) else []

        conixes = [concepts_all.index(c) for c in concepts]

        for conix in conixes:
            # C[conix] = 1.0
            if C[conix, LEVEL_IX] < 0:
                C[conix, LEVEL_IX] = 0
                C[conix, TIME_IX] = 0

        Xcat = X[catix]
        if Xcat[LEVEL_IX] < 0:
            Xcat[LEVEL_IX] = 0
        # LEVEL_IX = 0
        # TIME_IX = 1
        S[SS_XP_IX] = run_ct
        if (n_pass > 0):
            for conix in conixes:
                if lev + 1 > C[conix, LEVEL_IX]:
                    # C[conix] = max(C[conix], (1+lev))
                    C[conix, LEVEL_IX] = (C[conix, LEVEL_IX] + 1 + lev) / 2.0

            S[SS_SUCCESS_IX] += 1
            if (lev+1) > Xcat[LEVEL_IX]:
                Xcat[LEVEL_IX] = (1 + lev + Xcat[LEVEL_IX]) / 2.0
                # Xcat[LEVEL_IX] = max(Xcat[LEVEL_IX], (lev + 1))

        else:
            S[SS_FAILED_IX] += 1

        # S[SS_PASS_RATE_IX] = (run_ct-S[SS_FAILED_IX])/run_ct if run_ct!=0.0 else -1.0
        # X[:,TIME_IX] *= fade
        C[(C[:,TIME_IX]>=0), TIME_IX] += 1.0
        X[(X[:,TIME_IX]>=0), TIME_IX] += 1.0
        # X[catix, TIME_IX] = 1.0

        # X[:,TIME_IX]+=1 # increment the counter for all cat rows
        Xcat[TIME_IX] = 0 # then reset the counter for this cat row
        # X[catix] = Xcat

    # print("num new runs = {}".format(run_ct))

    # concatd = list(S.flatten()) + list(X.flatten()) + list(C.flatten())
    concatd = list(S.flatten()) + list(C.flatten())
    # print("Profile for user {}: {}".format(u, concatd))
    # print("Profile for user {}: {}".format(u,"".join(map(str, concatd))))

    cache[u] = runs, ass_ts, S, X, C #update the cache wuth the new values

    return concatd


