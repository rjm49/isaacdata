import hashlib
import json
import os
import pickle
import sys
from collections import OrderedDict
from os.path import dirname, abspath

import numpy
import pandas
import psycopg2
from psycopg2 import OperationalError

os.chdir(dirname(abspath(__file__))) #RESET PATHS TO THIS DIR

n_concepts = 101
n_components = 35
cache_dir = "../../../isaac_data_files/data_cache/"
def test_db_connexion():
    with open('./db_config.json') as json_data_file:
        config = json.load(json_data_file)
        db_config = config["postgresql"]
    try:
        conn = psycopg2.connect(database=db_config["db"], user=db_config["user"], password=db_config["passwd"],
                            host=db_config["host"], port=db_config["port"])
    except:
        return False
    return True
# DATABASE = test_db_connexion()
DATABASE=True
LOAD_FROM_CACHE = True
SAVE_TO_CACHE = True

def get_user_data(user_id_list):
    if user_id_list == "*":
        query = "select * from users"
        name = "all_users.csv"
    else:
        commaspaced = ",".join(map(str, user_id_list))
        query = "select * from users where id in ({})".format(commaspaced)
        h4sh = hashlib.md5(commaspaced.encode()).hexdigest()
        # h4sh = pickle.dumps(commaspaced)
        name = "users_{}.csv".format(h4sh)
    # print(query)
    df = make_db_call(query,name)
    df["date_of_birth"] = pandas.to_datetime(df["date_of_birth"])
    return df

def get_student_list(gr_id_list):
    try:
        L = ",".join(map(str, gr_id_list))
    except TypeError as te:
        L = gr_id_list

    query = "select user_id from group_memberships where group_id in ({})".format(L)
    # print(query)
    name = "student_list_{}.csv".format(L)
    return make_db_call(query,name)

def get_group_list(t_id):
    query = "select id from groups where owner_id = {}".format(t_id)
    name = "group_list_{}.csv"
    return make_db_call(query,name)

def extract_runs_w_timestamp_df2(attempts, pv_ts=None, this_ts=None):
    times = []
    qids = []
    cors = []
    lens = []
    num_correct = 0
    num_attempts = 0
    run_qid = None

    tm=None
    for ix, item in attempts.iterrows():
        # we want to isolate contiguous atts against a question into "qids" and determine whether each run was successful or not
        # since students can repeatedly run against a question, it is not sufficient just to filter on the question ID
        new_run_qid = item["question_id"]
        is_corr = item["correct"]
        tm = item["timestamp"]

        if (new_run_qid != run_qid):
            #             print("new run")
            qids.append(run_qid)
            lens.append(num_attempts)
            cors.append(num_correct)
            times.append(tm)
            run_qid = new_run_qid
            num_correct = 1 if is_corr else 0
            num_attempts = 1
        else:
            num_attempts += 1
            num_correct += (1 if is_corr else 0)

    qids.append(run_qid)
    lens.append(num_attempts)
    cors.append(num_correct)
    times.append(tm)

    if tm is None:
        return None

    uni = list(zip(times, qids, lens, cors))
    return uni[1:]  # lose the first entry, which is a dummy


def make_db_call(query, name):
    if LOAD_FROM_CACHE:
        try:
            fullname = cache_dir + name
            df = pandas.read_csv(fullname)
            #print("db cache hit! {}".format(name))
            return df
        except:
            if not DATABASE:
                print("No database conn and no cached version of ({}) found!".format(fullname))
                exit(1)

    with open('./db_config.json') as json_data_file:
        config = json.load(json_data_file)
        db_config = config["postgresql"]
    # print(db_config)

    conn = psycopg2.connect(database=db_config["db"], user=db_config["user"], password=db_config["passwd"],
                            host=db_config["host"], port=db_config["port"])
    curs = conn.cursor()
    # print("Opened database successfully")
    try:
        curs.execute(query)
        print("db hit! {}".format(name))
    except OperationalError as msg:
        print("Command skipped: ", msg)
    col_names = [i[0] for i in curs.description]
    # curs.fetchone()
    df = pandas.DataFrame(curs.fetchall(), columns=col_names)

    if SAVE_TO_CACHE:
        dir = os.path.dirname(cache_dir)
        if not os.path.exists(dir):
                os.makedirs(dir)
        df.to_csv(cache_dir+name)
    return df

def get_page_concepts(pid):
    query = "select distinct related_concepts from content_data where page_id = '{}'".format(pid)
    name = "concepts_{}.csv".format(pid)
    raw_df = make_db_call(query,name)
    if raw_df.iloc[0,0] is None:
        return []
    return list( raw_df.iloc[0,0] )

def get_q_names():
    query = "select question_id, title from content_data"
    name = "q_names.csv"
    return make_db_call(query,name)

def get_meta_data():
    query = "select question_id, page_id, level, subject, field, topic, related_questions, related_concepts, detailed_concept_sections from content_data"
    name = "meta_data.csv"
    return make_db_call(query,name)

def get_all_assignments():
    query = "select id, gameboard_id, group_id, owner_user_id, creation_date from assignments order by creation_date asc"
    name = "gb_assignments2.csv"
    return make_db_call(query,name)

def make_gb_question_map():
    query = "select id, questions from gameboards"
    raw_df = make_db_call(query,"gb_q_map.csv")
    map = {}
    for r in raw_df.iterrows():
        gb_id = r[1]["id"]
        qs = r[1]["questions"] # TODO must we eval()?
        if type(qs) is str:
            qs = eval(qs)
        map[gb_id] = qs
    return map

def init_objects(n_users, path="./config_files/", seed=None):
    qmeta = get_meta_data()

    cats = []
    diffs = OrderedDict()

    levels = OrderedDict()
    cat_lookup = OrderedDict()
    cat_ixs = OrderedDict()
    all_qids = []
    all_page_ids = []
    for r in qmeta.iterrows():
        r = r[1] # get the data part of the tuple
        q_id = r["question_id"]
        if q_id not in all_qids:
            all_qids.append(q_id)
        p_id = r["page_id"]
        if p_id not in all_page_ids:
            all_page_ids.append(p_id)
        cat = str(r["subject"]) + "/" + str(r["field"]) + "/" + str(r["topic"])
        # cat = r["category"]
        cats.append(cat)
        cat_lookup[q_id] = cat
        # diff_raw = r[DIF]
        # diffs[q_id] = -1 if (diff_raw == float("inf")) else diff_raw

        lv = numpy.float(r["level"])
        levels[q_id] = 0 if numpy.isnan(lv) else lv

    # replace any negative (formerlly inf) values with the max system difficulty
    # max_diff = max(diffs.values())
    # for k, v in diffs.items():
    #     if v < 0:
    #         diffs[k] = max_diff

    cats = list(set(cats))
    for ix, c in enumerate(cats):
        cat_ixs[c] = ix

    cat_page_lookup = {}
    for k in cat_lookup.keys():
        qpage = k.split("|")[0] #tODO tildas no longer in use!
        qsft = cat_lookup[k]
        if qpage not in cat_page_lookup:
            cat_page_lookup[qpage] = qsft

    lev_page_lookup = {}
    for k in levels.keys():
        qpage = k.split("|")[0]
        L = levels[k]
        if qpage not in lev_page_lookup:
            lev_page_lookup[qpage] = L

    return cats, cat_lookup, list(all_qids), None, None, levels, cat_ixs, cat_page_lookup, lev_page_lookup, list(all_page_ids)


def get_n_hot(name_list, page_ids):
    arr = numpy.zeros(len(page_ids))
    if type(name_list) is str:
        name_list = [name_list]
    for n in name_list:
        if n in page_ids:
            ix = page_ids.index(n)
            arr[ix] = 1.0
    return arr

def split_assts(assts, ass_n, split, NO_META=True, pid_override=None):
    # kl = list(assts.keys())
    # for k in kl:
    #     if len(assts[k][3]) != 1:
    #         del assts[k]
    # print(len(assts), "single hx assgts")

    kl = list(assts.keys())
    for k in kl:
        item = assts[k]
        if pid_override is not None:
            for i in item[3]:
                print(item[3])
                if i not in pid_override:
                    del assts[k]
                    break

    kl = list(assts.keys())
    for k in kl:
        item = assts[k]
        if NO_META:
            if sum(item[5]) > 0:
                del assts[k]
        else:
            if 0 == sum(item[5]):
                del assts[k]
    print(len(assts), "filtered hx assgts")

    tt = []
    tr = []
    kl = list(assts.keys())[:ass_n]
    # print(kl[0:10])
    # input("yam")
    for i, k in enumerate(kl):
        if i < split:
            tt.append(assts[k])
        else:
            tr.append(assts[k])
    return tr,tt

def jaccard(pred_in, Bin, con_page_lookup):
    A = set()
    B = set()

    for id in pred_in:
        cs : list = con_page_lookup[id]
        A.update(cs)

    for id in Bin:
        cs : list = con_page_lookup[id]
        B.update(cs)

    print(A,"\n\t",B)
    if len(A)==0 and len(B)==0:
        return 1.0
    else:
        return len( A.intersection(B)) / len(A.union(B))
