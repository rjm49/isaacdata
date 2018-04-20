import json
from collections import OrderedDict

import numpy
import pandas
import psycopg2
from psycopg2._psycopg import OperationalError

n_concepts = 101
n_components = 35

def get_user_data(user_id_list):
    if user_id_list == "*":
        query = "select * from users"
    else:
        commaspaced = ",".join(map(str, user_id_list))
        query = "select * from users where id in ({})".format(commaspaced)
    # print(query)
    raw_df = make_db_call(query)
    raw_df["date_of_birth"] = pandas.to_datetime(raw_df["date_of_birth"])
    return raw_df

def get_student_list(gr_id_list):
    try:
        L = ",".join(map(str, gr_id_list))
    except TypeError as te:
        L = gr_id_list

    query = "select user_id from group_memberships where group_id in ({})".format(L)
    # print(query)
    return make_db_call(query)

def get_group_list(t_id):
    query = "select id from groups where owner_id = {}".format(t_id)
    return make_db_call(query)


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


def make_db_call(query):
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
        print("db hit!")
    except OperationalError as msg:
        print("Command skipped: ", msg)
    col_names = [i[0] for i in curs.description]
    # curs.fetchone()
    df = pandas.DataFrame(curs.fetchall(), columns=col_names)
    return df

def get_meta_data():
    query = "select question_id, page_id, level, subject, field, topic, related_questions, related_concepts, detailed_concept_sections from content_data where published=TRUE"
    df = make_db_call(query)
    # df["category"] = df["subject"].map(str) + "/" + df["field"] + "/" + df["topic"]
    return df

def init_objects(n_users, path="./config_files/", seed=None):
    qmeta = get_meta_data()

    cats = []
    diffs = OrderedDict()

    levels = OrderedDict()
    cat_lookup = OrderedDict()
    cat_ixs = OrderedDict()
    all_qids = set()
    all_page_ids = set()
    for r in qmeta.iterrows():
        r = r[1] # get the data part of the tuple
        q_id = r["question_id"]
        all_qids.add(q_id)
        p_id = r["page_id"]
        all_page_ids.add(p_id)
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
        qpage = k.split("~")[0] #tODO tildas no longer in use!
        qsft = cat_lookup[k]
        if qpage not in cat_page_lookup:
            cat_page_lookup[qpage] = qsft

    lev_page_lookup = {}
    for k in levels.keys():
        qpage = k.split("~")[0]
        L = levels[k]
        if qpage not in lev_page_lookup:
            lev_page_lookup[qpage] = L

    return cats, cat_lookup, list(all_qids), None, None, levels, cat_ixs, cat_page_lookup, lev_page_lookup, list(all_page_ids)
