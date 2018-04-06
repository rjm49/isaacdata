import os
import codecs
import csv

import pandas
import datetime

from database_scripts.dbase_handler import get_group_deets

'''
Created on 16 Aug 2017

@author: Russell
'''
from collections import Counter, OrderedDict
from _collections import defaultdict
import numpy

q_dict = {}
q_counter = defaultdict(list)
p_counter = defaultdict(list)
q_id=None
max_n = 100

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def process_csv_chunk(df, fout):
    df = df[df['event_type'] == "ANSWER_QUESTION"]
    q_ids = df.question_id.unique()
    #print(df)
    
    for q_id in q_ids:
        print(q_id)
        rows = df[df['question_id'] == q_id]
        for group in chunker(list(rows.itertuples()), 100):
            cnt=0
            pss=0
            for row in group:
                cnt+=1
                if(row.correct):
                    pss+=1
            #print("s")
            fout.write(q_id + "," + str(cnt) + "," + str(pss) +"\n")




prebucket=False
bucket=True
base = "/home/rjm49/isaac_data_files/"



if __name__ == '__main__':

    base = "../../isaac_data_files/"

    # need to build a softmax classifier to recommend questions...
    # this would have qn output nodes

    testing_only = False
    # get assignments ....
    # ass_df = pandas.read_csv(base + "gb_assignments.csv")
    # gbd_df = pandas.read_csv(base + "gameboards.txt", sep="~")
    grp_df = pandas.read_csv(base + "group_memberships.csv")

    students = grp_df["user_id"]

    # for each student in class
    profiles = {}
    ss = []
    for psi in students:
        fname = "./by_user/{}.csv".format(psi)
        if os.path.isfile(fname):
            print(fname, "already exists, skipping")
        else:
            ss.append(psi)


    sample_size = 100
    while len(ss)>0:
        if len(ss)>sample_size:
            sample = ss[0:sample_size]
            ss = ss[sample_size:]
        else:
            sample=ss
            ss=[]

        psi_df = get_group_deets(sample)
        for psi in sample:
            fname = "./by_user/{}.csv".format(psi)
            out_df = psi_df[0]
            out_df = out_df[out_df["user_id"]==str(psi)]
            print(len(out_df))
            out_df.to_csv(fname, sep=",")
            print("wrote for ",psi)