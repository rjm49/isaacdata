import codecs
import csv

import pandas as pd
import datetime
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
    if(prebucket):
        fout = codecs.open(base+"raw/user_activity_df.csv","w+")
        f = codecs.open(base+"raw/isaac_MC_data_all","r", encoding='utf-8', errors='ignore')
        header = f.readline().replace(" ", "")
        fout.write(header)
        while True:
            line = f.readline()
            if not line : break # stop the loop if no line was read
            print(line,"\n")
            line = eval(line)
            fout.write( ",".join( map(str,line))  + "\n" )
        f.close() # after the loop, not in it
        fout.close()
        exit()

    fhandles = {}
    qmeta = OrderedDict()

    #users = open("./users.csv").read().splitlines()
    f=0
    #fout = open("bucketted_by100.csv", "w")
    if(bucket):
        # for df in pd.read_csv(base+'raw/user_activity_df.csv', encoding='utf-8',header=0, chunksize=20000):
            df = pd.read_csv(base+'raw/user_activity_df.csv', quoting=csv.QUOTE_NONE, header=0)
            df = df[df['event_type'] == "ANSWER_QUESTION"]
            df = df[df['level'] != 0 ]
            us = df['user_id'].unique()#[0:1000]

            for u in us:
                rows = df[df['user_id'] == u]
                #print(f,"file open")
                if u not in fhandles:
                    fname = "./by_user/{}.txt".format(u)
                    print("new file handler",fname)
                    fhandles[u] = open(fname,"w+")
                    #f+=1
                else:
                    if fhandles[u].closed:
                        print("reopening",fhandles[u].name)
                        fhandles[u] = open(fhandles[u].name,"a")
                        #f+=1
                for row in list(rows.itertuples()):
                    fhandles[u].write(str(row.timestamp) +","+ str(row.correct)+","+str(row.question_id)+"\n")
                fhandles[u].close()

    #                 if(f>2000):
    #                     for h in fhandles.values():
    #                         h.close()
    #                         f=0

    print("done")