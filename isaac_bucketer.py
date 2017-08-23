import pandas as pd
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

prebucket=True
if __name__ == '__main__':
    fhandles = {}
    users = set()
    qmeta = OrderedDict()
    #fout = open("bucketted_by100.csv", "w")
    if(prebucket):
        for df in pd.read_csv('qs.new2.txt', encoding='utf-8',header=0 ,engine = 'c', low_memory=True, chunksize=100000):
            df = df[df['event_type'] == "ANSWER_QUESTION"]
            q_ids = df.question_id.unique()
            for q_id in q_ids:
                rows = df[df['question_id'] == q_id]
                row0 = list(rows.itertuples())[0]
                if q_id not in fhandles:
                    qmeta[q_id] = OrderedDict([
                                   ("level",row0.level),
                                   ("subject",row0.subject),
                                   ("field",row0.field),
                                   ("topic",row0.topic),
                                   ("n_atts",0),
                                   ("n_pass",0)
                                   ])
                    fname = "./sep/{}.txt".format(q_id.replace("|","~"))
                    print("new file handler",fname)
                    fhandles[q_id] = open(fname,"w") 
                else:
                    print("reopening",fhandles[q_id].name)
                    fhandles[q_id] = open(fhandles[q_id].name,"w+")
                for row in list(rows.itertuples()):
                    users.add(row.user_id) # each user added once bc Set 
                    qmeta[q_id]["n_atts"]+=1
                    if row.correct:
                        qmeta[q_id]["n_pass"]+=1
                    fhandles[q_id].write(str(row.timestamp) +","+ str(row.correct)+","+str(row.user_id)+"\n")            
            for handle in fhandles.values():
                handle.close()
    
    metafile = open("qmeta.csv","w")
    for k in qmeta.keys():
        d = qmeta[k]
        ratio = d["n_atts"]/numpy.float64(d["n_pass"])
        metafile.write(",".join([k.replace("|","~")]+[str(x) for x in d.values()]) +","+ str(ratio) +"\n")
    metafile.close()
    
    ufile = open("users.csv","w")
    for u in users:
        ufile.write(u+"\n")
    ufile.close()
    print("done")