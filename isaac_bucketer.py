import numpy
import pandas as pd
'''
Created on 16 Aug 2017

@author: Russell
'''
import os
from collections import Counter, OrderedDict
from _collections import defaultdict

q_dict = {}
q_counter = defaultdict(list)
p_counter = defaultdict(list)
q_id=None
# max_n = 100

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

write_main = False
write_qmeta=False
write_users=False
write_atypes=True
prebucket=True
if __name__ == '__main__':
    fhandles = {}
    qids = set()
    users = set()
    qmeta = OrderedDict()
    #fout = open("bucketted_by100.csv", "w")    
    if(prebucket):
        for df in pd.read_csv('qs.new2.txt', encoding='utf-8',header=0 ,engine = 'c', low_memory=True, chunksize=100000):
            df = df[df['event_type'] == "ANSWER_QUESTION"]
            q_ids = df.question_id.unique()
            for q_id in q_ids:
                if q_id not in qids:
                    rows = df[df['question_id'] == q_id]
                    row0 = list(rows.itertuples())[0]
                    qids.add(q_id)
                    print("added",q_id, len(qids))
                    qmeta[q_id] = OrderedDict([
                        ("level",row0.level),
                        ("subject",row0.subject),
                        ("field",row0.field),
                        ("topic",row0.topic),
                        ("n_atts",[]),
                        ("n_pass",[]),
                        ("stretch_values", []),
                        ("answer_type",row0.answer_type)
                    ])
                        # if write_main:
                        #     fname = "./sep/{}.txt".format(q_id.replace("|","~"))
                        #     print("new file handler",fname)
                        #     fhandles[q_id] = open(fname,"w")
                # else:
                #     if write_main:
                #         print("reopening",fhandles[q_id].name)
                #         fhandles[q_id] = open(fhandles[q_id].name,"w+")

                qusers  = pd.unique(rows["user_id"])

                for u in qusers:
                    natts = rows[rows.user_id==u]
                    npasz = natts[natts.correct==True]
                    qmeta[q_id]["n_atts"].append(natts.shape[0])
                    qmeta[q_id]["n_pass"].append(npasz.shape[0])
                    if npasz.shape[0] > 0:
                        stretch = natts.shape[0] / npasz.shape[0]
                        qmeta[q_id]["stretch_values"].append(stretch)
                    users.add(u) # each user added once bc Set

    if write_atypes:
        atypefile = open("new_atypes.csv","w")
        for k in qmeta.keys():
            d = qmeta[k]
            n_atts = d["n_atts"]
            n_pass = d["n_pass"]
            stretches = d["stretch_values"]

            print("Str=",stretches)

            del d["n_atts"]
            del d["n_pass"]
            del d["stretch_values"]

            median_atts = numpy.median(n_atts)
            median_pass = numpy.median(n_pass)
            median_stretch = numpy.median(stretches)

            mean_atts = numpy.mean(n_atts)
            mean_pass = numpy.mean(n_pass)
            mean_stretch = numpy.mean(stretches)

            pass_rates = numpy.divide(n_pass, n_atts)
            median_passrate = numpy.median(pass_rates)
            mean_passrate = numpy.mean(pass_rates)

            # ratio = d["n_atts"]/numpy.float64(d["n_pass"])
            atypefile.write(",".join([k.replace("|","~")]+[str(x) for x in d.values()]) +","+ str(median_atts) + "," + str(median_pass) +","+ str(median_passrate) + ","+ str(mean_passrate) +","+ str(median_stretch)+","+str(mean_stretch)+"\n")
        atypefile.close()

    if write_qmeta:
        metafile = open("qmeta.csv","w")
        for k in qmeta.keys():
            d = qmeta[k]
            ratio = d["n_atts"]/numpy.float64(d["n_pass"])
            metafile.write(",".join([k.replace("|","~")]+[str(x) for x in d.values()]) +","+ str(ratio) +"\n")
        metafile.close()

    if(write_users):
        ufile = open("users.csv","w")
        for u in users:
            ufile.write(u+"\n")
        ufile.close()
        print("done")