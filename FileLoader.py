'''
Created on 2 Mar 2017

@author: Russell
'''
import codecs
import datetime

class FileLoader(object):
    '''
    classdocs
    '''


    def __init__(self, fname):
        '''
        Constructor
        '''
        self.filename = fname
        self._f = codecs.open(fname,"r")
        self._by_user = {}
        self.raw_rows = []
        self.column_names = []



    def clean(self, clean="default_clean.dat"):
#         f5652 = open("f5652.dat", mode='w')
#         f5653 = open("f5653.dat", mode='w')
        cleaned = open(clean,"w")

        for ln in self._f:
            print(ln)
            a_list = list(eval(ln))
            a_list[4] = a_list[4].timestamp()
            a_list = [str(x) for x in a_list]
            cleaned.write(",".join(a_list)+"\n")
        cleaned.close()
                
    def load_raw_only(self):
        print("loading raw data")
        with self._f as f:
            self.column_names=f.readline().split(", ")
            for a in f:
                a_list = eval(a)
                self.raw_rows.append(a_list)
        
    def load(self):
        t0 = None
        self.load_raw_only()
        for a_list in self.raw_rows:
            udict = {}
#                 print(a_list)
            uid = int(a_list[0])
            grp = int(a_list[1])
            nam = str(a_list[3])
            sex = a_list[8]
            if uid not in self._by_user:
                udict = {}
                self._by_user[uid]=udict
                udict["name"]=nam
                udict["group"]=grp
                udict["gender"]=None
                if (sex=="MALE" or sex=="FEMALE"):
                    udict["gender"]=sex
                udict["history"]={}
                udict["history_dict"]={}
#                     print("added new user")
            if not t0:
                t0 = a_list[4]
#                 tsp = float(a_list[8])
#                 print(a_list[4])
            tsp = a_list[4]
            qnm = str(a_list[5])
            cor = True if a_list[6]=="true" else False
            act = a_list[7]
            dif = a_list[2]
            typ = a_list[9]
            sjt = a_list[10]
            fld = a_list[11]
            top = a_list[12]
            hist_rec = (qnm, dif, cor, typ, sjt, fld, top, act)
            hist_dict = {
                            "question_name":qnm,
                            "correct":cor,
                            "action":act,
                            "difficulty":dif,
                            "type":typ,
                            "subject":sjt,
                            "field":fld,
                            "topic":top
                        }
            self._by_user[uid]["history"][tsp]=hist_rec
            self._by_user[uid]["history_dict"][tsp]=hist_dict
#                 print("stored ", hist_rec, "under", tsp)

    
    def output_for_R(self, csv):
        R_rows = []
        for row in self.raw_rows:
            R_row = []
            for el in row:
                if isinstance(el, datetime.date):
                    el = el.isoformat()
                R_row.append(str(el))
            R_rows.append(R_row)

        fout = open(csv, "w")
        fout.write(",".join(self.column_names))
        for r in R_rows:
            fout.write(",".join(r)+"\n")
        fout.close()
    
if __name__=="__main__":
#     fl = FileLoader(".\isaac_clean2.dat")
    fl = FileLoader(".\database\isaac_MC_data")
    fl.load() # load into raw_rows
    fl.output_for_R("isaac_R_format.csv") # convert raw_rows into R table
    #fl.clean(clean="isaac_all_questions.dat")
#     fl.load()
    all_students = fl._by_user
    print(len(all_students.keys()))
#     input("prompt")
    
    for k in all_students.keys():
        user = all_students[k]
        print(k, user["name"], user["group"], len(user["history"]))
        if k==42363:
            for hk in sorted(user["history"].keys()):
                hitem = user["history"][hk]
                print(hk, hitem)
            print("- - - - - -")