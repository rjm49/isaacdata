'''
Created on 2 May 2017

@author: Russell
'''
from FileLoader import FileLoader
from matplotlib import pyplot
import datetime
from irt.irt_engine import IRTEngine
from _collections import deque

if __name__ == '__main__':
    fl = FileLoader(".\database\isaac_MC_data")
    fl.load()
    all_students = fl._by_user    
#     buda_data = fl._by_user[42363] # get the all_students for user "BUDA"
    udata = fl._by_user[41431]
#     name = udata["name"]
    name = "average"
    
#     bhist = udata["history"]
    
    t0 = None
    bhist = []
    for student in all_students:
        shist = all_students[student]["history"]
        if len(shist)>99:
            for dtime in shist:
                if not t0 or dtime<t0:
                    t0 = dtime
                hitem = [dtime] + list(shist[dtime])
                bhist.append(hitem)
    
    bhist.sort()
    
    qstamp = []
    qdiff = []
    fstamp = []
    fdiff = []
    thdiff = []
    thstamp = []
    
    irt_eng = IRTEngine()
    
    #for hk in sorted(bhist.keys()):

    for helem in bhist:
        hk = helem[0]
        h = helem[1:]
#         h = bhist[hk]
#         t = (hk-t0) / 3.6e+3
        t = (hk - t0).total_seconds()
        print(t, h[1], h[2])
        if(h[2]):
            qdiff.append(h[1])
            qstamp.append(t)
        else:
            fdiff.append(h[1])
            fstamp.append(t)
    
        th = irt_eng.update(h[1], h[2])
        thdiff.append(th)
        thstamp.append(t)
        
pyplot.title(name+"'s question history")
# pyplot.plot(qstamp, qdiff, label="passes", marker="^", style="..", color="blue", zorder=-1)
# pyplot.scatter(fstamp, fdiff, label="fails", marker="x", color="red")
pyplot.plot(thstamp, thdiff, label="IRT", color="magenta", zorder=2)

pyplot.xlabel("Time attempted (h)")
pyplot.ylabel("Q difficulty")
pyplot.show()