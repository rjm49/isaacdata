'''
Created on 2 May 2017

@author: Russell
'''
from FileLoader import FileLoader
from matplotlib import pyplot
import datetime
from irt.irt_engine import IRTEngine
from _collections import deque
from scipy.optimize.minpack import curve_fit
import numpy

if __name__ == '__main__':
    fl = FileLoader(".\database\isaac_MC_data")
    fl.load()
    all_students = fl._by_user    
#     buda_data = fl._by_user[42363] # get the all_students for user "BUDA"
#     name = udata["name"]
    name = "average"
    
#     bhist = udata["history"]
    
    t0 = None
    bhist = []
    for student in all_students:
#     for student in [42363]:
        shist = all_students[student]["history"]
        if len(shist)>99:
            for dtime in shist:
#                 if not shist[dtime][0].startswith("masterclass") and shist[dtime][3]=='choice' and shist[dtime][5]=='mechanics':
                if not shist[dtime][0].startswith("masterclass") and shist[dtime][7]=="ANSWER_QUESTION" and shist[dtime][4]=="physics":
                    if not t0 or dtime<t0:
                        t0 = dtime
                    hitem = [dtime] + list(shist[dtime])
                    bhist.append(hitem)
    
    bhist.sort()
    print(len(bhist))
#     input("prompt")
    
    qstamp = []
    qdiff = []
    fstamp = []
    fdiff = []
    thdiff = []
    rdiff=[]
    thstamp = []
    
    irt_eng = IRTEngine()
    
    #for hk in sorted(bhist.keys()):
    roll_avg_list = deque([], maxlen=30)
    for helem in bhist:
        hk = helem[0]
        h = helem[1:]
#         h = bhist[hk]
#         t = (hk-t0) / 3.6e+3
#         t = (hk - t0).total_seconds() / 86400.0
        t = hk
        print(t, h[1], h[2])
        if(h[2]):
            roll_avg_list.append(h[1])
            rav = sum(roll_avg_list)/len(roll_avg_list)
            print("rav", t, roll_avg_list, len(roll_avg_list), rav)
#             input("prompt")
            res = h[1]
            qdiff.append(res)
            rdiff.append(rav)
            qstamp.append(t)
        else:
            fdiff.append(h[1])
            fstamp.append(t)

pyplot.title("42363 Pass history")
# pyplot.title("Physics Students Pass history")
pyplot.plot(qstamp, qdiff, label="passes", color="#ccccee", marker=".", zorder=1)
pyplot.scatter(fstamp, fdiff, label="fails", marker="x", color="red", zorder=-1)
pyplot.plot(qstamp, rdiff, label="30d RAV", color="magenta", zorder=2)

# x_new = numpy.linspace(min(qstamp), max(qstamp), 10)
powerlaw = lambda x, amp, index, inter : inter + amp * (x**index)
# popt,pcov = curve_fit(powerlaw, qstamp, qdiff)

# pyplot.plot(x_new, powerlaw(x_new, *popt), label="power law fit", color="green", zorder=-1)

pyplot.xlabel("Time attempted (days)")
pyplot.ylabel("Q difficulty")
pyplot.show()