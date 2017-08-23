'''
Created on 2 May 2017

@author: Russell
'''
from FileLoader import FileLoader
from matplotlib import pyplot
import datetime
from irt.irt_engine import IRTEngine
from _collections import deque, defaultdict
from scipy.optimize.minpack import curve_fit
import numpy
from sklearn.cluster.k_means_ import KMeans
from sklearn.preprocessing.data import StandardScaler
from sklearn.decomposition.pca import PCA


if __name__ == '__main__':
    fl = FileLoader(".\database\isaac_MC_data")
    fl.load()
    all_students = fl._by_user    
#     buda_data = fl._by_user[42363] # get the all_students for user "BUDA"
#     name = udata["name"]
    name = "average"
    
#     bhist = udata["history"]
    
    t0 = None
    m,f,h = 0,0,0

    svecs = []
    codes = []
    for student in all_students:
#     for student in [42363]:
        hist_dicts = all_students[student]["history_dict"]
        if len(hist_dicts)>100:
            print("vec'g",student, "w hist len ", len(hist_dicts))
            code = all_students[student]["name"]
            
            gender = 0 #some students have no specified gender
            if all_students[student]["gender"]=="MALE":
                gender=-1
                m+=1
                code = code+":M"
            elif all_students[student]["gender"]=="FEMALE":
                gender=1
                f+=1
                code = code+":F"
            else:
                h+=1
                code = code+":-"
            #create a vector for this student
            qnms = set()
            attempt_counts = defaultdict(int)
            total_attempts = 0
            total_successes = 0
            hint_counts = defaultdict(int)
            total_hints = 0
            domain_counts = defaultdict(int)
            total_diff = 0
            max_diff = 0
            min_diff = 6
            t0 = datetime.datetime.now()
            tmax = datetime.datetime.min
            for tstamp in hist_dicts:
                print(tstamp)
                print(t0)
                t0 = min(t0, tstamp)
                tmax = max(tmax, tstamp)
                hd = hist_dicts[tstamp]
                qnm = hd["question_name"]
                qnms.add(qnm)
                act = hd["action"]
                if act == "ANSWER_QUESTION":
                    attempt_counts[qnm] += 1
                    domain_counts[hd["subject"]]+=1
#                     print(domain_counts)
                    total_attempts +=1
                    if(hd["correct"]==True):
                        total_successes+=1
                    
                    diff = hd["difficulty"]
                    max_diff = max(max_diff, diff)
                    min_diff = min(min_diff, diff)
                    total_diff += hd["difficulty"]
                elif act == "VIEW_HINT":
                    hint_counts[qnm] += 1
                    total_hints += 1

            num_qs = len(qnms)
            if num_qs < 100:
                print("slacker, dropping")
                continue
            codes.append(code.lower())
            avg_atts = total_attempts/num_qs
            avg_hints = total_hints/total_attempts
            avg_diff = total_diff/total_attempts
            avg_succ = total_successes/total_attempts
#             print(domain_counts.values())
#             input("prompt")
            time_spent_in_physics = domain_counts["physics"] / sum(domain_counts.values())
            time_spent_in_maths = domain_counts["maths"] / sum(domain_counts.values())
            time_spent_in_chem = domain_counts["chemistry"] / sum(domain_counts.values())
            
            time_on_system = (tmax-t0).total_seconds()
            
            svec=[ gender, avg_atts, avg_hints, avg_diff, avg_succ , max_diff, min_diff, time_on_system ] #, time_spent_in_physics, time_spent_in_maths, time_spent_in_chem]
#             ml=1 if gender<0 else 0
#             fm=1 if gender>0 else 0
#             nn=1 if gender==0 else 0
#             svec=[gender, avg_atts, avg_hints, avg_succ, avg_diff]
            print(svec)
#             input("prompt")
            svecs.append(svec)
        #END per-student loop

    svecs = numpy.array(svecs)

#     gmarks = []
#     for sv in svecs:
#         if sv[0]==-1:
#             gmarks.append("D")
#         elif sv[0]==1:
#             gmarks.append("O")
#         else:
#             gmarks.append(".")

    scaler = StandardScaler()
    svecs = scaler.fit_transform(svecs)
    print(svecs)
    print(m,f,h)
    print("number of students",len(svecs))

    print("gender, avg_atts, avg_hints, avg_succ, avg_diff")
    kmeans = KMeans(n_clusters=3, n_jobs=-1).fit(svecs)
    labs = kmeans.labels_
    centres = kmeans.cluster_centers_
    print(centres)
    print(kmeans.inertia_)
    
    pca = PCA(n_components=2)
    tvecs = pca.fit_transform(svecs)
    
    x,y = zip(*tvecs)
    print(x)
    print(y)
    s=1.0
#     s = numpy.array(s)*5.0
#     print(s)
    cols = []
    for label in labs:
        if label==0:
            cols.append("blue")
        elif label==1:
            cols.append("brown")
        elif label==2:
            cols.append("green")
        elif label==3:
            cols.append("black")
        else:
            cols.append("magenta")
        
#     pyplot.scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, hold, data)
    fig, ax = pyplot.subplots()
    ax.scatter(x=x,y=y, s=50.0, c=cols, marker="o")
    for i, txt in enumerate(codes):
        ax.annotate(txt, (x[i],y[i]))
    
    print(tvecs)
#     print(pca.components_)
    print("params")
    print(pca.get_params(True))
    print(pca.explained_variance_ratio_) 

    print("components")
    print("gender, avg_atts, avg_hints, avg_diff, avg_succ , max_diff, min_diff ] #, time_spent_in_physics, time_spent_in_maths, time_spent_in_chem")
    print(pca.components_)

    print("means")
    print(pca.mean_)

    #  Two subplots, the axes array is 1-d
    f, axarr = pyplot.subplots(2)


    
    
    a = numpy.random.random((16, 16))
    print("a")
    print(a)

    axarr[0].imshow(a, cmap='hot', interpolation='nearest')
    b = numpy.random.random((2,20))
    axarr[1].scatter(*zip(b))

    pyplot.show()