import sklearn as sk

'''
Created on 24 Aug 2017

@author: Russell
'''
import os
import sys
from random import shuffle
from sklearn.metrics.cluster.unsupervised import calinski_harabaz_score,\
    silhouette_score, silhouette_samples
import datetime
import dateutil
import types
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

from sklearn import cluster, mixture
from sklearn.preprocessing.data import StandardScaler
import pandas
import numpy
import time
import warnings
from matplotlib import pyplot as plt
from itertools import cycle, islice
from sklearn.decomposition.pca import PCA
from numpy.random import choice
from numpy import random, unique
from math import isnan
from collections import Counter, OrderedDict
from utils.utils import extract_runs_w_timestamp

decay = 0.999
base = dateutil.parser.parse("2017-01-01 23:59:59")
date_list = [base + datetime.timedelta(days=x) for x in [i*30 for i in range(0,5)]]

def remove_zero_rc(arr, indx):
    zrows = numpy.nonzero(arr.sum(axis=1)==0)
    zcols = numpy.nonzero(arr.sum(axis=0)==0)
    arr = numpy.delete(arr, zrows, axis=0)
    arr = numpy.delete(arr, zcols, axis=1)
    return arr

create_xm = True
plot = True
if __name__ == '__main__':

    #build user experience matrix here....
    qmeta = pandas.read_csv("../qmeta.csv", header=None)
    users = open("../users.csv").read().splitlines()
    shuffle(users, lambda: 0.666)
    users = users[0:1000]
    print(users)
#     users = users[0:1000]
    
    levels = set()
    lev_lookup = {}
    
    SUB=3
    FLD=4
    TOP=5

    cats = set()
    cat_lookup = {}    
    cat_ixs = {}
    combo_ix=set()
    
    all_qids = set()
    for line in qmeta.itertuples():
        qid = line._1
        all_qids.add(qid)
        
        cat = str(line[SUB])+"/"+str(line[FLD])+"/"+str(line[TOP])
        cats.add(cat)
        cat_lookup[qid]= cat
        
        lev= -1 if isnan(line._2) else int(line._2)
        levels.add(lev)
        lev_lookup[qid]=lev
        
#         c_L = str(lev) # + "_" + str(lev)
        combo_ix.add(cat)
        
    for it in lev_lookup.items():   
        print(it[0],"\t\t\t",it[1])

    q_ixs={}
    #all_qids = unique(all_qids)
    for ix, q in enumerate(all_qids):
        q_ixs[q]=ix

    for ix, c in enumerate(cats):
        cat_ixs[c]=ix
        
    if create_xm:        
        #exp_mx = pandas.DataFrame( index=all_qids, columns=exp_cols )#, columns = questions)
        #cnt_mx = numpy.zeros(shape = (len(users), len(all_qids)), dtype=numpy.int32)
        
        #cnt_mx = pandas.DataFrame( index=users, columns=list(levels) )#, columns = questions)
        
        #exp_mx[:] = -1.0
        #exp_mx = numpy.zeros((len(users),len(questions)))
        #exp_mx.fillna(0.0, inplace=True)
        #cnt_mx.fillna(0.0, inplace=True)
        active_user_lookup = OrderedDict()
        exp_mx_lookup = OrderedDict()
        for target in date_list:
            active_users = []
            active_user_lookup[target] = active_users
            for uix,u in enumerate(users):
                uqatts = pandas.read_csv("../by_user/{}.txt".format(u), header=None)
                start = dateutil.parser.parse(uqatts.iloc[0,0])
                end = dateutil.parser.parse(uqatts.iloc[-1,0])
                if start < target <= end:
                    active_users.append(u)
            exp_mx_lookup[target] = numpy.zeros(shape = (len(active_users), len(all_qids))) 
            print("Created exp_mx of shape:", exp_mx_lookup[target].shape)
        
            for uix,u in enumerate(active_users):
                X = numpy.zeros(shape=(1,len(all_qids)))#, dtype=numpy.int32)
                uqatts = pandas.read_csv("../by_user/{}.txt".format(u), header=None)
                uqatts[0] = pandas.to_datetime(uqatts[0])
                uqatts = uqatts[uqatts[0]<=target]
#                 print("size for {}".format(target), uqatts.size)
                for run in extract_runs_w_timestamp(uqatts):
                    ts, q, n_atts, n_pass = run
#                     q = run[0]
#                     n_atts = run[1]
#                     n_pass = run[2]
                    q = q.replace("|","~")
    #                 L = lev_lookup[q]
                    c = cat_lookup[q]
                    #decay the user's previous career
                    exp_mx = exp_mx_lookup[target]
                    #exp_mx[uix] = decay * exp_mx[uix]
                    if(n_atts > 0):
                        qix = q_ixs[q]
                        exp_mx[uix, qix]=1.0
#                     else:
#                         qix = q_ixs[q]
#                         exp_mx[uix, qix]=-1.0
                #print(u,"done")
            print(target," ", len(active_users),"active:", active_users)        
        
#         exp_mx, all_qids = remove_zero_rc(exp_mx, all_qids)
        
        #exp_mx.to_csv("exp_mx_all.csv")
            numpy.savetxt("../by_date/mx{}.csv".format(target.strftime("%Y%m%d")), exp_mx, delimiter=",")#, fmt="%i")
            print("saved mx")
        
    colours = numpy.array([ '#880000','#008800','#000088','#888800','#880088','#008888' ])
    if plot:
        default_base = {'quantile': .3,
            'eps': .3,
            'damping': .9,
            'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 10}
        params = default_base.copy()
        
        print("creating clustering algos")
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
         
        kmeans = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
        
        i_dataset=0
        plot_num = 1
        
        fitted = False
        pca = PCA()
        for ix,target in enumerate(date_list):
            exp_mx = exp_mx_lookup[target]
            print("Loaded dataframe for {} = \n".format(target), exp_mx)
            
            X, y = (exp_mx, {})
#             zrows = numpy.nonzero(X.sum(axis=1)==0)
#             print("number of zero-rows:",len(zrows))
#             X = StandardScaler().fit_transform(X)
        
#         clustering_algorithms = [("spectral",spectral), ("kmeans", kmeans), ("AffProp", affinity_propagation)]
            name, algorithm = ("AffProp", affinity_propagation)
#         clustering_algorithms = [("kmeans", kmeans)]
        
                # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                
            algorithm.fit(X)
    
#             if hasattr(algorithm, 'labels_'):
#                 y_pred = algorithm.labels_.astype(numpy.int)
#             else:
            y_pred = algorithm.predict(X)
                    
            n_clusters=max(y_pred)+1
            print("n_clusters=",n_clusters)
            
            plt.subplot(1,5, ix+1)
            if ix == 0:
                plt.title(name, size=18)
        
            r = lambda: random.randint(255)

                #colours = numpy.array([ '#%02X%02X%02X'%(r(),r(),r()) for i in range(n_clusters) ])
            extra_cols_needed = n_clusters - colours.shape[0]
            print(extra_cols_needed, "extra cols needed")
            if extra_cols_needed > 0:
                print("adding colours")
                for _ in range(extra_cols_needed):
                    colours = numpy.append(colours,'#%02X%02X%02X'%(r(),r(),r()))
                    print("added col")
            print("colours=", colours)
            print(y_pred)
            c = Counter(y_pred)
            print(c.most_common())
            y_ranked = [ clustr for (clustr , cnt) in c.most_common() ]

            sil_score = 0
            ch_score = 0
            if n_clusters>1:
                ss = silhouette_samples(X,y_pred, metric="euclidean")
                ss = [s for s in ss if not isnan(s)]
                sil_score = numpy.mean(ss)
                ch_score = calinski_harabaz_score(X, y_pred)
                print("Sil=", sil_score)
                print("CHS=",ch_score)
            
            if not fitted:
                pca.fit(X)
                fitted = True
            to_plot = pca.transform(X)

            
            print(X.shape)
            print(y_pred.shape)
#             w_clusters = numpy.column_stack((X,y_pred))
#             numpy.savetxt("w_clusters.csv", w_clusters, delimiter=",")
            
            
            rankings = [y_ranked.index(y) for y in y_pred]
            plt.scatter(to_plot[:,0], to_plot[:,1], alpha=0.5, c=colours[rankings])
            
#                 plt.xlim(-2.5, 2.5)
#                 plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
#             plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
#                      transform=plt.gca().transAxes, size=15,
#                      horizontalalignment='right')
            plt.text(.99,.1 , "{}/{}".format(X.shape[0], n_clusters), transform=plt.gca().transAxes, size=8, horizontalalignment='right')
            plt.text(.99,.01 , "s{:.2f} ch{:.2f}".format(sil_score, ch_score), transform=plt.gca().transAxes, size=8, horizontalalignment='right')
            plt.text(.01,.99 , "{}".format(target.strftime("%Y-%m-%d")), transform=plt.gca().transAxes, size=8, horizontalalignment='left')
            
            
            plot_num += 1
    
            cl_ixs = algorithm.cluster_centers_indices_
            cl_sizes = [ c[ix] for ix in cl_ixs ]
            clusters = algorithm.labels_[cl_ixs]
            print(n_clusters,"clusters, labels are:",clusters)

            cl_uids = numpy.array(active_user_lookup[target])[cl_ixs]
            print(cl_uids)
#             
#             exemplars = pandas.DataFrame(data=algorithm.cluster_centers_, columns=all_qids, index=cl_uids)
#             
#             exemplars.insert(0, "cl_size", cl_sizes)
#             exemplars.insert(0, "cluster", clusters)
#             exemplars = exemplars.sort_values(by="cl_size", ascending=False)
#             
#             exemplars.to_csv("exemplars.csv")
    plt.show()

