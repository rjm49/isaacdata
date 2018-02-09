import sklearn as sk

'''
Created on 24 Aug 2017

@author: Russell
'''
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
from collections import Counter
from utils.utils import extract_runs_w_timestamp

base = "../../../isaac_data_files/"
create_xm = True
plot = True
if __name__ == '__main__':

    #build user experience matrix here....
    qmeta = pandas.read_csv(base+"qmeta.csv", header=None)
    users = open(base+"users.csv").read().splitlines()[0:1000]
    #users = choice(users, size=3000, replace=False)
    print(users)
#     users = users[0:1000]
    
    levels = set()
    lev_lookup = {}
    
    SUB=3
    FLD=4
    TOP=5

    cats = []
    cat_lookup = {}    
    combo_ix=set()
    
    all_qids = []
    for line in qmeta.itertuples():
        qid = line._1
        all_qids.append(qid)
        
        cat = str(line[SUB])+"/"+str(line[FLD])+"/"+str(line[TOP])
        cats.append(cat)
        cat_lookup[qid]= cat
        
        lev= -1 if isnan(line._2) else int(line._2)
        levels.add(lev)
        lev_lookup[qid]=lev
        
#         c_L = str(lev) # + "_" + str(lev)
        combo_ix.add(cat)
        
    for it in lev_lookup.items():   
        print(it[0],"\t\t\t",it[1])
        
    if create_xm:
        exp_cols = [x+"_X" for x in combo_ix] # + [x+"_N" for x in combo_ix] + [x+"_P" for x in combo_ix]
        #exp_cols = [x+"_P" for x in all_qids] 
        #exp_cols = [x+"_X" for x in all_qids] # + [x+"_N" for x in all_qids] + [x+"_P" for x in all_qids] + ["total_qs"]
        
        exp_mx = pandas.DataFrame( index=users, columns=exp_cols )#, columns = questions)
        print("Created exp_mx of shape:", exp_mx.shape)
        
        #cnt_mx = pandas.DataFrame( index=users, columns=list(levels) )#, columns = questions)
        
        #exp_mx[:] = -1.0
        #exp_mx = numpy.zeros((len(users),len(questions)))
        exp_mx.fillna(0.0, inplace=True)
        #cnt_mx.fillna(0.0, inplace=True)
        for i,u in enumerate(users):
#             print(u,"...")
            uqatts = pandas.read_csv(base+"by_user/{}.txt".format(u), header=None)
            runs = extract_runs_w_timestamp(uqatts)
            
            for run in runs:
                ts,q,n_atts,n_pass= run
                q = q.replace("|","~")
#                 L = lev_lookup[q]
                c = cat_lookup[q]
                exp_mx.loc[u,c+"_X"] += 1.0
#                 exp_mx.loc[u,c+"_N"] += n_atts
#                 exp_mx.loc[u,c+"_P"] += n_pass#/n_atts
#                 exp_mx.loc[u,"total_qs"] += 1.0
#                   if(p>0):
#                     exp_mx.loc[u,c] += 1.0
            print(u,"done")
        exp_mx.fillna(0.0, inplace=True)
        #print(exp_mx.shape)
        exp_mx = exp_mx.loc[exp_mx.sum(axis=1) > 0]
        print("new shape=", exp_mx.shape)
        exp_mx.to_csv("exp_mx_all.csv")
        print("saved mx")

    if plot:
        default_base = {'quantile': .3,
            'eps': .3,
            'damping': .9,
            'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 5}
        params = default_base.copy()
        
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
        if not create_xm:
            exp_mx = pandas.read_csv("exp_mx_all.csv", header=0, index_col=0)

#         w = len(exp_mx.columns)
        exp_mx = exp_mx.iloc[:,:]
#         exp_mx = exp_mx>0
#         exp_mx = exp_mx.drop(labels=["cluster","*_N","*_P"],axis=1)
        exp_mx = exp_mx.loc[exp_mx.sum(axis=1) > 0]
        print("exp shape", exp_mx.shape)
        
        datasets = [exp_mx]
        
        X, y = (exp_mx, {})
        X = StandardScaler().fit_transform(X)
#         clustering_algorithms = [("spectral",spectral), ("kmeans", kmeans), ("AffProp", affinity_propagation)]
        clustering_algorithms = [("AffProp", affinity_propagation)]
        for name, algorithm in clustering_algorithms:
                t0 = time.time()
        
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
        
                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(numpy.int)
                else:
                    y_pred = algorithm.predict(X)
                    print("pred y", y_pred)
                    
                n_clusters=0
                if hasattr(algorithm, 'cluster_centers_indices_'):
                    n_clusters=len(algorithm.cluster_centers_indices_)
                    print("no clusters:",len(algorithm.cluster_centers_indices_))
                else:
                    n_clusters=params['n_clusters']
                
                plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)
        
#                 colours = numpy.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                                      '#f781bf', '#a65628', '#984ea3',
#                                                      '#999999', '#e41a1c', '#dede00']),
#                                               int(max(y_pred) + 1))))
                
                #import random
                r = lambda: random.randint(255)
                colours = numpy.array([ '#%02X%02X%02X'%(r(),r(),r()) for i in range(max(y_pred)+1) ])
                print(colours)
                
                pca = PCA()
                to_plot = pca.fit_transform(X)

                #print(y_pred)
                c = Counter(y_pred)
                print(c.most_common())
                exp_mx.loc[:,"cluster"] = y_pred
                exp_mx.to_csv("exp_w_clusters.csv")
                
                
                plt.scatter(to_plot[:,0], to_plot[:,1], alpha=0.5, c=colours[y_pred])
        
#                 plt.xlim(-2.5, 2.5)
#                 plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plot_num += 1
        
                exemplars = pandas.DataFrame(columns=exp_mx.columns)
                exemplars["cl_size"]=0
                for i,c in enumerate(unique(y_pred)):
                    cluster_entries = exp_mx[exp_mx["cluster"]==c]
                    no = len(cluster_entries)
                    exemplar = cluster_entries.mean()
                    
                    if(no>10):
                        print(exemplar)
                    
                    exemplar["cl_size"]=no
                    exemplar.name= str(c)
                    exemplars = exemplars.append( exemplar )
                    #print(exemplars)
                exemplars.to_csv("exemplars.csv")
        plt.show()

