import sklearn as sk

from backfit.BackfitUtils import init_objects

'''
Created on 24 Aug 2017

@author: Russell
'''
import os
import sys
from random import shuffle
from sklearn.metrics.cluster.unsupervised import calinski_harabaz_score,\
    silhouette_score, silhouette_samples
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
from collections import Counter
from utils.utils import extract_runs_w_timestamp

decay = 0.999

def remove_zero_rc(arr, indx):
    zrows = numpy.nonzero(arr.sum(axis=1)==0)
    zcols = numpy.nonzero(arr.sum(axis=0)==0)
    arr = numpy.delete(arr, zrows, axis=0)
    arr = numpy.delete(arr, zcols, axis=1)
    return arr

create_xm = False
plot = True
if __name__ == '__main__':

    base = "../../../isaac_data_files/"

    #build user experience matrix here....
    qmeta = pandas.read_csv(base+"qmeta.csv", header=None)
    n_users=1000
    cats, cat_lookup, all_qids, users, diffs, levels, cat_ixs = init_objects(n_users, path=base)

    levels = set()
    lev_lookup = {}
    
    if create_xm:
        exp_mx = numpy.zeros(shape = (len(users), len(all_qids)))
        print("Created exp_mx of shape:", exp_mx.shape)
        
        for uix,u in enumerate(users):
            X = numpy.zeros(shape=(1,len(all_qids)))
            uqatts = pandas.read_csv(base+"by_user/{}.txt".format(u), header=None)
            runs = extract_runs_w_timestamp(uqatts)
            
            for run in runs:
                ts,q,n_atts,n_pass = run
                q = q.replace("|","~")
                c = cat_lookup[q]
                exp_mx[uix] = decay * exp_mx[uix]
                if(n_pass > 0):
                    qix = all_qids.index(q)
                    exp_mx[uix, qix]=1.0
            print(u,"done")

        numpy.savetxt(base+"exp_mx_all.csv", exp_mx, delimiter=",")
        print("saved mx")
        
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
        
        print("loading data frame")
        i_dataset=0
        plot_num = 1
        if not create_xm:
            #exp_mx = pandas.read_csv("exp_mx_all.csv", header=None, index_col=None)
            exp_mx = numpy.genfromtxt("exp_mx_all.csv", delimiter=",")

#         exp_mx = exp_mx.drop(labels=["cluster","*_N","*_P"],axis=1)
        #exp_mx = exp_mx.loc[exp_mx.sum(axis=1) > 0]
        #exp_mx = remove_zero_rc(exp_mx)
        
        print("exp shape", exp_mx.shape)
        
        datasets = [exp_mx]
        
        X, y = (exp_mx, {})
        zrows = numpy.nonzero(X.sum(axis=1)==0)
        print("number of zero-rows:",len(zrows))

        numpy.random.seed(666)
        ixs = numpy.random.choice(users, 1000, replace=False)
        print(ixs)
        
        test_users = []
        for ix in ixs:
            print(ix)
            test_users.append( users[int(ix)])
        print(ixs)
        X = X[ixs, :]
#         X = StandardScaler().fit_transform(X)
        
#         clustering_algorithms = [("spectral",spectral), ("kmeans", kmeans), ("AffProp", affinity_propagation)]
        clustering_algorithms = [("AffProp", affinity_propagation)]
#         clustering_algorithms = [("kmeans", kmeans)]
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
                c = Counter(y_pred)
                print(c.most_common())

#                 ixs_over_thresh = [k for (k,v) in c.items() if v > 1]
#                 print(ixs_over_thresh)
#                 sil_X = X[ixs_over_thresh, :]
#                 sil_y = y_pred[ixs_over_thresh]
                
#                 ss = silhouette_samples(sil_X,sil_y, metric='euclidean')
                ss = silhouette_samples(X,y_pred, metric="euclidean")
                ss = [s for s in ss if not isnan(s)]
                
                print("Sil=", numpy.mean(ss))
                print("CHS=",calinski_harabaz_score(X, y_pred))
                
                pca = PCA()
                to_plot = pca.fit_transform(X)
                
                print(X.shape)
                print(y_pred.shape)
                w_clusters = numpy.column_stack((X,y_pred))
                numpy.savetxt("w_clusters.csv", w_clusters, delimiter=",")
                
                
                
                plt.scatter(to_plot[:,0], to_plot[:,1], alpha=0.5, c=colours[y_pred])
                
#                 plt.xlim(-2.5, 2.5)
#                 plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plot_num += 1
        
                cl_ixs = algorithm.cluster_centers_indices_
                cl_sizes = [ c[ix] for ix in cl_ixs ]
                labels = algorithm.labels_
                clusters = labels[cl_ixs]

                cl_uids = numpy.array(test_users)[cl_ixs]
                print(cl_uids)
                
                exemplars = pandas.DataFrame(data=algorithm.cluster_centers_, columns=all_qids, index=cl_uids)
                
                exemplars.insert(0, "cl_size", cl_sizes)
                exemplars.insert(0, "cluster", clusters)
                exemplars = exemplars.sort_values(by="cl_size", ascending=False)
                
                exemplars.to_csv("exemplars.csv")
        plt.show()

