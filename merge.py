import numpy as np
import scipy
import sys
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

def merging(merging_list, clusters, init_tau, features, cluster_round, metric):
    for i in merging_list: # Merge by merging list
        cluster0 = tuple(clusters[i[0]])  # points inside cluster0
        cluster1 = tuple(clusters[i[1]])
        features0 = [features[k] for k in cluster0] #extract features of cluster0
        features1 = [features[k] for k in cluster1] #extract features of cluster1
        
        #########################################################################

        centroid0 = np.mean(features0, axis=0) # Get controid of cluster0
        centroid1 = np.mean(features1, axis=0) # Get controid of cluster1

        if metric == "cosine":
            gap = scipy.spatial.distance.cosine(centroid0, centroid1)
        elif metric == "euclidean":
            gap = scipy.spatial.distance.euclidean(centroid0, centroid1)
        if cluster_round == 0:
            if gap <= init_tau: #cluster0 and cluster 1 will be merged if their gap smaller than tau
                clusters[i[0]].extend(clusters[i[1]])
                clusters[i[1]] = []
        else:
            if gap <= init_tau: #cluster0 and cluster 1 will be merged if their gap smaller than tau
                clusters[i[0]].extend(clusters[i[1]])
                clusters[i[1]] = []

    return clusters


def merging_combine(merging_list, clusters, tau, tau1, features, cluster_round, metric, method):
    for i in merging_list:  # Merge by merging list
        cluster0 = tuple(clusters[i[0]])  # points inside cluster0
        cluster1 = tuple(clusters[i[1]])
        if len(cluster0) != 0 and len(cluster1) != 0:
            features0 = [features[k] for k in cluster0]  # extract features of cluster0
            features1 = [features[k] for k in cluster1]  # extract features of cluster1
            #########################################################################
            centroid0 = np.mean(features0, axis=0)  # Get controid of cluster0
            centroid1 = np.mean(features1, axis=0)  # Get controid of cluster1
            gap_cos = scipy.spatial.distance.cosine(centroid0, centroid1)
            gap_eu = scipy.spatial.distance.euclidean(centroid0, centroid1)

            if metric == "SUM":
                #tra = Normalizer(norm='l2').fit(gap_eu)
                #gap_eu = tra.transform(gap_eu)
                #print(gap_eu)
                gap = gap_cos + gap_eu
                print(gap, tau)
                if gap <= tau:
                    clusters[i[0]].extend(clusters[i[1]])
                    clusters[i[1]] = []
            elif method == "OR":
                if gap_cos <= tau_cos or gap_eu <= tau_eu:
                    clusters[i[0]].extend(clusters[i[1]])
                    clusters[i[1]] = []
            elif method == "AND":
                if gap_cos <= tau_cos and gap_eu <= tau_eu:
                    clusters[i[0]].extend(clusters[i[1]])
                    clusters[i[1]] = []


    return clusters

def merging_combine_sum(merging_list, clusters, tau, features, max_eu):
    for i in merging_list:  # Merge by merging list
        cluster0 = tuple(clusters[i[0]])  # points inside cluster0
        cluster1 = tuple(clusters[i[1]])
        #if len(cluster0) != 0 and len(cluster1) != 0:
        features0 = [features[k] for k in cluster0]  # extract features of cluster0
        features1 = [features[k] for k in cluster1]  # extract features of cluster1
        #########################################################################
        centroid0 = np.mean(features0, axis=0)  # Get controid of cluster0
        centroid1 = np.mean(features1, axis=0)  # Get controid of cluster1
        gap_cos = scipy.spatial.distance.cosine(centroid0, centroid1)
        gap_eu = scipy.spatial.distance.euclidean(centroid0, centroid1)

        gap = gap_cos + gap_eu / max_eu
        if gap <= tau:
            clusters[i[0]].extend(clusters[i[1]])
            clusters[i[1]] = []

    return clusters
