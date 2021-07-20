import numpy as np
import scipy
import sys
import math
import seaborn as sns
import matplotlib.pyplot as plt

def merging(merging_list, clusters, init_tau, features, cluster_round):
    for i in merging_list: # Merge by merging list
        cluster0 = tuple(clusters[i[0]])  # points inside cluster0
        cluster1 = tuple(clusters[i[1]])
        features0 = [features[k] for k in cluster0] #extract features of cluster0
        features1 = [features[k] for k in cluster1] #extract features of cluster1
        
        #########################################################################

        centroid0 = np.mean(features0, axis=0) # Get controid of cluster0
        centroid1 = np.mean(features1, axis=0) # Get controid of cluster1
        
        gap = scipy.spatial.distance.cosine(centroid0, centroid1)

        if cluster_round == 0:
            if gap <= init_tau: #cluster0 and cluster 1 will be merged if their gap smaller than tau
                clusters[i[0]].extend(clusters[i[1]])
                clusters[i[1]] = []
        else:
            if gap <= init_tau: #cluster0 and cluster 1 will be merged if their gap smaller than tau
                clusters[i[0]].extend(clusters[i[1]])
                clusters[i[1]] = []

    return clusters


