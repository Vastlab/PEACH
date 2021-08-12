import numpy as np
import os, sys
import scipy
import random
import time
import collections
import threading
import torch
import torch.nn as nn
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from pyflann import *
from tau_flann_pytorch import tolerance
from merge import merging, merging_combine, merging_combine_sum
from evaluate import convert_clusters_to_label
from sklearn.neighbors import NearestNeighbors
from clustering import KNN

def cosine(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum('nc,ck->nk', [x, y.T])
    distances = 1 - similarity
    return distances


def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
    return distances

def FACTO(features, gpu, metric="SUM", method = "SUM", no_singleton=False):
    torch.cuda.set_device(gpu)
    if features.shape[1] > 128:
        try:
            pca = PCA(n_components=128, whiten=False)
            pca.fit(features)
            features = pca.transform(features)
        except:
            pass
    points = features
    print("FACTO Start")
    length = len(points)
    # Get threshold
    if metric == "cosine+euclidean":
        estimated_gap_cos, estimated_gap_eu, nearest_points_cos, nearest_points_eu, init_length_cos, init_length_eu, nearest_cluster_with_distance_round_1_cos, nearest_cluster_with_distance_round_1_eu, nearest_points_dis_cos, nearest_points_dis_eu = tolerance(features, gpu, metric)
    elif metric == "SUM":
        estimated_gap, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis, max_eu = tolerance(features, gpu, metric)
    else:
        estimated_gap, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis = tolerance(features, gpu, metric)
    ################################################################################################
    clusters = [[i] for i in range(length)]

    ################################################################################################
    # Generate the average distance pairwise between each two clusters by calculating the nearest distance of each point from cluster A to the closest point from cluster B
    round = 0
    #############################################
    ###############Start Clustering##############
    #############################################
    while True:
        if round == 0:
            # In round 1 the centroids is the points no matter what's linkage
            # Merging by nearest points
            if metric == "cosine+euclidean":
                nearest_cluster_with_distance_cos = nearest_cluster_with_distance_round_1_cos
                nearest_cluster_with_distance_eu = nearest_cluster_with_distance_round_1_eu
                nearest_cluster_cos = []
                nearest_cluster_eu = []
                for m in sorted(nearest_cluster_with_distance_cos, key=takeSecond):
                    nearest_cluster_cos.append(m[1][1])
                for m in sorted(nearest_cluster_with_distance_eu, key=takeSecond):
                    nearest_cluster_eu.append(m[1][1])
                nearest_cluster = []
                ###############################AND################################
                processed = set()
                for a, b in zip(nearest_cluster_cos, nearest_cluster_eu):
                    if method == "OR":
                        if a not in processed:
                            nearest_cluster.append(a)
                            processed.add(a)
                        elif b not in processed:
                            nearest_cluster.append(b)
                            processed.add(b)
                    elif method == "AND":
                        if a == b:
                            nearest_cluster.append(a)
                ##################################################################
                merging_list = set()
                for m, n in enumerate(nearest_cluster):
                    merging_list.add(tuple((m, n)))

            else:
                nearest_cluster_with_distance = nearest_cluster_with_distance_round_1
                nearest_cluster = []
                nearest_cluster_dis = []
                for m in sorted(nearest_cluster_with_distance, key=takeSecond):
                    nearest_cluster_dis.append(m[0])
                    nearest_cluster.append(m[1][1])
                #############################################
                ###############Generate merging list#########
                #############################################
                nearest_cluster_with_distance = sorted(
                    nearest_cluster_with_distance)  # Sort by distance, process the smallest one first
                merging_list = set()
                merging_list_with_cluster_id = set()
                processed = set()
                # Generate merging list (cluster pairs to be merged)(sorted by shortest distance)
                for i, j in enumerate(nearest_cluster_with_distance):
                    if j[1][0] not in processed and j[1][1] not in processed:
                        merging_list.add(tuple(j[1]))
                        merging_list_with_cluster_id.add((tuple(clusters[j[1][0]]), tuple(clusters[j[1][1]])))
                        processed.add(j[1][0])
                        processed.add(j[1][1])

        else:
            centroids = [np.mean([points[j] for j in i], axis=0) for i in clusters]
            X = np.array(centroids)
            ###############################################################################################

            """
            tra = Normalizer(norm='l2').fit(X)
            X = tra.transform(X)
            flann = FLANN()
            result, result_dis = flann.nn(X, X, num_neighbors=2, algorithm="kdtree", trees=32, checks=512)
            nearest_cluster = np.array([cls[1] for cls in result])
            nearest_cluster_dis = np.array([dis[1] for dis in result_dis])"""

            X = torch.Tensor(X)
            if metric == "cosine":
                dist = cosine(X, X)
            elif metric == "euclidean":
                dist = euclidean(X, X)
            elif metric == "SUM":
                eu_dis = euclidean(X, X)
                #dist = cosine(X, X) + eu_dis / torch.max(eu_dis)
                dist = cosine(X, X) + eu_dis / max_eu
            knn = dist.topk(2, largest=False)
            result = knn.indices.cpu().numpy()
            nearest_cluster = np.array([cls[1] for cls in result])
            nearest_cluster_dis = [dist[i][j] for i, j in enumerate(nearest_cluster)]


            nearest_cluster_with_distance = [[j, [k, i]] for k, (i, j) in
                                             enumerate(zip(nearest_cluster, nearest_cluster_dis))]

            #############################################
            ###############Generate merging list#########
            #############################################
            nearest_cluster_with_distance = sorted(
                nearest_cluster_with_distance)  # Sort by distance, process the smallest one first
            merging_list = set()
            merging_list_with_cluster_id = set()
            processed = set()
            # Generate merging list (cluster pairs to be merged)(sorted by shortest distance)
            for i, j in enumerate(nearest_cluster_with_distance):
                if j[1][0] not in processed and j[1][1] not in processed:
                    merging_list.add(tuple(j[1]))
                    merging_list_with_cluster_id.add((tuple(clusters[j[1][0]]), tuple(clusters[j[1][1]])))
                    processed.add(j[1][0])
                    processed.add(j[1][1])

            # Select by shorest distance. Each cluster only merge once per round.
            # example: sorted nearest_cluster_label [[0, 5], [5, 0], [2, 3], [3, 2], [4, 5], [1, 3]] then we got merging_list [[0, 5], [2, 3]]
            # Now let's decide to merge cluster0 and cluster1 from merging list

        #############################################
        ###############Start merging#################
        #############################################
        old_clusters = set()
        for i in clusters:
            old_clusters.add(tuple(i))  # find old (last round) clusters
        #print("Merging")
        if metric == "cosine+euclidean":
            clusters = merging_combine(merging_list, clusters, estimated_gap_cos, estimated_gap_eu, features, round, metric, method)
        elif metric == "SUM":
            clusters = merging_combine_sum(merging_list, clusters, estimated_gap, features, max_eu)
        else:
            clusters = merging(merging_list, clusters, estimated_gap, features, round, metric)
        # rembember old and new clusters
        result_clusters = [k for k in clusters if len(k) != 0]
        #########################################################################################
        if len(clusters) == len(result_clusters):  # Break if there is no new clusters found (nothing to merge).
            break
        clusters = result_clusters
        round += 1
    # clusters = result_clusters

    if no_singleton == True:
        true_clusters = [i for i in result_clusters if len(i) != 1]
        singletons = [k for k in clusters if len(k) == 1]
        centroids = [np.mean(cluster) for cluster in true_clusters]
        for single in singletons:
            dis = [scipy.spatial.distance.cosine(single, centroid) for centroid_id, centroid in enumerate(centroids)]
            index = np.argsort(dis)[0]
            true_clusters[index].extend(single)
        clusters = true_clusters


    labels = np.array(convert_clusters_to_label(clusters, length))
    print("FACTO Done")
    true_clusters = [i for i in clusters if len(i) != 1]
    single = len(clusters) - len(true_clusters)
    print("True Clusters: ", len(true_clusters), "Singletons: ", single)
    return labels


def takeFirst(elem):
    return elem[0]


def takeSecond(elem):
    return elem[1]


def takeThird(elem):
    return elem[2]


def takeFourth(elem):
    return elem[3]


def thread(threads):
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()



