import numpy as np
import threading
import math
import random
import torch
import torch.nn as nn
import weibull as weibull
import itertools
import scipy
import sklearn
import csv
from sklearn.preprocessing import Normalizer
#from FACTO import gpu_torch_distances
from collections import Counter

def cosine(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum('nc,ck->nk', [x, y.T])
    distances = 1-similarity
    return distances
    
def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
    return distances

def gpu_torch_distances(data, batch_size, metric):
    dist = []
    batch_size = batch_size
    mutilple_data = [data[i * batch_size:(i + 1) * batch_size] for i in
                         range((len(data) + batch_size - 1) // batch_size)]
    Y_global = torch.tensor(np.array(data)).cuda()
    for i, chunk1 in enumerate(mutilple_data):
        X_global = torch.tensor(np.array(chunk1)).cuda()
        if metric == "cosine":
            dis = cosine(X_global, Y_global)
        elif metric == "euclidean":
            dis = euclidean(X_global, Y_global)
        dist.append(dis.cpu())
    del X_global, Y_global
    dist = torch.cat(dist)
    return dist

################################################################
##################Tau-Simple###################################
################################################################
def tolerance(features, gpu, metric, batch_size, EVT):
    # Use up to 50000 simples to compute tau
    if len(features) >= 50000:
        features = np.array(random.choices(features, k = 50000))
############################################
    if metric == "cosine" or metric == "euclidean":
        dist = gpu_torch_distances(features, batch_size, metric)
        #tau, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis = compute_tau(distances, features, metric, "NA", 0, total_distances, max_dis)
        tau, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis = compute_tau(
            dist, features, metric, "NA", 0, batch_size, EVT)
        return tau, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis
    elif metric == "SUM":
        X = torch.Tensor(features)
        cosine_distances = cosine(X, X)
        euclidean_distances = euclidean(X, X)
        max_eu = torch.max(euclidean_distances)
        euclidean_distances = euclidean_distances / max_eu

        tau_cos, nearest_points_cos, init_length_cos, nearest_cluster_with_distance_round_1_cos, nearest_points_dis_cos = compute_tau(
            cosine_distances, features, "cosine", "SUM", 0, batch_size, EVT)
        tau_eu, nearest_points_eu, init_length_eu, nearest_cluster_with_distance_round_1_eu, nearest_points_dis_eu = compute_tau(
            euclidean_distances, features, "euclidean", "SUM", max_eu, batch_size, EVT)
        tau = tau_cos + tau_eu

        return tau, nearest_points_cos, init_length_cos, nearest_cluster_with_distance_round_1_cos, nearest_points_dis_cos, max_eu
        #return tau, nearest_points_eu, init_length_eu, nearest_cluster_with_distance_round_1_eu, nearest_points_dis_eu

def compute_tau(distances, features, metric, method, max_eu, batch_size, EVT):
    ################################################
    avg_all_distances = torch.median(distances).cpu().numpy()
    max_dis = torch.max(distances).cpu().numpy()
    knn = distances.topk(2, largest=False)
    result = knn.indices.cpu().numpy()
    nearest_cluster = np.array([cls[1] for cls in result])
    nearest_points_dis = [distances[i][j] for i, j in enumerate(nearest_cluster)]

    nearest_points = nearest_cluster
    nearest_cluster_with_distance_round_1 = [[j, [k, i]] for k, (i, j) in enumerate(zip(nearest_cluster, nearest_points_dis))]
    nearest_cluster_with_distance_round_1 = sorted(nearest_cluster_with_distance_round_1)

    ########################################################################################
    appear = dict(Counter(nearest_points))
    appear_count = [[j, i] for i, j in enumerate(appear)]
    # count the appearance of each kernel points
    # generate order
    order = [i[1] for i in sorted(appear_count, reverse=True)]
    # add non kernel points to order
    processed = set()
    init = []
    for count, i in enumerate(order):
        j = nearest_points[i]
        if i not in processed and j not in processed:
            init.append([i, j])
            processed.add(i)
            processed.add(j)
    init = init[0: int(len(init))]
    N = len(init)
    init_length = N
    init_features = [[features[i[0]], features[i[1]]] for i in init] #features of initial groups.
    ######################################################################################################
    centroids = [np.mean(i, axis=0) for i in init_features]
    dist = gpu_torch_distances(centroids, batch_size, metric)
    knn = dist.topk(2, largest=False)
    result = knn.indices.cpu().numpy()
    nearest_init = np.array([cls[1] for cls in result])

    ##########################################################################################################
    nearest_init_combo = [[m, init[n]] for m, n in zip(init, nearest_init)]
    ########################################################################################
    gxs = []
    for pair1, pair2 in nearest_init_combo:
        features0 = [features[k] for k in pair1] #extract features of cluster0
        features1 = [features[k] for k in pair2] #extract features of cluster1
        centroid0 = np.mean(features0, axis=0).reshape(-1) # Get controid of cluster0
        centroid1 = np.mean(features1, axis=0).reshape(-1) # Get controid of cluster1
        if metric == "cosine":
            gx = scipy.spatial.distance.cosine(centroid0, centroid1)
            gxs.append(gx)
        elif metric == "euclidean":
            gx = scipy.spatial.distance.euclidean(centroid0, centroid1)
            gxs.append(gx)
    if method == "SUM" and metric == "euclidean":
        gxs = np.array(gxs)
        gxs = gxs / max_eu.cpu().numpy()
        if EVT == True:
            tau = get_tau(torch.Tensor(nearest_points_dis),1,'PEACH',tailfrac=1,pcent=.999,usehigh=True,maxmodeerror=1)* avg_all_distances / max_dis
        else:
            tau = np.max(gxs) * avg_all_distances / max_dis
        print("Tau Detected: ", tau)
        return tau, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis
    else:
        if EVT == True:
            tau = get_tau(torch.Tensor(nearest_points_dis),1,'PEACH',tailfrac=1, pcent=.999,usehigh=True,maxmodeerror=1) * avg_all_distances / max_dis
        else:
            tau = max(gxs) * avg_all_distances / max_dis
        print("Tau Detected: ", tau)
        return tau, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis

    
def nan_to_num(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)

################################################################
##################EVT-VERSION###################################
################################################################
def get_tau(data,maxval,name,tailfrac=1,pcent=.99,usehigh=True,maxmodeerror=.05):
    tw =  weibull.weibull()
    tau = -1
    while(tau < 0):
      nbin=100
      nscale = 10
      fullrange = torch.linspace(0,maxval,nbin)
      fsize = max(3,int(tailfrac*len(data)))    
      if(usehigh):
          tw.FitHighTrimmed(data.view(1,-1),fsize)
      else:
          tw.FitLowReversed(data.view(1,-1),fsize)
      parms = tw.return_all_parameters()
      if(usehigh):
          tau=  parms['Scale']*np.power(-np.log((1-pcent)),(1/parms['Shape'])) - parms['translateAmountTensor'] + parms['smallScoreTensor']
      else:
          tau = parms['translateAmountTensor']- parms['smallScoreTensor']-(parms['Scale']*np.power(-np.log((pcent)),(1/parms['Shape'])))
      if(math.isnan(tau)):
          print( name , "Parms", parms)        
          tau = torch.mean(data)
      wmode = float(parms['translateAmountTensor']- parms['smallScoreTensor']+ (parms['Scale']*np.power((parms['Shape']-1)/(parms['Shape']),1./parms['Shape']          )))

      wscoresj = tw.wscore(fullrange)
      probj = nan_to_num(tw.prob(fullrange))
      if(torch.sum(probj) > .001):
          probj = probj/torch.sum(probj)
      datavect=data.numpy()
      histc,hbins = np.histogram(datavect,bins=nbin,range=[0,1])
      imode = hbins[np.argmax(histc[0:int(tau*nbin+1)])]
      merror = abs(imode-wmode)
      if(merror > maxmodeerror):
          #outlier detected, reduce tail fraion and force loop
          tailfrac = tailfrac - .05
          tau = -1
    print(name," EVT Tau with data fraction ", round(tailfrac*100, 2)," Percentile ",pcent*100,"   is ", float(tau.numpy()))
    return tau.numpy()

def nan_to_num1(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)


def get_tau1(data,maxval,tailfrac=.25,pcent=.999):
    #tw =  weibull.weibull(translateAmountTensor=.001)
    tw = weibull.weibull()
    nbin=200
    nscale = 10
    #fullrange = torch.linspace(0,torch.max(ijbbdata),nbin)
    fullrange = torch.linspace(0,maxval,nbin)
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    #print( name , "Data mean, max", torch.mean(ijbbdata),torch.max(ijbbdata))
    imean = torch.mean(data)
    istd = torch.std(data)
    imax = torch.max(data)
    tw.FitHighTrimmed(data.view(1,-1),int(tailfrac*len(data)))
    parms = tw.return_all_parameters()
    wscoresj = tw.wscore(fullrange)
    probj = nan_to_num(tw.prob(fullrange))
    if(torch.sum(probj) > .001):
        probj = probj/torch.sum(probj)
    tau=  parms['Scale']*np.power(-np.log((1-pcent)),(1/parms['Shape'])) - parms['translateAmountTensor'] + parms['smallScoreTensor']        
    return tau.numpy()

def thread(threads):
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()

def takeSecond(elem):
    return elem[1]
