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
from pyflann import *
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

################################################################
##################Tau-Simple###################################
################################################################
def tolerance(features, gpu):
    # Use up to 10000 simples to compute tau
    if len(features) >= 10000:
        select_features = np.array(random.choices(features, k = 10000))
    else:
        select_features = features

############################################
    total_distances = []
    max_dis = []
    torch.cuda.set_device(gpu)
    X_global = torch.tensor(select_features).cuda()
    Y_global = torch.tensor(select_features).cuda()
    distances = cosine(X_global, Y_global)
    distances = distances.cpu().numpy()
    total_distances.append(np.median(distances))
    max_dis.append(np.max(distances))
    del X_global, Y_global
################################################
    avg_all_distances = np.median(total_distances)
    tra = Normalizer(norm='l2').fit(features)
    X = tra.transform(features)
    # Use FLANN to find KNN
    flann = FLANN()
    result, result_dis = flann.nn(X, X, num_neighbors=2, algorithm="kdtree", trees=8, checks=128)
    nearest_cluster = np.array([cls[1] for cls in result])
    nearest_points_dis = np.array([dis[1] for dis in result_dis])
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

    #print("Computing Nearest init groups")
    centroids = [np.mean(i,axis=0) for i in init_features]
    tra = Normalizer(norm='l2').fit(centroids)
    X = tra.transform(centroids)
    flann = FLANN()
    result, result_dis = flann.nn(X, X, num_neighbors=2, algorithm="kdtree", trees=8, checks=128)
    nearest_init = np.array([cls[1] for cls in result])

    ##########################################################################################################
    #print("Computing tolerance")
    nearest_init_combo = [[m, init[n]] for m, n in zip(init, nearest_init)]
    ########################################################################################
    gxs = []
    #print("Computing taus")
    for pair1, pair2 in nearest_init_combo:
        features0 = [features[k] for k in pair1] #extract features of cluster0
        features1 = [features[k] for k in pair2] #extract features of cluster1
        centroid0 = np.mean(features0, axis=0) # Get controid of cluster0
        centroid1 = np.mean(features1, axis=0) # Get controid of cluster1
        gx = scipy.spatial.distance.cosine(centroid0, centroid1) * 1
        gxs.append(gx)
    name = 'name'
    #tau = get_tau(torch.Tensor(nearest_points_dis),1,name,tailfrac=1,pcent=.999,usehigh=True,maxmodeerror=1)* avg_all_distances / max(max_dis)
    tau = max(gxs) * avg_all_distances / max(max_dis)
    return 0, 0, tau, nearest_points, init_length, nearest_cluster_with_distance_round_1, nearest_points_dis, 0, 0
    
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
