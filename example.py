from PEACH import PEACH
import numpy as np
import h5py
import torch
import csv
from sklearn import metrics
#data = np.random.rand(100,5)

########Get data###############
from torchvision import datasets, transforms
h = h5py.File('mnist_LeNet.hdf5')
features = h["features"]
features = np.array(features, dtype = float)

test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])), shuffle=False)
subjects = [int(target.cpu().numpy()) for data, target in test_loader]
count = 0
##############################
######use FACTO###############
result = PEACH(features, 0, no_singleton = False, metric = "cosine", batch_size = 4096) # 0 means GPU0

y_true = np.array(subjects)
y_pred = np.array(result)
NMI = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='max')
print("NMI:", NMI)
