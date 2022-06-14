# PEACH
PEACH Autonomous Clustering

Based on Pytorch and Flann

Requirement: Python3, Pytorch, PyFlann, Scikit-learn, SciPy.

Python Flann: "pip install pyflann-py3"

Example MNIST: "python example.py" 

Set "no_singleton = True" will cluster all samples (and take longer).

Set metric = "euclidean" will use Euclidean distance, default is Cosine.

Computing Cost: Cluster whole ImageNet2021 within 30 mins on single TITAN V
