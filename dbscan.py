# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:04:46 2020

@author: jesus
"""

import pandas as pd
import io
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
import sklearn.neighbors
from sklearn.neighbors import kneighbors_graph
import statistics
import math

def bic(K, cidx, X):
    k = 0
    P = len(X.iloc[1,:])
    N = len(X)
    sigma_j = dict()
    xi = []
    while k < K:
        suma = 0
        group_k = list(filter(lambda x: cidx[x] == k, range(len(cidx))))
        sigma = dict()
        sigma_j[k] = dict()
        j = 0
        while j < P:    
            sigma[j] = statistics.stdev(X.iloc[:,1])**2
            if len(group_k) < 2:
                sigma_j[k][j] = 0
            else:                
                sigma_j[k][j] = statistics.stdev(X.iloc[group_k,1])**2
            suma = suma + 0.5 * math.log(sigma[j] + sigma_j[k][j])    
            j+=1
        xi.append(-1 * len(group_k) * suma)
        k+=1
    return -2*sum(xi)+2*K*P*math.log(N)    


######## MAIN ########
df = pd.read_csv("Wholesale customers data.csv")

#1. Variables que deben intervenir
data = df.iloc[:,2:]

#2. Calculo número ideal de clusters
K=2
BIC = []

while K <= 10:
    kmeans = KMeans(n_clusters=K, init='random', n_init=10)
    kmeans.fit(data)
    BIC.append(bic(K, kmeans.labels_, data))
    K += 1

X = list(range(2, 11))

plt.scatter(X, BIC)
plt.plot()
plt.show()

#Los resultados del BIC indican que el número ideal de clusters es K=4
K=4

estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data)
#data = X_pca
print(estimator.explained_variance_ratio_) 

plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()

dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(X_pca)

minPts=4
A = kneighbors_graph(X_pca, minPts, include_self=False)
Ar = A.toarray()

seq = []
for i,s in enumerate(X_pca):
    for j in range(len(X_pca)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])
            
seq.sort()
plt.plot(seq)
plt.show()

db = DBSCAN(eps=4000, min_samples=minPts).fit(data)
core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Numero de clusteres obtenidos: %d' % n_clusters_)
from sklearn import metrics
#metrics.
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))

estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data)
print(estimator.explained_variance_ratio_) 

unique_labels = set(labels)
colors = plt.cm.Spectral(numpy.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Utilizamos blanco para el ruido
        col = 'w'
           
    class_member_mask = (labels == k)
    xy = X_pca[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=17)

    xy = X_pca[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=5)

plt.show()

# Data Analysis #
groups_types = set(labels)
groups = dict()
representatives = dict()
i = 0;

for i in groups_types:
    groups[i] = pd.DataFrame()
    representatives[i] = dict()

for i in range(len(data)):
    groups[labels[i]] = groups[labels[i]].append(data.iloc[i,:])
    
for i in groups_types:
    atributes = len(groups[i].iloc[0,:])    
    for j in range(atributes):
        representatives[i][groups[i].iloc[:,j].name] = statistics.mean(
                groups[i].iloc[:,j])



        

    

