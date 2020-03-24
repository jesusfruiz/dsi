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
    kmeans = KMeans(n_clusters=K, init='random', n_init=20)
    kmeans.fit(data)
    BIC.append(bic(K, kmeans.labels_, data))
    K += 1

X = list(range(2, 11))

plt.scatter(X, BIC)
plt.plot()
plt.show()

#3. Principio de Pareto
value_per_client = (data.sum(axis=1)).sort_values(ascending=False)
total_value = value_per_client.sum()    

client_20percent = int(value_per_client.size*0.2)
value_80percent = total_value*0.8

value = 0

for client in range(0,client_20percent):
    value += value_per_client[client]    
    
print('El 20 % de los clientes suponen el', value*100/total_value, 
      '% de los ingresos\n')

value2 = 0
clients2 = 0
for client_value in value_per_client:
    if (value2 <= value_80percent):
        value2 += client_value
        clients2 += 1

print('El 80 % de los ingresos provienen del', 
      clients2*100/value_per_client.shape[0], '% de los clientes\n')


#Los resultados del BIC indican que el número ideal de clusters es K=4
K=4

# Jackknife #
kmeans = KMeans(n_clusters=K, init='random', n_init=40)
SSE = dict()
for i in range(len(data)):
    data_aux = data.drop(i)
    kmeans.fit(data_aux)
    SSE[i] = kmeans.inertia_

sigma=statistics.stdev(SSE.values())
mu=statistics.mean(SSE.values())
umbral=2;

outliers = []
for i in range(len(data)):
    if abs(SSE[i]-mu)>umbral*sigma:
        outliers.append(i);

estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data)
print(estimator.explained_variance_ratio_) 

for i in range(len(X_pca)):
    if i in outliers:
        col = 'k'
    else:
        col = 'w'
        
    plt.plot(X_pca[i, 0], X_pca[i, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=5)
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()

print("Los outliers son:")
for i in outliers:
    print(i)
    print(data.iloc[i,:])
    print("")
    data = data.drop(i) #Remove the outlier from data
    
# Calc PCA without outliers    
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data)
print(estimator.explained_variance_ratio_)

#4. Parametrización del algoritmo por medio del método de k-distancias
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(X_pca)

minPts= 4 #Elección debido a la gran acumulación de los grupos
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

# 5. Algoritmo DBSCAN con el eps=4500 obtenido en k-distancias
db = DBSCAN(eps=4500, min_samples=minPts).fit(data)
core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Numero de clusteres obtenidos: %d' % n_clusters_)

from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))

estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(data)
print(estimator.explained_variance_ratio_) 

#pintamos el clustering
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

# Hito 2: Data Analysis #
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


#3. Estudio estadístico
print('El G0 ingresa', groups[0].sum(axis=1).sum(), 'euros con', groups[0].shape[0], 'personas')
print('El G1 ingresa', groups[1].sum(axis=1).sum(), 'euros con', groups[1].shape[0], 'personas')
print('El G2 ingresa', groups[2].sum(axis=1).sum(), 'euros con', groups[2].shape[0], 'personas')
print('El grupo de outliers ingresa', groups[-1].sum(axis=1).sum(), 'euros con', groups[-1].shape[0], 'personas')
print()

# gastos/producto
#groups[1].sum()

# porcentaje gastos/producto
#for product in groups[1].sum():
#    print((product/groups[1].sum().sum())*100)

print('El G0 consume 40% frescos, 22% ultramarinos')
print('El G1 consume 40% ultramarinos y 25% lácteos')
print('El G2 consume 82% frescos')
print('El grupo de outliers consume 31% frescos, 26% ultramarinos y 19% lácteos')


# Test paramétrico
#import scipy.stats as ss
##media, desviacion = ss.norm.fit(data["Altura"])
#
#means = dict()
#deviations = dict()
#
##parametric_test = dict
#
#for i in groups_types:
#    means[i] = pd.DataFrame()
#    deviations[i] = pd.DataFrame()
#        
#    means[i].append(ss.norm.fit(groups[i]['Fresh'])[0])
#    means[i]['Fresh'], deviations[i]['Fresh'] = ss.norm.fit(groups[i]['Fresh'])

#    means[i], deviations[i] = ss.norm.fit(groups[i]['Fresh'])
#    means[i], deviations[i] = ss.norm.fit(groups[i]['Milk'])
#    means[i], deviations[i] = ss.norm.fit(groups[i]['Grocery'])
#    means[i], deviations[i] = ss.norm.fit(groups[i]['Frozen'])
#    means[i], deviations[i] = ss.norm.fit(groups[i]['Detergents_Paper'])
#    means[i], deviations[i] = ss.norm.fit(groups[i]['Delicassen'])
    
#    means['Milk'], deviations['Milk'] = ss.norm.fit(groups[0]['Milk'])
#    means['Grocery'], deviations['Grocery'] = ss.norm.fit(groups[0]['Grocery'])
#    means['Frozen'], deviations['Frozen'] = ss.norm.fit(groups[0]['Frozen'])
#    means['Detergents_Paper'], deviations['Detergents_Paper'] = ss.norm.fit(groups[0]['Detergents_Paper'])
#    means['Delicassen'], deviations['Delicassen'] = ss.norm.fit(groups[0]['Delicassen'])

        

    

