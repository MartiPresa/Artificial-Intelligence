from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Implementacion de kmeans para encontrar 2 clusters (k = 2) en datos de dimension 2 (d = 2)
k = 2

# genero los datos dispersos
data_1 = np.random.randn(4,2) + 5       
data_2 = np.random.randn(4,2) + 10
print('data1 ', data_1)
print('data2 ',data_2)
data = np.concatenate((data_1, data_2), axis = 0)
print('data ', data)

# elijo 3 centros al azar (puedo generarlos con un random)
# center_1 = np.array([1,1])
# center_2 = np.array([5,5])
# center_3 = np.array([8,1])

# @title 
# Number of clusters
k = 2
# Number of training data (cantidad de datos)
n = data.shape[0] 
# Number of features in the data (DIMENSION : cantidad de columnas de los vectores)
c = data.shape[1]  #renombrarlo d

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0) # promedio
std = np.std(data, axis = 0) # desviacion estandar
centers = np.random.randn(k,c)*std + mean   # para que elijo 3 centros a mano arriba si aca los genero random????

# Plot the data and the centers generated as random
plt.scatter(data[:,0], data[:,1], s=7)
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
plt.show()