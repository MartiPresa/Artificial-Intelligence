from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Implementacion de kmeans para encontrar 2 clusters (k = 2) en datos de dimension 2 (d = 2)
k = 2
dim = 2
# genero los datos dispersos
data_1 = np.random.randn(4,2) + 5       
data_2 = np.random.randn(4,2) + 10
data = np.concatenate((data_1, data_2), axis = 0)

# print('data ', data)

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

def graficar(data,centers):
    plt.scatter(data[:,0], data[:,1], s=7)
    plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
    plt.show()

# graficar(data,centers)
# -----------------------INICIALIZACIONES-----------------------------

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

# inicializa un vector de cluster
# clusters = np.zeros(n)    
clusters = np.zeros(n)   
# inicializa la matriz de distancias
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# print("centers",centers,"\n")

# Salgo del while cuando los centros no variaron, es decir, el error es 0

def recalculaCentros(data,clusters,distances,error,centers,centers_new,centers_old):
    cantiteraciones = 0
    while error != 0  & cantiteraciones < 100 :
        # Measure the distance to every center
        for i in range(k):
            distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1) # devuelve el indice del elemento minimo
        
        # print("distancias",distances)
        print("clusters",clusters)

        centers_old = deepcopy(centers_new)

        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
        cantiteraciones = cantiteraciones + 1

    #---------------Calculo distancia INTRACLUSTER----------------------------

    indices = {}  # Crear un diccionario para almacenar las variables
    for i in range(k):
        nombre_variable = f"cluster{i}"
        arreglo = np.where(clusters == i)  # hago un arreglo con los indices de los datos que pertenecen al cluster i
        indices[nombre_variable] = arreglo
    
        # while indices[nombre_variable][0][j] != None: 
        #     print(indices[nombre_variable][0][j])
        
    for nombre, valor in indices.items():
        print(f"{nombre} = {valor}")
        # print(f"{nombre} = {valor[0][0]}")

    # distIntracluster = np.zeros((n,n))  

    for i in range(k):
        nombre_variable = f"cluster{i}"
        j = 0
        # print(indices[nombre_variable][nombre],indices[nombre_variable][valor])
        
    matricesDistIntracluster = {}  # Crear un diccionario para almacenar las variables
    for i in range(k):
        nombre_variable = f"cluster{i}"
        dim = len(indices[nombre_variable][0])
        # print(dim)
        mat = np.zeros((dim,dim))
        matricesDistIntracluster[nombre_variable] = mat

    dim = dim = len(indices[nombre_variable][0])

    # for j in range(dim):
    #     for x in range(dim):
    #         print(indices[nombre_variable][0][j], " menos ", indices[nombre_variable][0][x])
    #         # matricesDistIntracluster[nombre_variable][0][:,j] = np.linalg.norm(data[indices[nombre_variable][0][j]] - data[indices[nombre_variable][0][x]], axis=0)
            
    # for nombre, valor in matricesDistIntracluster.items():
    #     print(f"{nombre} = {valor}")
  #--------------------------------------------------------------------------------          
    # print("clusters",clusters)


recalculaCentros(data,clusters,distances,error,centers,centers_new,centers_old)
# print(clusters)



# colors=['orange', 'blue', 'green']
# for i in range(n):
#     plt.scatter(data[i, 0], data[i,1], s=7, color = colors[clusters[i]])
# plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)

# graficar(data,centers_new)