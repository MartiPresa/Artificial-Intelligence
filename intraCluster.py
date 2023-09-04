from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.cluster as skl


indices = {}  # Crear un diccionario para almacenar las variables

k = 2

for i in range(k):
    nombre_variable = f"cluster{i}"
    arreglo = np.where(clusters == i)
    indices[nombre_variable] = arreglo

# Ahora puedes acceder a las variables usando sus nombres almacenados en el diccionario
for nombre, valor in variables.items():
    print(f"{nombre} = {valor}")

# Distancia entre los datos de un mismo cluster
def calculaIntracluster():
    distIntracluster = np.zeros((n,n))  # k = clusters, dim = dimension
    
    for nombre, valor in indices.items():
        valor.__sizeof__
        nombre_variable = f"cluster{i}"
        indices[nombre_variable].legth
        matriz_extraida = data[indices_filas][:, :] 
        distIntracluster[:,i] = np.linalg.norm(data - centers[i], axis=1)

matricesDistIntracluster = {}  # Crear un diccionario para almacenar las variables
for i in range(k):
    nombre_variable = f"cluster{i}"
    mat = np.zeros()
    matricesDistIntracluster[nombre_variable] = arreglo