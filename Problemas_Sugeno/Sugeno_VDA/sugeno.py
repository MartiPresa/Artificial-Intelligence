import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
import time
from sklearn.model_selection import train_test_split
# from skfuzzy import control as ctrl
from random import choice
import pandas as pandas
from sklearn.metrics import mean_squared_error
# import Sugeno_act1 as sugeno
import clusteringSustractivo as cl

# ruta del archivo de datos
path= '/Users/valentina/Documents/GitHub/Artificial-Intelligence/Problemas_Sugeno/Sugeno_VDA/samplesVDA1.txt'
# path = 'C:/Users/marti/OneDrive/Documentos/GitHub/Artificial-Intelligence/Problemas_Sugeno/Sugeno_VDA/samplesVDA1.txt'

def eleccion_datos(datos):
    # X = np.array(datos)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
    # scaler = StandardScaler().fit(X_train)
   
    # aux_datos = datos
    datos_test = []
    cant_datos_test = int(len(datos)*0.2)

    #Selecciona datos de test al azar
    data_frame = pandas.DataFrame(datos)
    filas_aleatorias = data_frame.sample(n=cant_datos_test)
    datos_test = filas_aleatorias.values

    datos_training = []
    [datos_training.append(x) for x in datos if x not in datos_test]
    # print(f'DATOS = {datos_training}')
    # print(f' cantidad de datos entrenamiento= {len(datos_training)}')

    return np.array(datos_training),np.array(datos_test)

def lee_arch(path):
    datos = []
    # Abre el archivo y lee los datos
    with open(path, "r") as file:
        valorx = 0
        for linea in file:
            # Convierte los valores de cadena a números de punto flotante
            valory = float(linea.strip())
            # Agrega los valores a la lista de datos
            datos.append([valorx, valory])
            valorx = valorx+2.5

    # Convierte la lista de datos en una matriz NumPy
    auxDatos = np.array(datos)
   
    return auxDatos

# Inciso A) Grafico del VDA vs tiempo
def muestra(auxDatos):

    # Extraer los valores de x y y de auxDatos
    valores_x = auxDatos[:, 0]  # Columna 0 (valorx)
    valores_y = auxDatos[:, 1]  # Columna 1 (valory)

    # Crear el gráfico
    plt.scatter(valores_x, valores_y, label='Datos')  # Graficar los datos
    plt.xlabel('Valor de x')
    plt.ylabel('Valor de y')
    plt.title('Gráfico de datos')
    plt.legend()  # Mostrar la leyenda
    plt.grid(True)  # Mostrar cuadrícula
    plt.show()  # Mostrar el gráfico

datos = lee_arch(path)
# muestra(datos)

training_data = []
test_data = [] 
training_data, test_data = eleccion_datos(datos)
# print(training_data)
data_x = training_data[:,0] 
# print(data_x)
data_y = training_data[:,1]

# hago clustering 
r, c = cl.clustering_sustractivo(training_data,1)
# grafico clusters
plt.figure()
plt.scatter(training_data[:,0],training_data[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')
plt.show()

# ------------------------------------------------------GEN FIS------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:56:16 2020

@author: Daniel Albornoz

Implementación similar a genfis2 de Matlab.
Sugeno type FIS. Generado a partir de clustering substractivo.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import time

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []


    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = cl.clustering_sustractivo(data, radii)

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        print("nivel acti")
        print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        print("sumMu")
        print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        print(solutions)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()



# ------------------------------------------------------TEST GEN FIS 1 ed------------------------------------------------------
def my_exponential(A, B, C, x):
    return A*np.exp(-B*x)+C

# data_x = np.arange(-10,10,0.1)
# data_y = -0.5*data_x**3-0.6*data_x**2+10*data_x+1 #my_exponential(9, 0.5,1, data_x)

plt.plot(data_x, data_y)
# plt.ylim(-20,20)
# plt.xlim(-7,7)

data = np.vstack((data_x, data_y)).T

fis2 = fis()
fis2.genfis(data, 1.1)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')
plt.show()
fis2.solutions
fis2.rules