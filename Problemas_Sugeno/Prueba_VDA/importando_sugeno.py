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
from Sugeno_act1 import fis_Sugeno
import clusteringSustractivo as cl
import copy

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

def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.2, random_state=42)
    training_data =  np.column_stack((X_train, y_train))
    test_data =  np.column_stack((X_test,y_test))
    return training_data,test_data

def graficar(auxDatos):

    # Extraer los valores de x y y de auxDatos
    valores_x = auxDatos[:, 0]  # Columna 0 (valorx)
    valores_y = auxDatos[:, 1]  # Columna 1 (valory)

    # Crear el gráfico
    plt.scatter(valores_x, valores_y, label = "Datos")  # Graficar los datos
    plt.xlabel('Tiempo [ms]')
    plt.ylabel('VDA')
    plt.title('Gráfico de datos')
    plt.legend()  # Mostrar la leyenda
    plt.grid(True)  # Mostrar cuadrícula
    plt.show()  # Mostrar el gráfico

def sobremuestreo(data):
   
    num_filas, num_columnas = data.shape
    promedios = np.zeros((num_filas-1, num_columnas))
    for i in range(num_filas-1):
        promedios [i,:] = (abs(data[i,:] + data[i+1,:])/2.0)

    return promedios


path = 'C:/Users/marti/OneDrive/Documentos/GitHub/Artificial-Intelligence/Problemas_Sugeno/Sugeno_VDA/samplesVDA1.txt'
datos = lee_arch(path)
data = []
training_data = []
test_data = [] 

data = lee_arch(path)
training_data, test_data = split_data(data)
data_x = training_data[:,0] 
data_y = training_data[:,1] # target


# graficar(data)

#--------------------------Un solo modelo---
# sugeno = fis_Sugeno()
# # sugeno.genfis(training_data, 0.55,rb) # ra,rb
# sugeno.genfis_k(training_data,5)
# #Almacena la regresion lineal en r
# reg = sugeno.evalModelo(np.vstack(data_x))
# x,y = sugeno.mse(training_data,test_data)
# sugeno.muestra(training_data,test_data)

#------------------------------Modelos con clustering sustractivo
# mse = []
# ratio = []
# i = 0
# rb = 0.0
# while (i < 40):
#     sugeno = fis_Sugeno()
#     sugeno.genfis(training_data, 0.55,rb) # ra,rb
#     #Almacena la regresion lineal en r
#     reg = sugeno.evalModelo(np.vstack(data_x))
#     x,y = sugeno.mse(training_data,test_data)
#     mse.append(y)
#     ratio.append(rb)
#     # print(f'MSE test {y}')
#     # print(f'MSE training {x}')
#     # sugeno.muestra(training_data,test_data)
#     i += 1
#     rb += 0.1
# print(mse)
# print(ratio)
# plt.scatter(mse, ratio, label='Datos')  # Graficar los datos
# plt.xlabel('mse')
# plt.ylabel('ratio')
# plt.title('mse vs rb')
# plt.legend()  # Mostrar la leyenda
# plt.grid(True)  # Mostrar cuadrícula
# plt.show()  # Mostrar el gráfico


#------------------------------Modelos con clustering Kmeans
mse = []
test_mse = 10000
training_mse = 0
k_clusters = []
modelos = []
k = 0
i = 3
while (i <= 20):
    sugeno = fis_Sugeno()
    modelos.append(sugeno)
    sugeno.genfis_k(training_data,i) # ra,rb
    reg = sugeno.evalModelo(np.vstack(data_x))
    x,y = sugeno.mse(training_data,test_data)
    mse.append(y)
    if(test_mse > y): # guardo el mejor modelo
        test_mse = y
        training_mse = x
        k = i
        mejor_sugeno = copy.deepcopy(sugeno)
    k_clusters.append(i)
    i += 1

# plt.scatter(k_clusters,mse, label='Datos')  # Graficar los datos
# plt.ylabel('mse')
# plt.xlabel('Cantidad de reglas R')
# plt.title('R vs mse')
# plt.yticks(mse)
# plt.xticks(k_clusters)
# plt.legend()  # Mostrar la leyenda
# plt.grid(True)  # Mostrar cuadrícula
# plt.show()  # Mostrar el gráfico

plt.plot(k_clusters, mse, label='Datos', marker='o', linestyle='-')  # Graficar una línea con puntos
plt.ylabel('mse')
plt.xlabel('Cantidad de reglas R')
plt.title('R vs mse')
plt.yticks(mse)
plt.xticks(k_clusters)
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar cuadrícula
plt.show()  # Mostrar el gráfico

mejor_sugeno.muestra(training_data,test_data,f"Sugeno con {k} clusters")
mejor_sugeno.grafica_mse(test_data,training_mse)
# print(f"training_mse {training_mse}")
#------------------Entreno Sugeno con todos los datos y con los hiperparametros del mejor_sugeno
nuevo_sugeno = fis_Sugeno()
nuevo_sugeno.genfis_k(data,k)
nuevo_training_mse,y = nuevo_sugeno.mse(data,data)
# print(f"test error {test_mspe}")
# print(f"training error {training_mspe}")
# print(f"training error nuevo sugeno{x}")

datos_sobremuestreo = sobremuestreo(data)
mejor_sugeno.grafica_mse(datos_sobremuestreo,training_mse)

nuevo_mse,y = nuevo_sugeno.mse(datos_sobremuestreo,datos_sobremuestreo)
nuevo_sugeno.grafica_mse(datos_sobremuestreo,nuevo_training_mse)
# viejo_mspe, x = mejor_sugeno.mspe(datos_sobremuestreo,datos_sobremuestreo)

# graficar(datos_sobremuestreo)
# print(f"modelo viejo: test error sobremuestreo {viejo_mspe}")
# print(f"modelo nuevo: test error sobremuestreo {nuevo_mspe}")
