import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
import time
from sklearn.model_selection import train_test_split
from random import choice
import pandas as pandas
from sklearn.metrics import mean_squared_error
from Sugeno_generico import fis_Sugeno
import clustering as cl
import copy

def lee_arch(path):
    datos = []
    with open(path, "r") as file:
        valorx = 0
        for linea in file:
            valory = float(linea.strip())
            datos.append([valorx, valory])
            valorx = valorx+2.5
    auxDatos = np.array(datos)
    return auxDatos

def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.2, random_state=42)
    training_data =  np.column_stack((X_train, y_train))
    test_data =  np.column_stack((X_test,y_test))
    return training_data,test_data

def graficar(auxDatos):

    valores_x = auxDatos[:, 0]  
    valores_y = auxDatos[:, 1] 

    plt.scatter(valores_x, valores_y, label = "Datos") 
    plt.xlabel('Tiempo [ms]')
    plt.ylabel('VDA')
    plt.title('Gráfico de datos')
    plt.legend() 
    plt.grid(True) 
    plt.show()  #

def sobremuestreo(data):
   
    num_filas, num_columnas = data.shape
    promedios = np.zeros((num_filas-1, num_columnas))
    for i in range(num_filas-1):
        promedios [i,:] = (abs(data[i,:] + data[i+1,:])/2.0)

    return promedios


path = 'Problemas_Sugeno\TP_SUGENO_VDA\samplesVDA1.txt'
datos = lee_arch(path)
data = []
training_data = []
test_data = [] 

data = lee_arch(path)
training_data, test_data = split_data(data)
data_x = training_data[:,0] 
data_y = training_data[:,1] # target

graficar(data)

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
    mse_train,mse_test = sugeno.mspe(training_data,test_data) # retorna mse_train,mse_test,mspe_train,mspe_test
    mse.append(mse_test)
    if(test_mse > mse_test): # guardo el mejor modelo
        test_mse = mse_test
        training_mse = mse_train
        k = i
        mejor_sugeno = copy.deepcopy(sugeno)
    k_clusters.append(i)
    i += 1


plt.plot(k_clusters, mse, label='Datos', marker='o', linestyle='-')  # Graficar una línea con puntos
plt.ylabel('mspe')
plt.xlabel('Cantidad de reglas R')
plt.title('R vs mspe')

# Establecer el rango en el eje Y para mostrar todos los datos sin superponerse
plt.ylim(min(mse) - 0.05, max(mse) + 0.05)

plt.xticks(k_clusters)
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar cuadrícula
plt.show()  # Mostrar el gráfico


mejor_sugeno.muestra(training_data,test_data,f"Sugeno con {k} clusters")
mejor_sugeno.grafica_mse(test_data)

#------------------Entreno Sugeno con todos los datos y con los hiperparametros del mejor_sugeno
nuevo_sugeno = fis_Sugeno()
nuevo_sugeno.genfis_k(data,k)
nuevo_training_mse,mse_test = nuevo_sugeno.mse(data,data)

datos_sobremuestreo = sobremuestreo(data)
mejor_sugeno.grafica_mse(datos_sobremuestreo)

nuevo_mse,mse_test = nuevo_sugeno.mse(datos_sobremuestreo,datos_sobremuestreo)
nuevo_sugeno.grafica_mse(datos_sobremuestreo)


