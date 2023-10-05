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

plt.figure()
plt.scatter(training_data[:,0],training_data[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')
plt.show()