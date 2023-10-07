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
    plt.scatter(valores_x, valores_y, label='Datos')  # Graficar los datos
    plt.xlabel('Valor de x')
    plt.ylabel('Valor de y')
    plt.title('Gráfico de datos')
    plt.legend()  # Mostrar la leyenda
    plt.grid(True)  # Mostrar cuadrícula
    plt.show()  # Mostrar el gráfico



path = 'C:/Users/marti/OneDrive/Documentos/GitHub/Artificial-Intelligence/Problemas_Sugeno/Sugeno_VDA/samplesVDA1.txt'
datos = lee_arch(path)
data = []
training_data = []
test_data = [] 

data = lee_arch(path)
training_data, test_data = split_data(data)
data_x = training_data[:,0] 
data_y = training_data[:,1] # target
mse = []
ratio = []
# graficar(data)
i = 0
rb = 0.0
while (i < 40):
    
    sugeno = fis_Sugeno()
    sugeno.genfis(training_data, 1.1,rb) # ra,rb
    #Almacena la regresion lineal en r
    reg = sugeno.evalModelo(np.vstack(data_x))
    x,y = sugeno.mse(training_data,test_data)
    mse.append(y)
    ratio.append(rb)
    # print(f'MSE test {y}')
    # print(f'MSE training {x}')
    # sugeno.muestra(training_data,test_data)
    i += 1
    rb += 0.1

print(mse)
print(ratio)
plt.scatter(mse, ratio, label='Datos')  # Graficar los datos
plt.xlabel('mse')
plt.ylabel('ratio')
plt.title('mse vs rb')
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar cuadrícula
plt.show()  # Mostrar el gráfico

# print(training_data)