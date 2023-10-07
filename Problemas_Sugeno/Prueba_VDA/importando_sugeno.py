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

path = 'C:/Users/marti/OneDrive/Documentos/GitHub/Artificial-Intelligence/Problemas_Sugeno/Sugeno_VDA/samplesVDA1.txt'


data = []
training_data = []
test_data = [] 

data = lee_arch(path)
X_train, X_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.2, random_state=42)
