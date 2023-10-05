from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, Birch
from collections import namedtuple
import matplotlib as plt
import numpy as np
import pandas

Features = namedtuple('Input', ['max', 'min' ,'centroid'])

Rule = namedtuple('Rule', ['centroid' ,'sigma'])

class Sugeno():

    def __init__(self):
        self.rules = []
        self.memberfunc = []
        self.features = []

    def _train(self, data, target, n_clusters):
        clusters = clustering_factory(n_clusters, data,'birch')
        maxValue = np.max(data, axis=0)
        minValue = np.min(data, axis=0)

        self.features = list(Input(maxValue[i], minValue[i],clusters[:,i]) for i in enumerate(clusters))
        
        #Calculo del sigma de la campana de gauss
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])

        act = list(self.gaussmf(data,cluster,sigma))
        pass

    def _test(self, data, target):
        pass

    def _mse(self):
        pass

    # Calcula el nivel de pertenencia de cada dato a las gaussianas
    def gaussmf(self, data, mean, sigma):
        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

def clustering_factory(n_clusters, data, type: ['kmeans','birch']):
    pass
    
def cargar_datos(path):
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
    return np.array(datos)

def dividir_datos(datos, porcentaje):
    datos_test = []
    cant_datos_test = int(len(datos)*porcentaje)

    #Selecciona datos de test al azar
    data_frame = pandas.DataFrame(datos)
    filas_aleatorias = data_frame.sample(n=cant_datos_test)
    datos_test = filas_aleatorias.values

    datos_training = []
    [datos_training.append(x) for x in datos if x not in datos_test]
    return datos_training, datos_test

def graficar(datos):

    # Extraer los valores de x y y de datos
    val_x = datos[:, 0]  
    val_y = datos[:, 1] 

    plt.scatter(val_x, val_y)  
    plt.xlabel('Tiempo')
    plt.ylabel('VDA')
    plt.title('Gráfico de datos')
    plt.grid(True) 
    plt.show()

path = 'samplesVDA1.txt'
datos = cargar_datos(path)
graficar(datos)

training_data = []
test_data = [] 
training_data, test_data = sugeno.eleccion_datos(path)
 
data_x = training_data[:,0] 
data_y = training_data[:,1] 

