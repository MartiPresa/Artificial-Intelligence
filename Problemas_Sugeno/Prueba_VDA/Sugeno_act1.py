import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
import time
from skfuzzy import control as ctrl
# from skfuzzy.membership import gaussmf
from random import choice
import pandas as pandas
from sklearn.metrics import mean_squared_error
import clusteringSustractivo as cl
from sklearn.model_selection import train_test_split

# Calcula el nivel de pertenencia de cada dato a las gaussianas
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

    def getCentroids(self):
        return self.centroids
    

    def view(self, ax):
        x = np.linspace(self.minValue,self.maxValue,20)

        for m in self.centroids:
            sigma = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,sigma)
            ax.plot(x,y)
    
class fis_Sugeno:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []
        self.clusters = []
        self.k = 0
        self.labels = []

    def get_k(self):
        return self.k
    
    def getLabels(self):
        return self.labels
    
    def getClusters(self):
        return self.clusters
    
    # def clusteringSustractivo(datos, r):
    #     return cl.clustering_sustractivo(datos,r)

    def genfis_k(self, data, k):
        start_time = time.time()
        self.k = k
        # self.labels, self.clusters = cl.clustering_sustractivo(data, radii,rb) # hace clustering
        self.labels, self.clusters = cl.clustering_kmeans(data,k) # hace clustering
        cluster_center = self.clusters
        # print(f'CLUSTERS = {self.clusters}')
        #print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)    #numero de clusters

        cluster_center = cluster_center[:,:-1] #se queda con todas las columnas menos la ultima
        P = data[:,:-1] # se queda con la primera columna, osea los X
        # T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        #print(f'INPUTS = {self.inputs}')
        self.rules = cluster_center
        self.training(data)

    def genfis(self, data, radii,rb):

        start_time = time.time()
        # self.labels, self.clusters = cl.clustering_sustractivo(data, radii,rb) # hace clustering
        self.labels, self.clusters = cl.clustering_kmeans(data, radii,rb) # hace clustering
        cluster_center = self.clusters
        # print(f'CLUSTERS = {self.clusters}')
        #print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)    #numero de clusters

        cluster_center = cluster_center[:,:-1] #se queda con todas las columnas menos la ultima
        P = data[:,:-1] # se queda con la primera columna, osea los X
        # T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        #print(f'INPUTS = {self.inputs}')
        self.rules = cluster_center
        self.training(data)

    def training(self, data):
        datos_entrada = data[:,:-1] #elimina la ultima columna
        target = data[:,-1] #se queda con la ultima columna
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        # Calcula los sigmas de cada campana de gauss
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])

        #F contiene un array por cada cluster, que tienen los arrays? ni puta idea
        f = [np.prod(gaussmf(datos_entrada,cluster,sigma),axis=1) for cluster in self.rules]
        # print(f'valor de F {f}')

        nivel_acti = np.array(f).T #pasa el valor de f traspuesto
        #quedan los valores de pertenencia a los clusters por columnas
        # print("nivel acti")
        # print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))# por cada dato suma el valor de pertenencia a cada cluster y lo pone en un vector
        # print("sumMu")
        # print(sumMu)

        datos_entrada = np.c_[datos_entrada, np.ones(len(datos_entrada))] #Agrega una columna de 1's a la matriz de datos de entrada
        # print(datos_entrada)
        n_vars = datos_entrada.shape[1] #almacena la cantidad de columnas en la matriz anterior
        #print(f'SHAPE DATOS {datos_entrada.shape}')

        #n_vars = 2
        # np.arange(0,n_vars) = [0,1]
        #tile repite lo que haya en el primer parametro, la cantidad de veces indicada en
        #el segundo parametro. En este caso escribe [0 1 0 1 0 1]
        orden = np.tile(np.arange(0,n_vars), len(self.rules)) 
        
        #En este caso, como el segundo parametro es [1,n_vars], devuelve un array que contiene
        #un array con la cantidad de elementos repetidos n_vars veces. 
        acti = np.tile(nivel_acti,[1,n_vars]) 
        inp = datos_entrada[:, orden]
        #Estos ultimos 2 pasos amplian las matrices a 6 columnas
        # print(f'acti {acti}')
        # print(f'inp{inp}')

        #Hace una especie de agregacion entre todas las reglas (Suavizar rectas)
        A = acti*inp/sumMu # hace una especie de promedio pesado CREO
        #acti: valor de pertenencia
        #inp: valor de la mf en un valor x 
        #sumMu: suma de los valores de pertenencia
        
       

        # print(f'VALOR DE A {A}')
        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = target

        #CUADRADOS MINIMOS --> lstsq 
        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None) #solutions es el mse CREO
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        # print(solutions)
        return 0

    def evalModelo(self, data):
        
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

    def mse(self,training_data,test_data):
        entrada_test = test_data[:,:-1]
        target_test = test_data[:,-1]
        target_train = training_data[:,-1]
        salida_test = self.evalModelo(np.vstack(entrada_test))
        salida_train = self.evalModelo(np.vstack(training_data[:,0]))
        mse_train = mean_squared_error(target_train, salida_train)
        mse_test = mean_squared_error(target_test, salida_test)
        return mse_train,mse_test

    def mspe(self,training_data,test_data): #Mean Squared Percentage Error
        entrada_test = test_data[:,:-1]
        target_test = test_data[:,-1]
        target_train = training_data[:,-1]
        salida_test = self.evalModelo(np.vstack(entrada_test))
        salida_train = self.evalModelo(np.vstack(training_data[:,0]))
        mse_train = mean_squared_error(target_train, salida_train)
        mse_test = mean_squared_error(target_test, salida_test)
        mspe_test = mse_test * 100 / np.mean(target_test)
        mspe_train = mse_train * 100 / np.mean(target_train)
        return mspe_train,mspe_test
    
    def muestra(self,training_data,testing_data,title):
        data_x = training_data[:,0] 
        data_y = training_data[:,1] # target
        reg = self.evalModelo(np.vstack(data_x))
        # r,c = cl.clustering_sustractivo(training_data,1)
        r,c = self.labels,self.clusters
        fig,(ax0, ax1) = plt.subplots(nrows=2,figsize=(15, 10))
        ax0.set_title(title)
        ax0.scatter(training_data[:,0],training_data[:,1], c=r)
        ax0.scatter(c[:,0],c[:,1], marker='X')
        self.viewInputs(ax1)
        ax1.set_title('Campanas de Gauss')
        # ax2.set_title('Rectas?')
        # ax2.plot(data_x,data_y)
        # ax2.plot(data_x,reg,linestyle='--')
        # ax2.scatter(testing_data[:,0],testing_data[:,1], c='green') # muestro los datos de test para ver cuanto error hay con el modelo
        ax0.scatter(c[:,0],c[:,1], marker='X')
        plt.show()

    def viewInputs(self, ax):
        gaussianas = []
        for input in self.inputs:
            gaussianas.append(input.view(ax))
        return gaussianas

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
