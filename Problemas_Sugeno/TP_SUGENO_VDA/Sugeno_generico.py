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
import clustering as cl
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
        self.mse_train = 0
        self.mse_test = 0

    def get_k(self):
        return self.k
    
    def getLabels(self):
        return self.labels
    
    def getClusters(self):
        return self.clusters
    def get_mse_train(self):
        return self.mse_train
    def get_mse_test(self):
        return self.mse_test

    def genfis_k(self, data, k):
        start_time = time.time()
        self.k = k
        self.labels, self.clusters = cl.clustering_kmeans(data,k) # hace clustering
        cluster_center = self.clusters
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1] 
        P = data[:,:-1] 
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.training(data)

    def genfis(self, data, radii,rb):

        start_time = time.time()
        self.labels, self.clusters = cl.clustering_kmeans(data, radii,rb) 
        cluster_center = self.clusters
        n_clusters = len(cluster_center)    #numero de clusters

        cluster_center = cluster_center[:,:-1] 
        P = data[:,:-1] 
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.training(data)

    def training(self, data):
        datos_entrada = data[:,:-1] 
        target = data[:,-1] 
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])

        f = [np.prod(gaussmf(datos_entrada,cluster,sigma),axis=1) for cluster in self.rules]
     
        nivel_acti = np.array(f).T 
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        datos_entrada = np.c_[datos_entrada, np.ones(len(datos_entrada))] 
        n_vars = datos_entrada.shape[1] 
        orden = np.tile(np.arange(0,n_vars), len(self.rules)) 
      
        acti = np.tile(nivel_acti,[1,n_vars]) 
        inp = datos_entrada[:, orden]
       
        A = acti*inp/sumMu 
        b = target

        #CUADRADOS MINIMOS --> lstsq 
        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None) 
        self.solutions = solutions 
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
        # mspe_test = mse_test * 100 / np.mean(target_test)
        # mspe_train = mse_train * 100 / np.mean(target_train)
        return self.error_porcentual(mse_train,training_data),self.error_porcentual(mse_test,test_data)
    
    def error_porcentual(self,mse,data):
        target_test = data[:,-1]
        return mse * 100 / np.mean(target_test)
    
    def grafica_mse(self, datos):
            salidas = self.evalModelo(np.vstack(datos[:,0]))
            error_test= mean_squared_error(datos[:, 1], salidas)
            error_test = self.error_porcentual(error_test,datos)
            error_test = "{:.2f}".format(error_test)
            # mspe_train = "{:.2f}".format(mspe_train)
            plt.figure(figsize=(10,5))
            plt.scatter(datos[:, 0], datos[:, 1], label='Targets', c='g')
            plt.scatter(datos[:, 0], salidas,
                        label='Salida del modelo (Y)', c='red')
            for i, point in enumerate(datos[:, 0]):
                x = datos[i, 0]
                y = datos[i, 1]
                color = 'red' if abs(salidas[i]-y)/100 > 0.2 else 'green'
                plt.plot([x, x], [y, salidas[i]], c=color,
                        alpha=0.5, linestyle='--')

            plt.xlabel('Tiempo [ms]')
            plt.ylabel('VDA')
            plt.title('Targets vs Salida del modelo')
            plt.text(200, 620, f'MSE Test= {error_test}%', bbox={
                    'facecolor': 'skyblue', 'alpha': 0.5, 'pad': 8})
            # plt.text(200, 580, f'MSE Train= {mspe_train}%', bbox={
            #         'facecolor': 'skyblue', 'alpha': 0.5, 'pad': 8})
            plt.legend()
            plt.grid(True)
            plt.show()

    def muestra(self,training_data,testing_data,title):
        data_x = training_data[:,0] 
        data_y = training_data[:,1] 
        reg = self.evalModelo(np.vstack(data_x))
        r,c = self.labels,self.clusters
        fig,(ax0, ax1) = plt.subplots(nrows=2,figsize=(15, 10))
        ax0.set_title(title)
        ax0.scatter(training_data[:,0],training_data[:,1], c=r)
        ax0.scatter(c[:,0],c[:,1], marker='X')
        self.viewInputs(ax1)
        ax1.set_title('Campanas de Gauss')
        ax0.scatter(c[:,0],c[:,1], marker='X')
        plt.show()

    def viewInputs(self, ax):
        gaussianas = []
        for input in self.inputs:
            gaussianas.append(input.view(ax))
        return gaussianas

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
