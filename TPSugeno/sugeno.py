from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, Birch
from collections import namedtuple
import numpy as np


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

    def _test(self, data, target):
        pass

    def _mse(self):
        pass

    # Calcula el nivel de pertenencia de cada dato a las gaussianas
    def gaussmf(self, data, mean, sigma):
        return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

def clustering_factory(n_clusters, data, type: ['kmeans','birch']):
    pass
    