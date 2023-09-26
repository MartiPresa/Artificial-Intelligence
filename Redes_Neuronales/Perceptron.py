import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

class PerceptronSimple:
    
    def __init__(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs
    

    def reporte(self):
        return self.weights, self.bias
    
        # Función de activación: Heaviside
    def activation(self, z):
        return np.heaviside(z, 0)
    
    def fit(self, X, y): # en X cantidad de c y de k
        n_features = X.shape[1]
        
        # Inicialización de parámetros (w y b)
        self.weights = np.zeros((n_features)) # la misma cantidad que las entradas
        self.bias = 0
        
        # Iterar n épocas
        for epoch in range(self.epochs): # iteramos segun la cantidad de epocas
            
            # De a un dato a la vez
            for i in range(len(X)):
                z = np.dot(X, self.weights) + self.bias # Producto escalar de entradas y pesos + b
                y_pred = self.activation(z)             # Función de activación no lineal (Heaviside)
                
                #Actualización de pesos y bias
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i]) * X[i] # y[i] - y_pred[i]) es el error basicamente xd
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])
                
        return self.weights, self.bias
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

# EL BIAS NO ESTA ATADO A NINGUN VALOR DE ENTRADA, MODIFICA LA ORDENADA AL ORIGEN, PERMITE MOVER LA RECTA

#-------------------------------------------------------------------------

iris = load_iris() 

X = iris.data[:, (0, 1)] # petal length, petal width
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#-------------------------------------------------------------------------
# plot
fig, ax = plt.subplots()
ax.scatter(X[:,1], X[:,0],c=y)
plt.show()

#-------------------------------------------------------------------------
