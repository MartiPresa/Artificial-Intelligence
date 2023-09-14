import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
import time
from skfuzzy import control as ctrl
from skfuzzy.membership import gaussmf
from random import choice
import pandas as pandas
from sklearn.metrics import mean_squared_error

#ALGORITMO DE CLUSTERING SUSTRACTIVO
def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb==0:
        Rb = Ra*1.15
    
    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    P = distance_matrix(ndata,ndata)
    alpha=(Ra/2)**2
    P = np.sum(np.exp(-P**2/alpha),axis=0)

    centers = []
    i=np.argmax(P)
    C = ndata[i]
    p=P[i]
    centers = [C]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
        restarP = True
        i=np.argmax(P)
        C = ndata[i]
        p=P[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,C))
        elif p<RejectRatio*pAnt:
            continuar=False
        else:
            dr = np.min([np.linalg.norm(v-C) for v in centers])
            if dr/Ra+p/pAnt>=1:
                centers = np.vstack((centers,C))
            else:
                P[i]=0
                restarP = False
        if not any(v>0 for v in P):
            continuar = False
    distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)
    return labels, centers

def gaussmf(data, mean, sigma):
    "Calcula el nivel de pertenencia de cada dato a las gaussianas"
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


    def view(self, ax):
        x = np.linspace(self.minValue,self.maxValue,20)
        #plt.figure()
        #gauss = []
        for m in self.centroids:
            sigma = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,sigma)
            ax.plot(x,y)
            #gauss.append((x,y))
        #return gauss
    
class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []

    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = subclust2(data, radii) # hace clustering

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)    #numero de clusters

        cluster_center = cluster_center[:,:-1] #se queda con todas las columnas menos la ultima
        P = data[:,:-1] # se queda con la primera columna, osea los X
        # T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        datos_entrada = data[:,:-1] #elimina la ultima columna
        target = data[:,-1] #se queda con la ultima columna
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        # Calcula los sigmas de cada campana de gauss
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])

      
        #F contiene un array por cada cluster, que tienen los arrays? ni puta idea
        f = [np.prod(gaussmf(datos_entrada,cluster,sigma),axis=1) for cluster in self.rules]
        print(f'valor de F {f}')

        nivel_acti = np.array(f).T #pasa el valor de f traspuesto
        #quedan los valores de pertenencia a los clusters por columnas
        print("nivel acti")
        print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))# por cada dato suma el valor de pertenencia a cada cluster y lo pone en un vector
        print("sumMu")
        print(sumMu)

        datos_entrada = np.c_[datos_entrada, np.ones(len(datos_entrada))] #Agrega una columna de 1's a la matriz de datos de entrada
        print(datos_entrada)
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
        print(f'acti {acti}')
        print(f'inp{inp}')

        #Hace una especie de agregacion entre todas las reglas (Suavizar rectas)
        A = acti*inp/sumMu # hace una especie de promedio pesado CREO
        #acti: valor de pertenencia
        #inp: valor de la mf en un valor x 
        #sumMu: suma de los valores de pertenencia
        
       

        print(f'VALOR DE A {A}')
        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = target

        #CUADRADOS MINIMOS --> lstsq 
        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None) #solutions es el mse CREO
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        print(solutions)
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


    def viewInputs(self, ax):
        gaussianas = []
        for input in self.inputs:
            gaussianas.append(input.view(ax))
        return gaussianas

# path ='4to/Inteligencia Artificial/Practica/Artificial-Intelligence/Problemas_Sugeno/diodo.txt'
path ='diodo.txt'
datos = []
# Abre el archivo y lee los datos
with open(path, "r") as file:
    for linea in file:
        # Divide cada línea en dos valores utilizando el separador (tabulación o espacio)
        valores = linea.strip().split('\t')  
        # Convierte los valores de cadena a números de punto flotante
        valores = [float(valor) for valor in valores]
        # Agrega los valores a la lista de datos
        datos.append(valores)

# Convierte la lista de datos en una matriz NumPy
auxDatos = datos[:]
print(f' TODOS LOS DATOS = {datos}')
print(f' cantidad de datos total= {len(datos)}')
datos_test = []
cant_datos_test = int(len(datos)*0.2)

#Selecciona datos de test al azar
data_frame = pandas.DataFrame(datos)
filas_aleatorias = data_frame.sample(n=cant_datos_test)
datos_test = filas_aleatorias.values

datos = []
[datos.append(x) for x in auxDatos if x not in datos_test]
print(f'DATOS = {datos}')
print(f' cantidad de datos entrenamiento= {len(datos)}')

matriz_datos = np.array(datos)

data_x = matriz_datos[:,0] 
data_y = matriz_datos[:,1]

#plt.plot(data_x, data_y)
# plt.ylim(-20,20)
# plt.xlim(-7,7)

data = np.vstack((data_x, data_y)).T # ES NECESARIO? si queda igual que la matriz
fis2 = fis()
fis2.genfis(data, 1.1)
#Almacena la regresion lineal en r
reg = fis2.evalModelo(np.vstack(data_x))

r,c = subclust2(matriz_datos,1)

fig,(ax0, ax1, ax2) = plt.subplots(nrows=3,figsize=(15, 10))
ax0.set_title('Clustering de los datos')
ax0.scatter(matriz_datos[:,0],matriz_datos[:,1], c=r)
ax0.scatter(c[:,0],c[:,1], marker='X')
fis2.viewInputs(ax1)
ax1.set_title('Campanas de Gauss')
ax2.set_title('Rectas?')
ax2.plot(data_x,data_y)
ax2.plot(data_x,reg,linestyle='--')

#print(f'SOLUCIONES {fis2.solutions}')
#print(f'REGLAS {fis2.rules}')


#plt.figure()
# plt.show()
#fis3.rules

#TESTEAMOS CON LOS DATOS DE TEST
entrada_test = datos_test[:,:-1]
target_test = datos_test[:,-1]
# target_train = datos[:,-1]

salida_test = fis2.evalModelo(np.vstack(entrada_test))
# salida_train = fis2.evalModelo(np.vstack(datos[:,0]))

# mse_train = mean_squared_error(target_train, salida_train)
mse_test = mean_squared_error(target_test, salida_test)

# print(f'MSE train {mse_train}')
print(f'MSE test {mse_test}')
plt.show()