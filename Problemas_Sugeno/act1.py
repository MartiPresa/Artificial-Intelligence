import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
import time
from skfuzzy import control as ctrl
from skfuzzy.membership import gaussmf

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


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []

    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = subclust2(data, radii)

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        print("nivel acti")
        print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        print("sumMu")
        print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        print(solutions)
        return 0

    def evalfis(self, data):
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


    def viewInputs(self):
        for input in self.inputs:
            input.view()

path ='4to/Inteligencia Artificial/Practica/Artificial-Intelligence/Problemas_Sugeno/diodo.txt'
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
matriz_datos = np.array(datos)

# x=np.linspace(-10,10,50)
# X,Y = np.meshgrid(x,x)
# Z = X**2+Y**2

# data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# fis3 = fis()
# fis3.genfis(data,1.2)
# fis3.viewInputs()

# r = fis3.evalfis(np.vstack((X.ravel(), Y.ravel())).T)
# r = np.reshape(r, X.shape)
# Usar esos datos para generar X, Y y Z
"""
X = matriz_datos[:, 0]
Y = matriz_datos[:, 1]
# Z = matriz_datos[:, 2]

# Crear una matriz 2D de X y Y utilizando meshgrid
X, Y = np.meshgrid(X, Y)

# Crear un objeto FIS (Sistema de Inferencia Fuzzy)
fis3 = ctrl.Antecedent(np.vstack((X.ravel(), Y.ravel())), 'input')
fis3['input'] = fis3.universe

# Definir una función de membresía gaussiana para 'input'
media = np.mean(np.vstack((X.ravel(), Y.ravel())), axis=1)
desviacion_estandar = np.std(np.vstack((X.ravel(), Y.ravel())), axis=1)
fis3['input'] = gaussmf(fis3.universe, media, desviacion_estandar)

# Crear reglas y definir la lógica difusa como lo haces en tu código original

# Evaluar el FIS
r = fis3.defuzzify(np.vstack((X.ravel(), Y.ravel())).T)
r = np.reshape(r, X.shape)


fig = plt.figure()
ax = fig.add_subplot(projection = '3d') #fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Z, cmap=cm.Blues,
                        linewidth=0, antialiased=False, alpha=0.3)

surf = ax.plot_surface(X,Y, r, cmap=cm.Reds,
                        linewidth=0, antialiased=False, alpha=0.8)
"""

def my_exponential(A, B, C, x):
    return A*np.exp(-B*x)+C

data_x = matriz_datos[:,0]
data_y = matriz_datos[:,1]

plt.plot(data_x, data_y)
# plt.ylim(-20,20)
plt.xlim(-7,7)

data = np.vstack((data_x, data_y)).T
fis2 = fis()
fis2.genfis(data, 1.1)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')

#print(f'SOLUCIONES {fis2.solutions}')
print(f'REGLAS {fis2.rules}')

r,c = subclust2(matriz_datos,1)
plt.figure()
plt.scatter(matriz_datos[:,0],matriz_datos[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')
plt.show()

#fis3.rules


plt.show()