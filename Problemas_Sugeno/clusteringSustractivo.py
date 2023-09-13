"""Subtractive Clustering Algorithm
"""
__author__ = 'Daniel Albornoz'


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb==0:
        Rb = Ra*1.15
    
    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    # 14/05/2020 cambio list comprehensions por distance matrix
    #P = np.array([np.sum([np.exp(-(np.linalg.norm(u-v)**2)/(Ra/2)**2) for v in ndata]) for u in ndata])
    #print(P)
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

# c1 = np.random.rand(15,2)+[1,1]
# c2 = np.random.rand(10,2)+[10,1.5]
# c3 = np.random.rand(5,2)+[4.9,5.8]
# m = np.append(c1,c2, axis=0)
# m = np.append(m,c3, axis=0)

# r,c = subclust2(m,2)

# plt.figure()
# plt.scatter(m[:,0],m[:,1])
# plt.scatter(c[:,0],c[:,1], marker='X')
# print(c)

# c1 = np.random.rand(150,2)+[1,1]
# c2 = np.random.rand(100,2)+[10,1.5]
# c3 = np.random.rand(50,2)+[4.9,5.8]
# m = np.append(c1,c2, axis=0)
# m = np.append(m,c3, axis=0)
path ='4to/Inteligencia Artificial/Practica/Artificial-Intelligence/Problemas_Sugeno/diodo.txt'
#print (m)

# Inicializa una lista vacía para almacenar los datos
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

# Imprime la matriz resultante
print(matriz_datos)

# print(diodo)
r,c = subclust2(matriz_datos,1)
plt.figure()
plt.scatter(matriz_datos[:,0],matriz_datos[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')
plt.show()