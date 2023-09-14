import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Lee los datos con la libreria pandas
archivo_ruta = 'D:/Backup/Facultad/4to/Inteligencia Artificial/Practica/datos.txt'
lineas = pd.read_csv(archivo_ruta, header=0, names=['x','y'], sep=r'\s+') 

#Obtiene un 20% de los datos al azar (replace = False evita que tomes el mismo dato que ya usaste)
lineas_aleatorias = lineas.sample(frac=0.2, replace=False, random_state=1)

#usamos el KMeans de la liberaria sklearn y aplicamos el clustering
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(lineas_aleatorias)

#Graficamos
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

#ax1 corresponde al grafico de todos los datos
ax1.scatter(lineas['x'], lineas['y'], s=20, c='blue')

#ax2 corresponde al grafico del 20% de los datos
ax2.scatter(lineas_aleatorias['x'], lineas_aleatorias['y'], s=20, c='purple')

#ax3 corresponde al grafico de los datos clusterizados
ax3.scatter(lineas_aleatorias.loc[y_kmeans == 0, 'x'], lineas_aleatorias.loc[y_kmeans == 0, 'y'], s=20, c='purple')
ax3.scatter(lineas_aleatorias.loc[y_kmeans == 1, 'x'], lineas_aleatorias.loc[y_kmeans == 1, 'y'], s=20, c='orange')
ax3.scatter(lineas_aleatorias.loc[y_kmeans == 2, 'x'], lineas_aleatorias.loc[y_kmeans == 2, 'y'], s=20, c='pink')

#en ax3 se grafica cada centro de cluster
ax3.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 30, c = 'red', label = 'Centroids')

#mostramos los graficos
plt.show()