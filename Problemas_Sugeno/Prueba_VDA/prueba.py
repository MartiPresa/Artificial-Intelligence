import numpy as np

def calcular_promedio_consecutivo_por_columna(data):
   
    num_filas, num_columnas = data.shape
    promedios = np.zeros((num_filas-1, num_columnas))
    # promedios[-1,:] = data[-1,:]
    # if(num_filas%2):
    #     promedios = np.zeros((num_filas, num_columnas))
    #     promedios[-1,:] = data[-1,:]
    # else:
    #     promedios = np.zeros((num_filas, num_columnas))
    
    for i in range(num_filas-1):
        print(i)
        promedios [i,:] = (abs(data[i,:] + data[i+1,:])/2.0)

    return promedios

# Ejemplo de matriz de elementos
matriz_original = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [1,1,1,1],[2,2,2,2]])

# Calcular el promedio de elementos consecutivos por columna
promedios_consecutivos_por_columna = calcular_promedio_consecutivo_por_columna(matriz_original)

# Imprimir los promedios
print("Promedios de elementos consecutivos por columna:")
print(promedios_consecutivos_por_columna)
