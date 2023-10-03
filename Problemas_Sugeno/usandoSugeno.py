import Sugeno_act1 as sugeno

# Lectura de los datos, y selecciona cuales son para training y cuales para test
path ='Problemas_Sugeno/diodo.txt'


training_data = []
test_data = [] 
training_data, test_data = sugeno.eleccion_datos(path)

data_x = training_data[:,0] 
data_y = training_data[:,1]

#plt.plot(data_x, data_y)
# plt.ylim(-20,20)
# plt.xlim(-7,7)

# data = np.vstack((data_x, data_y)).T # ES NECESARIO? si queda igual que la matriz
Sugeno = sugeno.fis_Sugeno()
Sugeno.genfis(training_data, 1.1)
#Almacena la regresion lineal en r
reg = Sugeno.evalModelo(np.vstack(data_x))

# r,c = clustering_sustractivo(matriz_datos,1)
r,c = sugeno.fisInput.getCentroids
# r,c = sugeno.cl.clustering_sustractivo(training_data,1)


#print(f'SOLUCIONES {fis2.solutions}')
#print(f'REGLAS {fis2.rules}')


#plt.figure()
# plt.show()
#fis3.rules

# print(f'MSE train {mse_train}')
x,y = mse(training_data,test_data)
print(f'MSE test {y}')
print(f'MSE training {x}')

muestra(training_data,test_data)