import random
import numpy as np



def fitness(x):
    #Funcion objetivo
    return 300-(x-15)**2

def poblacionInicial(K, D, start, end):
    poblacion = []
    for i in range(K):
        #Asignacion de atributos random a cada individuo
        for j in range(D):
            poblacion.append(random.randint(start,end))
    return np.array(poblacion)

#convertimos los atributos a binario
def enteroABinario(values):
    binarios = []
    cant_bits = len(np.binary_repr(max(values)))
    for value in values:
        binarios.append(np.binary_repr(value, cant_bits))
    return binarios

def binarioAEntero(values):
    enteros = []
    for value in values:
        enteros.append(int(str(value),2))
    return enteros
        
def calculo_fitness_N(poblacion):
    x = []
    for individuo in poblacion:
        x.append(fitness(individuo))
    return np.array(x)

def normalizacion(mat):
    suma = np.sum(mat)
    return mat/suma

def seleccion_padres(cant_padres, poblacion, fitness):
    #Roulette
    probabilidad = normalizacion(fitness)
    print(probabilidad)
    padres = []
    while(len(padres) < cant_padres):
        padres.append(np.random.choice(poblacion,p=probabilidad))
    return padres

def mejores_individuos(cant_individuos, fitness):
    mejores = []
    while(len(mejores) < cant_individuos):
        maximo = fitness.pop(max(fitness))
        mejores.append(maximo)
    return mejores

def generarHijos(padres, cromosomas, cant_hijos, puntos_cruce):
    crossover = []
    for padre in padres:
        cromosomas[padre]

# N poblacion inicial
N = 6
cant_padres = 2
poblacion = poblacionInicial(N,1,0,31)
cromosomas = enteroABinario(poblacion)
print(poblacion)
print(f'cromosomas {cromosomas}')
fitness = calculo_fitness_N(poblacion)
print(fitness)
padres = []
padres = seleccion_padres(cant_padres,poblacion,fitness)
print(f'Padres {padres}')
elite = mejores_individuos(2, fitness[:]) # se pasa fitness como copia
print(f'Elite {elite}')