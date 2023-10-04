import random
import numpy as np

class geneticAlgorithm():

    def __init__(self, fitness, poblacion) -> None:
        pass



class AG():

    def __init__(self,K, D, start, end,cant_padres):
        self.poblacion = self.poblacionInicial(K, D, start, end)
        self.cromosomas = self.enteroABinario(self.poblacion)
        self.fitness = self.calculo_fitness_N(self.poblacion)
        self.cant_padres = cant_padres

    def fitness(self,x):
    #Funcion objetivo
        return 300-(x-15)**2

    def poblacionInicial(self,K, D, start, end):
        poblacion = []
        for i in range(K):
            #Asignacion de atributos random a cada individuo
            for j in range(D):
                poblacion.append(random.randint(start,end))
        return np.array(poblacion)

    #convertimos los atributos a binario
    def enteroABinario(self,values):
        binarios = []
        cant_bits = len(np.binary_repr(max(values)))
        for value in values:
            binarios.append(np.binary_repr(value, cant_bits))
        return binarios

    def binarioAEntero(self,values):
        enteros = []
        for value in values:
            enteros.append(int(str(value),2))
        return enteros
            
    def calculo_fitness_N(self,poblacion):
        x = []
        for individuo in poblacion:
            x.append(self.fitness(individuo))
        return np.array(x)

    def normalizacion(self,mat): #probabilidad de seleccion
        suma = np.sum(mat)
        return mat/suma

    def mejor_individuo(self,mat):
        return np.max(mat)

    def mejor_individuo(self):
        return np.max(self.fitness)
    
    def seleccion_padres(self):
        #Roulette
        probabilidad = self.normalizacion(self.fitness)
        print(probabilidad)
        padres = []
        while(len(padres) < self.cant_padres):
            padres.append(np.random.choice(self.poblacion,p=probabilidad))
        return padres

    def mejores_individuos(self,cant_individuos, fitness):
        mejores = []
        while(len(mejores) < cant_individuos):
            maximo = fitness.pop(max(fitness))
            mejores.append(maximo)
        return mejores

    def generarHijos(self,padres, cromosomas, cant_hijos, puntos_cruce):
        crossover = []
        for padre in padres:
            cromosomas[padre]

# N poblacion inicial
N = 6
cant_padres = 2
ag = AG(N,1,0,31,cant_padres)

# poblacion = poblacionInicial(N,1,0,31)
# cromosomas = enteroABinario(poblacion)
print(ag.poblacion)
print(f'cromosomas {ag.cromosomas}')
# fitness = calculo_fitness_N(poblacion)
print(ag.fitness)
print(f'Mejor individuo: {ag.mejor_individuo()}')
padres = []
padres = ag.seleccion_padres()
print(f'Padres {padres}')
# elite = mejores_individuos(2, fitness[:]) # se pasa fitness como copia
# print(f'Elite {elite}')