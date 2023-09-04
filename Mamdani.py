"""
Ejercicio: Sistema de puntuación 
Concepto: B-, B, B+
Numerico: [0-100]
Total: [0-100]
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

concept = np.arange(0,11,1)
numeric = np.arange(0,100,1)
total = np.arange(0,100,1)

#print(concept)

conceptBmin = fuzz.trimf(concept, [0,0,7])
conceptBmed = fuzz.trimf(concept, [5, 8,10])
conceptBmax = fuzz.trapmf(concept, [7,9,10,10])

numericBmin = fuzz.trimf(numeric, [0,0,50])
numericBmed = fuzz.trimf(numeric, [30, 60,80])
numericBmax = fuzz.trimf(numeric, [60,100,100]) 
#numericBmax = fuzz.trapmf(numeric, [70,80,100,100]) 

fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 9))

ax0.plot(concept, conceptBmin, 'b', linewidth=1.5, label='Bien-')
ax0.plot(concept, conceptBmed, 'g', linewidth=1.5, label='Bien')
ax0.plot(concept, conceptBmax, 'r', linewidth=1.5, label='Bien+')
ax0.set_title('Concept')
ax0.legend()

ax1.plot(numeric, numericBmin, 'b', linewidth=1.5, label='Baja')
ax1.plot(numeric, numericBmed, 'g', linewidth=1.5, label='Media')
ax1.plot(numeric, numericBmax, 'r', linewidth=1.5, label='Alta')
ax1.set_title('Numeric')
ax1.legend()

plt.show()