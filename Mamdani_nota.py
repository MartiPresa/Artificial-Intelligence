"""
Reglas de Mamdani

R1
- Si NOTA es BAJA entonces DESAPROBADO

R2
- SR1 Si CONCEPTO es REGULAR y NOTA es MEDIA entonces HABILITADO <-- min|__ max
- SR2 Si CONCEPTO es REGULAR y NOTA es ALTA entonces HABILITADO <--min  |

R3
- SR3 Si CONCEPTO es BUENO y NOTA es ALTA entonces PROMOCION   <-- min   | 
- SR4 Si CONCEPTO es EXCELENTE y NOTA es ALTA entonces PROMOCION <-- min |---> max
- SR5 Si CONCEPTO es EXCELENTE y NOTA es MEDIA entonces PROMOCION <-- min| 

"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

concept = np.arange(0, 10.2, 0.2)
numeric = np.arange(0, 101, 1)
total = np.arange(0, 101, 1)

# FUZZIFICATION

# Generate fuzzy membership function
conceptReg = fuzz.gaussmf(concept, 0, 4)
conceptBueno = fuzz.gaussmf(concept, 8, 1.5)
conceptExc = fuzz.gaussmf(concept, 10, 1.5)

numericBajo = fuzz.trimf(numeric, [0, 0, 50])
numericMed = fuzz.trimf(numeric, [30, 55, 80])
numericAlto = fuzz.trimf(numeric, [60, 100, 100])

totalMin = fuzz.trimf(total, [0, 0, 40])
totalMed = fuzz.trimf(total, [30, 50, 70])
totalMax = fuzz.trimf(total, [60, 100, 100])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(concept, conceptReg, 'b', linewidth=1.5, label='Regular')
ax0.plot(concept, conceptBueno, 'g', linewidth=1.5, label='Bueno')
ax0.plot(concept, conceptExc, 'r', linewidth=1.5, label='Excelente')
ax0.set_title('Concepto')
ax0.legend()

ax1.plot(numeric, numericBajo, 'b', linewidth=1.5, label='Bajo')
ax1.plot(numeric, numericMed, 'g', linewidth=1.5, label='Medio')
ax1.plot(numeric, numericAlto, 'r', linewidth=1.5, label='Alto')
ax1.set_title('Nota numerica')
ax1.legend()

ax2.plot(total, totalMin, 'b', linewidth=1.5, label='Desaprobacion')
ax2.plot(total, totalMed, 'g', linewidth=1.5, label='Habilitacion')
ax2.plot(total, totalMax, 'r', linewidth=1.5, label='Promocion')
ax2.set_title('Nota Final')
ax2.legend()

# INFERENCE

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!

concept_level_reg = fuzz.interp_membership(concept, conceptReg, 5)
concept_level_bueno = fuzz.interp_membership(concept, conceptBueno, 5)
concept_level_exc = fuzz.interp_membership(concept, conceptExc, 5)

numeric_level_bajo = fuzz.interp_membership(numeric, numericBajo, 100)
numeric_level_med = fuzz.interp_membership(numeric, numericMed, 100)
numeric_level_alto = fuzz.interp_membership(numeric, numericAlto, 100)

print(f'CONCEPTO REG{concept_level_reg} \n CONCEPTO BUENO{concept_level_bueno} \n CONCEPTO EXC{concept_level_exc} ')
print(f'NOTA BAJA{numeric_level_bajo} \n NOTA MEDIA{numeric_level_med} \n NOTA ALTA{numeric_level_alto} ')

# Rule 1 -DESAPROBAR
nota_final_des = np.fmin(numeric_level_bajo, totalMin)

# Rule 2 -HABILITAR
subrule1 = (concept_level_reg * 0.2 + numeric_level_med * 0.8)
subrule2 = (concept_level_reg* 0.2 + numeric_level_alto * 0.8)
# subrule1 or subrule2 --> fmax
active_rule2 = np.fmax(subrule1, subrule2) 

# hacemos multiplicacion en vez de usar el fmin?
nota_final_hab = np.fmin(active_rule2, totalMed) 

# Rule 3 - PROMOCIONAR
subrule3 = (concept_level_bueno * 0.2 + numeric_level_alto * 0.8)
subrule4 = (concept_level_exc * 0.2 + numeric_level_alto * 0.8)
subrule5 = np.fmin(concept_level_exc, numeric_level_med)
active_rule3 = np.fmax(subrule3, np.fmax(subrule4,subrule5))
#active_rule3 = np.fmax(subrule3, subrule4)

# TRUNCA el grafico de nota final max
nota_final_promo = np.fmin(active_rule3, totalMax)

# pone un piso para rellenar la funcion truncada
nota0 = np.zeros_like(total)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(total, nota0, nota_final_hab, facecolor='b', alpha=0.7)
ax0.plot(total, totalMin, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(total, nota0, nota_final_des, facecolor='g', alpha=0.7)
ax0.plot(total, totalMed, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(total, nota0, nota_final_promo, facecolor='r', alpha=0.7)
ax0.plot(total, totalMax, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('INFERENCIA')
plt.tight_layout()

# AGREGATION
aggregated = np.fmax(nota_final_hab,
                     np.fmax(nota_final_des, nota_final_promo))

# DEFUZZIFICATION
nota = fuzz.defuzz(total, aggregated, 'centroid')
print(f'NOTA = {nota}')
nota_activation = fuzz.interp_membership(total, aggregated, nota)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(total, totalMin, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(total, totalMed, 'g', linewidth=0.5, linestyle='--')
ax0.plot(total, totalMax, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(total, nota0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([nota, nota], [0, nota_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('AGREGACION')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# Generate trapezoidal membership function on range [0, 1]
x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])

# Defuzzify this membership function five ways
defuzz_centroid = fuzz.defuzz(x, mfx, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x, mfx, 'bisector')
defuzz_mom = fuzz.defuzz(x, mfx, 'mom')
defuzz_som = fuzz.defuzz(x, mfx, 'som')
defuzz_lom = fuzz.defuzz(x, mfx, 'lom')

# Collect info for vertical lines
labels = ['centroid', 'bisector', 'mean of maximum', 'min of maximum',
          'max of maximum']
xvals = [defuzz_centroid,
         defuzz_bisector,
         defuzz_mom,
         defuzz_som,
         defuzz_lom]
colors = ['r', 'b', 'g', 'c', 'm']
ymax = [fuzz.interp_membership(x, mfx, i) for i in xvals]

# Display and compare defuzzification results against membership function
plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)
plt.ylabel('Fuzzy membership')
plt.xlabel('Universe variable (arb)')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()
