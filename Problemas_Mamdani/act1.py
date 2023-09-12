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

tamano_X = np.arange(-20, 20, 1)
tamano_y = np.arange(-3, 15, 0.1)


# FUZZIFICATION

# Generate fuzzy membership function  
# Un sistema de inferencia difuso con una sola entrada y una única salida se describe mediante las siguientes reglas:
# SI 𝒙 ES pequeño ENTONCES 𝒚 es pequeño
# SI 𝒙 ES mediano ENTONCES 𝒚 es mediano
# SI 𝒙 ES grande ENTONCES 𝒚 es grande

# Las variables lingüísticas de la entrada se definen mediante funciones trapezoidales:
# 𝜇𝑝𝑒𝑞𝑢𝑒ñ𝑜(𝑥) = 𝑡𝑟𝑎𝑝𝑚𝑓(𝑥,[−20, −15, −6, −3])
# 𝜇𝑚𝑒𝑑𝑖𝑎𝑛𝑜(𝑥) = 𝑡𝑟𝑎𝑝𝑚𝑓(𝑥,[−6, −3,3,6])
# 𝜇𝑔𝑟𝑎𝑛𝑑𝑒(𝑥) = 𝑡𝑟𝑎𝑝𝑚𝑓(𝑥,[3,6,15,20])

# Las variables lingüísticas de la salida son:
# 𝜇𝑝𝑒𝑞𝑢𝑒ñ𝑜(𝑦) = 𝑡𝑟𝑎𝑝𝑚𝑓(𝑦,[−2.46, −1.46,1.46,2.46])
# 𝜇𝑚𝑒𝑑𝑖𝑎𝑛𝑜(𝑦) = 𝑡𝑟𝑎𝑝𝑚𝑓(𝑦,[1.46,2.46,5,7])
# 𝜇𝑔𝑟𝑎𝑛𝑑𝑒(𝑦) = 𝑡𝑟𝑎𝑝𝑚𝑓(𝑦,[5,7,13,15])

# Observar que entrada y salida se definen sobre universos diferentes.
# Calcular por el método de centroide la salida a las entradas x = 8, -5, 5, 8.

upeq_x = fuzz.trapmf(tamano_X,[-20,-15,-6,-3])
umed_x = fuzz.trapmf(tamano_X, [-6,-3,3,6])
ugrande_x = fuzz.trapmf(tamano_X,[3,6,15,20])

upeq_y = fuzz.trapmf(tamano_y,[-2.46,-1.46,1.46,2.46] )
umed_y = fuzz.trapmf(tamano_y,[1.46,2.46,5,7])
ugrande_y = fuzz.trapmf(tamano_y, [5,7,13,15])

# Visualize these universes and membership functions
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 9))

# value_x = 8
# value_x = -8
# value_x = 5
value_x = -5

ax0.plot(tamano_X, upeq_x, 'b', linewidth=1.5, label='Piccolo')
ax0.plot(tamano_X, umed_x, 'g', linewidth=1.5, label='Mediano')
ax0.plot(tamano_X, ugrande_x, 'r', linewidth=1.5, label='Grande')

ax0.set_title('Valor de x')
ax0.legend()

ax1.plot(tamano_y, upeq_y, 'b', linewidth=1.5, label='Piccolo')
ax1.plot(tamano_y, umed_y, 'g', linewidth=1.5, label='Mediano')
ax1.plot(tamano_y, ugrande_y, 'r', linewidth=1.5, label='Grande')
ax1.set_title('Valor de y')
ax1.legend()

# INFERENCE

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!

x_peq = fuzz.interp_membership(tamano_X, upeq_x, value_x)
x_med = fuzz.interp_membership(tamano_X, umed_x, value_x)
x_grande = fuzz.interp_membership(tamano_X, ugrande_x, value_x)

# Rule 1 
y_peq = np.fmin(x_peq, upeq_y)


# Rule 2 
y_med = np.fmin(x_med, umed_y) 

# Rule 3
y_grande = np.fmin(x_grande, ugrande_y)

print("ypeq",y_peq) 
print("ymed",y_med)
print("ygrande",y_grande)


print(f'X_PEQ = {x_peq}')
print(f'X_MED = {x_med}')
print(f'X_GRANDE = {x_grande}')

# pone un piso para rellenar la funcion truncada
salida = np.zeros_like(tamano_y)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(tamano_y, salida, y_peq, facecolor='b', alpha=0.7)
ax0.plot(tamano_y, y_peq, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(tamano_y, salida, y_med, facecolor='g', alpha=0.7)
ax0.plot(tamano_y, y_med, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(tamano_y, salida, y_grande, facecolor='r', alpha=0.7)
ax0.plot(tamano_y, y_grande, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('INFERENCIA')
plt.tight_layout()

# AGREGATION
aggregated = np.fmax(y_med,
                     np.fmax(y_peq, y_grande))

# DEFUZZIFICATION
y_final = fuzz.defuzz(tamano_y, aggregated, 'centroid')
print(f'VALOR DE Y = {y_final}')
y_activation = fuzz.interp_membership(tamano_y, aggregated, y_final)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(tamano_y, y_peq, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(tamano_y, y_med, 'g', linewidth=0.5, linestyle='--')
ax0.plot(tamano_y, y_grande, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(tamano_y, salida, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([y_final, y_final], [0, y_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('AGREGACION')

plt.show()
