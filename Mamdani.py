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
# -----------------------------------------------

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
# qual_level_lo = fuzz.interp_membership(x_qual, qual_lo, 6.5)
concept_level_lo = fuzz.interp_membership(concept, conceptBmin, 6.5)
concept_level_md = fuzz.interp_membership(concept, conceptBmed, 6.5)
concept_level_hi = fuzz.interp_membership(concept, conceptBmax, 6.5)


numeric_level_lo = fuzz.interp_membership(numeric, numericBmin, 9.8)
numeric_level_md = fuzz.interp_membership(numeric, numericBmed, 9.8)
numeric_level_hi = fuzz.interp_membership(numeric, numericBmax, 9.8)

# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
active_rule1 = np.fmax(qual_level_lo, serv_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
tip_activation_lo = np.fmin(active_rule1, tip_lo)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
tip_activation_md = np.fmin(serv_level_md, tip_md)

# For rule 3 we connect high service OR high food with high tipping
active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, tip_hi)
tip0 = np.zeros_like(x_tip)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_tip, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_tip, tip0, tip_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip, tip0, tip_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
