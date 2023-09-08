import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
concept = np.arange(0,11,1)
numeric = np.arange(0,100,1)
total = np.arange(0,100,1)

# Generate fuzzy membership function

conceptReg = fuzz.trimf(concept, [0,0,7])
conceptBueno = fuzz.trimf(concept, [5, 8,10])
conceptExc = fuzz.trapmf(concept, [7,9,10,10])

#qual_lo = fuzz.gaussmf(x_qual, 2, 2)
#qual_md = fuzz.gaussmf(x_qual, 5, 1)
#qual_hi = fuzz.gaussmf(x_qual, 8, 0.5)

numericMin = fuzz.trimf(numeric, [0,0,50])
numericMed = fuzz.trimf(numeric, [30, 60,80])
numericMax = fuzz.trimf(numeric, [60,100,100]) 

totalMin = fuzz.trimf(total, [0, 0, 130])
totalMed = fuzz.trimf(total, [0, 130, 250])
totalMax = fuzz.trimf(total, [130, 250, 250])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(concept, conceptReg, 'b', linewidth=1.5, label='Regular')
ax0.plot(concept, conceptBueno, 'g', linewidth=1.5, label='Bueno')
ax0.plot(concept, conceptExc, 'r', linewidth=1.5, label='Excelente')
ax0.set_title('Concepto')
ax0.legend()

ax1.plot(numeric, numericMin, 'b', linewidth=1.5, label='Poor')
ax1.plot(numeric, numericMed, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(numeric, numericMax, 'r', linewidth=1.5, label='Amazing')
ax1.set_title('Nota numerica')
ax1.legend()

ax2.plot(total, totalMin, 'b', linewidth=1.5, label='Low')
ax2.plot(total, totalMed, 'g', linewidth=1.5, label='Medium')
ax2.plot(total, totalMax, 'r', linewidth=1.5, label='High')
ax2.set_title('Nota Final')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
qual_level_lo = fuzz.interp_membership(concept, conceptReg, 6.5)
qual_level_md = fuzz.interp_membership(concept, conceptBueno, 6.5)
qual_level_hi = fuzz.interp_membership(concept, conceptExc, 6.5)

serv_level_lo = fuzz.interp_membership(numeric, numericMin, 9.8)
serv_level_md = fuzz.interp_membership(numeric, numericMed, 9.8)
serv_level_hi = fuzz.interp_membership(numeric, numericMax, 9.8)

# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
active_rule1 = np.fmax(qual_level_lo, serv_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
tip_activation_lo = np.fmin(active_rule1, totalMin)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
tip_activation_md = np.fmin(serv_level_md, totalMed)

# For rule 3 we connect high service OR high food with high tipping
active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, totalMax)
tip0 = np.zeros_like(total)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(total, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(total, totalMin, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(total, tip0, tip_activation_md, facecolor='g', alpha=0.7)
ax0.plot(total, totalMed, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(total, tip0, tip_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(total, totalMax, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# Aggregate all three output membership functions together
aggregated = np.fmax(tip_activation_lo,
                     np.fmax(tip_activation_md, tip_activation_hi))

# Calculate defuzzified result
tip = fuzz.defuzz(total, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(total, aggregated, tip)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(total, totalMin, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(total, totalMed, 'g', linewidth=0.5, linestyle='--')
ax0.plot(total, totalMax, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(total, tip0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

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