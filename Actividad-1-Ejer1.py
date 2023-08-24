from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# armo dos grupos de datos
data_1 = np.random.randn(4,2) + 5       
data_2 = np.random.randn(4,2) + 10
print('data1 ', data_1)
print('data2 ',data_2)

# data_2 = np.random.randn(200,2) 
# data_3 = np.random.randn(200,2) 

data = np.concatenate((data_1, data_2), axis = 0)
print('data ', data)
plt.scatter(data[:,0], data[:,1])
plt.show()