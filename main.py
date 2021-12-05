import random as rn
import numpy as np
from scipy.optimize import least_squares

alpha_j = 0.0016
alpha_a = 0.006
betta_j = 0.0000007
betta_a = 0.000000075
gamma_j = 0.00008
gamma_a = 0.004
delta_j = 0.000016
delta_a = 0.00006
sigma1 = 1
sigma2 = 1

D = 120
D0 = 70  

for i in range(2, 9):
   for j in range(1, i):
      a = 12*i + j*2 

print(a)