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

def J(strat):
   M1 = (sigma1*(strat[0] + D))
   M2 = (-sigma2*(strat[0] + D + strat[1]/2))
   M3 = (-2*(np.pi*strat[1])**2)
   M4 = (-((strat[0] + D0)**2 + (strat[1]**2)/2))
   M5 = (sigma1*(strat[2] + D))
   M6 = (-sigma2*(strat[2] + D + strat[3]/2))
   M7 = (-2*(np.pi*strat[3])**2)
   M8 = (-((strat[2] + D0)**2 + (strat[3]**2)/2))
   r = alpha_a*M5 + betta_a*M7 + delta_a*M8
   s = gamma_a*M6
   p = alpha_j*M1 + betta_j*M3 + delta_j*M4
   q = gamma_j*M2
   return np.array([s + p + q - np.sqrt(4*r*p + (p + q - s)**2)])

Aj = round(rn.uniform(-D, 0), 4)
Bj = round(rn.uniform(-min(Aj + D, -Aj), min(Aj + D, -Aj)), 4)
Aa = round(rn.uniform(-D, 0), 4)
Ba = round(rn.uniform(-min(Aa + D, -Aa), min(Aa + D, -Aa)), 4)
strategy = np.array([Aj, Bj, Aa, Ba])
abc = least_squares(J, strategy, bounds=(-120, 120))
abc.x

