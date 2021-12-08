import random as rn
import numpy as np
from scipy.optimize import least_squares
import pandas as pd
from collections import defaultdict

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
Nmax = 30
data_strat = pd.read_csv('data_strat.csv')

# Подсчет макропараметров
def macroparams(data_strat):
   Aj = data_strat['A_J'].tolist()
   Bj = data_strat['B_J'].tolist()
   Aa = data_strat['A_A'].tolist()
   Ba = data_strat['B_A'].tolist()
   collect = defaultdict(list)
   collect_M = defaultdict(list)
   collect_MM = defaultdict(list)
   for i in range(1, 9):
      collect['M' + str(i)] = []
   for i in range(0, Nmax):
      collect['M1'].append(sigma1*(Aj[i] + D))
      collect['M2'].append(-sigma2*(Aj[i] + D + Bj[i]/2))
      collect['M3'].append(-2*(np.pi*Bj[i])**2)
      collect['M4'].append(-((Aj[i] + D0)**2 + (Bj[i]**2)/2))
      collect['M5'].append(sigma1*(Aa[i] + D))
      collect['M6'].append(-sigma2*(Aa[i] + D + Ba[i]/2))
      collect['M7'].append(-2*(np.pi*Ba[i])**2)
      collect['M8'].append(-((Aa[i] + D0)**2 + (Ba[i]**2)/2)) 
   for i in range(1, 9):
      for j in range(1, 9):
         if i == j:
            collect_M['M' + str(i) + 'M' + str(j)].append(np.array(collect['M' + str(i)])*np.array(collect['M' + str(j)])[0])
         else:
            continue
   for i in range(2, 9):
      for j in range(1, i):
         collect_MM['M' + str(i) + 'M' + str(j)].append(2*np.array(collect['M' + str(i)])*np.array(collect['M' + str(j)])[0])

   M1, M2, M3 = macro_norm(collect, collect_M, collect_MM)
   data_macro = {}
   data = pd.DataFrame(data = data_macro)
   for i in range(1, 9):
      data['M' + str(i) + '_n'] = M1['M' + str(i) + '_n'][0]
   """ for i in range(1, 9):
      for j in range(1, 9):
         if i == j:
            data['M' + str(i) + 'M' + str(j) + '_n'] = M2['M' + str(i) + 'M' + str(j) + '_n'][0]
         else:
            continue """
   """ for i in range(2, 9):
      for j in range(1, i):
         data['M' + str(i) + 'M' + str(j) + '_n'] = M3['M' + str(i) + 'M' + str(j) + '_n'] """
   data.to_csv("data_macro.csv", index=False)
   return M3

# Нормализуем макропараметры для увеличения скорости классификатора
def macro_norm(collect, collect_M, collect_MM):
   M1 = defaultdict(list)
   M2 = defaultdict(list)
   M3 = defaultdict(list)

   for i in range(1, 9):
      M1['M' + str(i) + '_n'].append(np.array(collect['M' + str(i)])/max(np.abs(collect['M' + str(i)])))

   for i in range(1, 9):
      for j in range(1, 9):
         if i == j:
            M2['M' + str(i) + 'M' + str(j) + '_n'].append(np.array(collect_M['M' + str(i) + 'M' + str(j)])/max(np.abs(collect_M['M' + str(i) + 'M' + str(j)])))
         else:
            continue   

   for i in range(2, 9):
      for j in range(1, i):
         M3['M' + str(i) + 'M' + str(j) + '_n'].append(np.array(collect_MM['M' + str(i) + 'M' + str(j)])/max(np.abs(collect_MM['M' + str(i) + 'M' + str(j)])))
   
   return M1, M2, M3

data = macroparams(data_strat)
data