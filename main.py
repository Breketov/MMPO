import csv
import random as rn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" def rand_AB(Nmax):
   A_ = []
   B_ = []
   for i in range(0, Nmax):
      A = round(rn.uniform(-D, 0), 3)
      B = round(rn.uniform(-min(A + D, -A), min(A + D, -A)), 3)
      A_.append(A)
      B_.append(B)
   return A_, B_


def strategy(t, A, B):
   x_yng = []
   for i in range(0, Nmax):
      x_yng_ = A[i] + B[i]*np.cos(2*np.pi*t)
      x_yng.append(x_yng_)
   return x_yng

Nmax = 20
D = 120
D0 = 70  
sigma1 = 0.25
sigma2 = 0.0003 """



"""
xy = []
M1 = []
M2 = []
M3 = []
M4 = []
for i in range(0, Nmax):
   A_ = round(rn.uniform(-D, 0), 3)
   B_ = round(rn.uniform(-min(A_ + D, -A_), min(A_ + D, -A_)), 3)
   A.append(A_)
   B.append(B_)

t = np.linspace(0, 1, Nmax)
for i in range(0, len(A)):
   xy_ = A[i] + B[i]*np.cos(2*np.pi*t)
   xy.append(xy_)

d = {'A': A, 'B': B}
data_A_B = pd.DataFrame(data = d)

data_A_B.to_csv("data_A_B.csv", index=False, sep="\t")

for i in range(0, len(A)):
   M1_ = round(sigma1*(A[i] + D), 3)
   M2_ = round(-sigma2*(A[i] + D + B[i]/2), 3)
   M3_ = round(-2*(np.pi*B[i])**2, 3)
   M4_ = round(-((A[i] + D0)**2 + (B[i]**2)/2), 3)
   M1.append(M1_)
   M2.append(M2_)
   M3.append(M3_)
   M4.append(M4_)

e = {'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4}
data_M = pd.DataFrame(data = e)

data_M.to_csv("data_M.csv", index=False, sep="\t")
z = -112.289+ 5.17*np.cos(2*np.pi*t)
plt.plot(t, z)
plt.show()
"""
""" 
AB = rand_AB(Nmax)
t = np.linspace(0, 1, Nmax)
x = strategy(t, AB[0], AB[1]) """

""" 
data_AB = pd.read_csv('data_AB.csv')

Ay = data_AB["A_yng"].tolist()
By = data_AB["B_yng"].tolist()
print(Ay)
print(By) """

alpha_y = 0.0016
alpha_o = 0.006
betta_y = 0.0000007
betta_o = 0.000000075
gamma_y = 0.00008
gamma_o = 0.004
delta_y = 0.000016
delta_o = 0.00006

D = 120
D0 = 70  
sigma1 = 1
sigma2 = 1
Nmax = 1000


""" 
data_M = pd.read_csv('data_M.csv')
M1 = data_M['M1'].tolist()
M2 = data_M['M2'].tolist()
M3 = data_M['M3'].tolist()
M4 = data_M['M4'].tolist()
M5 = data_M['M5'].tolist()
M6 = data_M['M6'].tolist()
M7 = data_M['M7'].tolist()
M8 = data_M['M8'].tolist()
J = []
for i in range(0, 10):
   r = 0
   s = 0
   p = 0
   q = 0
   r = alpha_o*M5[i] + betta_o*M7[i] + delta_o*M8[i]
   s = gamma_o*M6[i]
   p = alpha_y*M1[i] + betta_y*M3[i] + delta_y*M4[i]
   q = gamma_y*M2[i]
   j_ =  -s - p - q + np.sqrt(4*r*p + (p + q - s)**2)
   J.append(j_)

print(J.sort())
 """

""" 
sigm1 = 0.25
sigm2 = 0.003
C0 = 60
C = 140
#lam = [0.1 / sigm1, 1 / sigm2, 0.000025, 0.1]
#lam = [0.5 / sigm1, 1 / sigm2, 0.000025, 0.01]
lam = [0.5, 1, 0.000025, 0.01]

from random import randint
import pandas as pd

def gen_AB():
    A = randint(-C * 100, 0)
    B1 = randint(0, min(-A, A + C * 100))
    B2 = -B1
    return A / 100, B1 / 100, B2 / 100 

def get_strategy_of_behavior(N):
    arr = []
    for i in range(N // 2):
        A, B1, B2 = gen_AB()
        while (A, B1) in arr:
            A, B1, B2 = gen_AB()
        arr.append((A, B1))
        arr.append((A, B2))
    data = pd.DataFrame(columns=['A', 'B'], data = arr)
    data['C'] = C
    data['C0'] = C0
    return data

def get_strategy_of_behavior_1(N):
    n = int(N ** 0.5)
    arr = []
    for i in range(n):
        for j in range(n):
            A = -C + C * i / n
            B1 =  min(A + C, -A) * j / n
            B2 = -B1
            if (A, B1) not in arr:
                arr.append((A, B1))
            if (A, B2) not in arr:
                arr.append((A, B2))
    data = pd.DataFrame(columns=['A', 'B'], data = arr)
    data['C'] = C
    data['C0'] = C0
    return data
        
strat_b = get_strategy_of_behavior_1(30)
print(strat_b)  

 """



""" 

#_______________________________________________________________________________________________________________________-
# Тут типо дискриминант ФИШЕРА
# https://russianblogs.com/article/94131183549/

# Тут тоже есть хорошие данные которые больше похожи на то что нахывается дискриминантом ФИШЕРА
# https://question-it.com/questions/479815/linejnyj-diskriminant-fishera-v-python

# Тут реализован код из 1 ссылки
def LDA(x, y):      # x: all the input vector   y: labels
   x_1 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
   x_2 = np.array([x[i] for i in range(len(x)) if y[i] == -1])

   mju1 = np.mean(x_1, axis=0)     # mean vector
   mju2 = np.mean(x_2, axis=0)

   sw1 = np.dot((x_1 - mju1).T, (x_1 - mju1))    # Within-class scatter matrix
   sw2 = np.dot((x_2 - mju2).T, (x_2 - mju2))
   sw = sw1 + sw2

   return np.dot(np.linalg.inv(sw), (mju1 - mju2))

mean_1 = (-5, 0)   
mean_2 = (5, 0)
cov = [[1, 0],    
       [0, 1]]
size = 200          

np.random.seed(1)
x_1 = np.random.multivariate_normal(mean_1, cov, size)

np.random.seed(2)
x_2 = np.random.multivariate_normal(mean_2, cov, size)

x = np.vstack((x_1, x_2))
y = [-1] * 200 + [1] * 200      

plt.scatter(x_1[:, 0], x_1[:, 1], color='blue', marker='o', label='Positive')
plt.scatter(x_2[:, 0], x_2[:, 1], color='red', marker='x', label='Negative')
plt.legend(loc='upper left')
plt.title('Original Data')


w = LDA(x, y)

x1 = 1
y1 = -1 / w[1] * (w[0] * x1)

x2 = -1
y2 = -1 / w[1] * (w[0] * x2)

plt.plot([x1, x2], [y1, y2], 'r')
plt.show()

print(w)


 """