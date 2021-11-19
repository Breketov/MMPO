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

C, C0, sigma1, sigma2=120,70,1,1
args=[C,C0,sigma1,sigma2]
N=40
alpha_j=0.0016
alpha_a=0.006
beta_j=0.0000007
beta_a=0.000000075
delta_j=0.000016
delta_a=0.00006
gamma_j=0.00008
gamma_a=0.004

lamb=[alpha_j,alpha_a,beta_j,beta_a,delta_j,delta_a,gamma_j,gamma_a]

def getData ():
    Aj=np.empty(N,dtype=float)
    Bj=np.empty(N,dtype=float)
    Aa=np.empty(N,dtype=float)
    Ba=np.empty(N,dtype=float)

    M1=np.empty(N,dtype=float)
    M2=np.empty(N,dtype=float)
    M3=np.empty(N,dtype=float)
    M4=np.empty(N,dtype=float)
    M5=np.empty(N,dtype=float)
    M6=np.empty(N,dtype=float)
    M7=np.empty(N,dtype=float)
    M8=np.empty(N,dtype=float)
    

    for i in range(0,N):
        Aj[i]=random.uniform(-C, 0)
        Bj[i]=random.uniform(-min(Aj[i]+C,-1*Aj[i]), min(Aj[i]+C,-1*Aj[i]))
        Aa[i]=random.uniform(-C, 0)
        Ba[i]=random.uniform(-min(Aa[i]+C,-1*Aa[i]), min(Aa[i]+C,-1*Aa[i]))

        M1[i]=(sigma1*(Aj[i]+C))
        M2[i]=-(sigma2*(Aj[i]+C+Bj[i]/2))
        M3[i]=-((2*pi)**2)*(Bj[i]**2)/2
        M4[i]=-((Aj[i]+C0)**2+(Bj[i]**2)/2)
        M5[i]=(sigma1*(Aa[i] + C))
        M6[i]=-(sigma2*(Aa[i] + C + Ba[i]/2))
        M7[i]=-((2*pi)**2)*(Ba[i]**2)/2
        M8[i]=-((Aa[i] + C0)**2 + (Ba[i]**2)/2)


    cols=['A_j','B_j','A_a','B_a','M1','M2','M3','M4','M5','M6','M7','M8']
    data = pd.DataFrame({
        cols[0]:Aj,cols[1]:Bj,
        cols[2]:Aa,cols[3]:Ba,
        cols[4]:M1,cols[5]:M2,cols[6]:M3,cols[7]:M4,
        cols[8]:M5,cols[9]:M6,cols[10]:M7,cols[11]:M8
    })
    data_ = pd.DataFrame(data =  data )
    data_.to_csv("data_parameters.csv", index=False)
    return data

ndata=getData()
ndata

def get_strategyParam ():
    data_strategyParam=getData().drop(['A_j','B_j','A_a','B_a'], axis=1)
    scaler = preprocessing.MinMaxScaler()
    norm_data = scaler.fit_transform(data_strategyParam)
    names=data_strategyParam.columns
    # cols=['M1','M2','M3','M4','M5','M6','M7','M8']
    data_strategyParam=pd.DataFrame(norm_data,columns=names)
    return data_strategyParam

get_strategyParam ()

data=get_strategyParam ()
data.loc[1]


def fitness(v):
    return np.dot(v, lamb)

def get_target(v,w):
    if fitness(v)-fitness(w)>0:
        return 1
    return -1



def get_Sample():
    data_arr=get_strategyParam().to_numpy()
    train_data_arr = []
    for i in range(0,len(data_arr)):
        for j in range(i+1,len(data_arr)):
            v=np.empty(8,dtype=float)
            w=np.empty(8,dtype=float)
            # v=(data.iloc[i, :]).to_numpy()
            # w=(data.iloc[j, :]).to_numpy()
            v=data_arr[i,:]
            w=data_arr[j,:]
            str_=[*(v-w), get_target(v,w)]   #должен быть M1,M2,M3,M4,{-1,1}
            train_data_arr.append(str_)
    trainSample=pd.DataFrame(columns=['M1', 'M2', 'M3', 'M4','M5','M6','M7','M8', 'target'], data=train_data_arr)
    data = pd.DataFrame(data =  trainSample )
    data.to_csv("Sample.csv", index=False)
    return trainSample

get_Sample()



import matplotlib.pyplot as plt
%matplotlib inline

X1 = X[y==1]
X0 = X[y==-1]

for i in range(len(X.columns)):
    for j in range(i + 1, len(X.columns)):
        x=np.linspace(-1, 1)
        plt.figure(figsize=(7, 7))
        plt.scatter(x = X1[X.columns[i]], y=X1[X.columns[j]], marker='.')
        plt.scatter(x = X0[X.columns[i]], y=X0[X.columns[j]], marker='x')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.legend()
        plt.xlabel(X.columns[i])
        plt.ylabel(X.columns[j])
        plt.grid()
        plt.show()


X=get_Sample().loc[:,'M1':'M8']
y=get_Sample().loc[:,'target']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)  
X_train, y_train

X_train_SVC, y_train_SVC=X_train, y_train
X_train_SVC, y_train_SVC

X_train_LDA, y_train_LDA=X_train, y_train
X_train_LDA, y_train_LDA

X_test_SVC, y_test_SVC=X_test, y_test
X_test_SVC, y_test_SVC

X_test_LDA, y_test_LDA=X_test, y_test
X_test_LDA, y_test_LDA

from sklearn.svm import SVC 
#экземпляр классификатора
svc_model = SVC(kernel = 'linear')
#обучения классификатора
svc_model.fit(X_train_SVC, y_train_SVC)
SVC_prediction = svc_model.predict(X_test_SVC) 
SVC_prediction

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 10))


sns.scatterplot(x=X_train_SVC.iloc[:,0], 
                y=X_train_SVC.iloc[:,1], 
                hue=y_train_SVC, 
                s=8)

w = svc_model.coef_[0]
b = svc_model.intercept_[0]
x_points = np.linspace(-1, 1)   
y_points = -(w[0] / w[1]) * x_points - b / w[1]  
plt.plot(x_points, y_points, c='r')

lamb_method=svc_model.coef_[0]
#lamb=[alpha_j,alpha_a,beta_j,beta_a,delta_j,delta_a,gamma_j,gamma_a]

def calc_A_j(lamb):
    return (sigma1*lamb[0]-sigma2*lamb[6])/(2*lamb[4])-C0

def calc_B_j(lamb):
    return -sigma2*lamb[6]/(2*lamb[4]+2*(2*pi)**2*lamb[2]) 

def calc_A_a(lamb):
    return (sigma1*lamb[1]-sigma2*lamb[7])/(2*lamb[5])-C0

def calc_B_a(lamb):
    return -sigma2*lamb[7]/(2*lamb[5]+2*(2*pi)**2*lamb[3])  

A_j_method, B_j_method= calc_A_j(lamb_method),calc_B_j(lamb_method)
A_a_method, B_a_method= calc_A_a(lamb_method),calc_B_a(lamb_method)
A_j, B_j= calc_A_j(lamb),calc_B_j(lamb)
A_a, B_a= calc_A_a(lamb),calc_B_a(lamb)

print('Вычисленные стратегий A и B по коэффициентам, полученным методом опорных векторов:\n', A_j_method, B_j_method, A_a_method, B_a_method)
print('Вычисленные стратегий A и B по коэффициентам, которые были даны:\n', A_j, B_j, A_a, B_a)

t=np.linspace(0,1)
x_j_method=A_j_method+B_j_method*np.cos(2*pi*t)
x_a_method=A_a_method+B_a_method*np.cos(2*pi*t)
x_j=A_j+B_j*np.cos(2*pi*t)
x_a=A_a+B_a*np.cos(2*pi*t)
plt.xlabel("t") 
plt.ylabel("x")
plt.grid()  
plt.plot(t,x_j_method,label='молодые особи(метод)',color='b')
plt.plot(t,x_a_method,label='взрослые особи(метод)',color='r')
plt.plot(t,x_j,label='молодые особи(данные)', color='g')
plt.plot(t,x_a,label='взрослые особи(данные)',color='m')
plt.legend()
plt.show()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_LDA = sc.fit_transform(X_train_LDA)
X_test_LDA = sc.transform(X_test_LDA)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

lda_model = LDA(n_components=1)
X_train_LDA = lda_model.fit_transform(X_train_LDA, y_train_LDA)
X_test_LDA = lda_model.transform(X_test_LDA)
LDA_prediction= lda_model.predict(X_test_LDA)
LDA_prediction