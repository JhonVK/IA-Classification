import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

base_credit= pd.read_csv(r'C:/Users/joaov/OneDrive/Área de Trabalho/Python/CSV/credit_data.csv')
base_credit['age'][base_credit['age']<0]= base_credit['age'][base_credit['age']>0].mean()
print(base_credit)
print(base_credit.isnull().sum())


base_credit['age'].fillna((base_credit['age']).mean(), inplace = True)

x_credit=base_credit.iloc[:, 1:4].values

y_credit=base_credit.iloc[:, 4].values


from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()
x_credit = scaler_credit.fit_transform(x_credit)


grafico=px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')


from sklearn.model_selection import train_test_split

x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit, y_credit, test_size = 0.25, random_state = 0)
print(x_credit_treinamento.shape)##1500 registros(linhas)3 colunas serao enviados para os algoritmos apreenderem 75% da base total
print(y_credit_treinamento.shape)
print(x_credit_teste.shape, y_credit_teste.shape)##500 linhas serao usadas no teste

print(np.unique(y_credit_treinamento))

import pickle
with open(r'C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', mode = 'wb') as f:
  pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)



  
