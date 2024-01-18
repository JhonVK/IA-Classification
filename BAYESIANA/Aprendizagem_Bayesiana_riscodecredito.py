import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

base_risco_credito= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\risco_credito.csv')
print(base_risco_credito)

x_risco=base_risco_credito.iloc[:, 0:4].values ##(0 ate 3)
y_risco=base_risco_credito.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
label_historia=LabelEncoder()
label_divida=LabelEncoder()
label_garantia=LabelEncoder()
label_renda=LabelEncoder()

x_risco[:, 0]=label_historia.fit_transform(x_risco[:, 0])
x_risco[:, 1]=label_divida.fit_transform(x_risco[:, 1])
x_risco[:, 2]=label_garantia.fit_transform(x_risco[:, 2])
x_risco[:, 3]=label_renda.fit_transform(x_risco[:, 3])

print(x_risco)

import pickle
with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\PKLS\risco_credito.pkl', mode='wb') as f:
  pickle.dump([x_risco, y_risco], f)

from sklearn.naive_bayes import GaussianNB

naive_risco=GaussianNB()
naive_risco.fit(x_risco, y_risco)
print(naive_risco.classes_)##mostra as classes
##testar algo
# história boa (0), dívida alta (0), garantias nenhuma (1), renda > 35 (2)
# história ruim (2), dívida alta (0), garantias adequada (0), renda < 15 (0)
#valores de cada classe sao observaveis no terminal print(x_risco)
previsao= naive_risco.predict([[0,0,1,2], [2,0,0,0]])
print(previsao)