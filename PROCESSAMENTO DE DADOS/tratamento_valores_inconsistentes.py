import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
base_credit= pd.read_csv(r'C:\Users\joaov\Downloads\credit_data.csv')

print(base_credit.loc[base_credit['age']<0])
base_credit2= base_credit.drop('age', axis=1)#base_credit2 recebe p base_credit sem a coluna age
print(base_credit2)

base_credit3= base_credit.drop(base_credit[base_credit['age']<0].index)#apaga apenas as linhas com age<0
print(base_credit3)


#preencher os valores inconsistentes manualmente é outra forma de fazer isso...
#preencher valores com a média:
print(base_credit.mean())#entretando aqui esta fazendo a media com os valores errados tambem...
print(base_credit['age'].mean())#media com valores errado tbm...

print(base_credit['age'][base_credit['age']>0].mean())##aqui fez a media sem valores inconsistentes

base_credit.loc[base_credit['age']<0, 'age'] = 40.92##aqui os valores errados de age recebem o valor da media

print(base_credit[base_credit['age']<0])##aqui prova que valores inconsistentes nao existem mais(mudou para 40.92)

print(base_credit.head(27))

grafico=px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()
