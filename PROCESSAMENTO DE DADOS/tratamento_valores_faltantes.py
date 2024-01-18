import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
base_credit= pd.read_csv(r'C:\Users\joaov\Downloads\credit_data.csv')


print(base_credit.isnull().sum())##mostra valores faltantes (nao preenchido), é um comando do panda


print(base_credit.loc[pd.isnull(base_credit['age'])])##mostra os registros de valor nulo do age

#agora vamos preencher os valores nulos:

base_credit['age'].fillna(base_credit['age'][base_credit['age']>0].mean(), inplace= True)##aqui eu substitui os valores pela media do age>0 (coloquei maior do que zero(diferente da aula)pois não estou reutilizando o banco de dados anterior)

print(base_credit.loc[(base_credit['clientid']==29)  | (base_credit['clientid']==31) | (base_credit['clientid']==32)])
print(base_credit.loc[base_credit['clientid'].isin ([29,31,32])]) ## esse e o de cima fazem a mesma coisa


