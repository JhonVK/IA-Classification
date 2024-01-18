import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
base_credit= pd.read_csv(r'C:\Users\joaov\Downloads\credit_data.csv')
##precisores==x
##classe== y

x_credit=base_credit.iloc[:, 1:4].values ###primeira coluna é linha (: é todas as linhas), o segundo é a coluna(1:4 (1 até 3(income age e loan))(.values converte para o valor q o numpy usa))
print(x_credit)
type(x_credit)

print()

y_credit=base_credit.iloc[:, 4].values
print(y_credit)
type(y_credit)


print(x_credit[:, 0])##todas as rendas
print(x_credit[:, 0].min())#menor renda
print(x_credit[:, 0].max())#maior renda

print(x_credit[:,1].min())#menor idade(Deu erro pois o age aqui neste codigo nao foi tratado)
print(x_credit[:,1].max())#maior idade(Deu erro pois o age aqui neste codigo nao foi tratado)

##valores na mesma escala:
from sklearn.preprocessing import StandardScaler
scaler_credit= StandardScaler()
x_credit=scaler_credit.fit_transform(x_credit)
#já escalonados:
print(x_credit[:, 0].min())#menor renda
print(x_credit[:, 0].max())#maior renda

