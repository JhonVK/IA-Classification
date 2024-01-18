import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

base_census= pd.read_csv(r'C:\Users\joaov\Downloads\census.csv')


x_census=base_census.iloc[:, 0:14].values##ate native country
print(base_census.columns)
print(x_census)
y_census=base_census.iloc[:, 14]##apenas income 
print(y_census)

from sklearn.preprocessing import LabelEncoder ##  transformando strings em numeros
label_encoder_workclass= LabelEncoder()
label_encoder_education= LabelEncoder()
label_encoder_marital= LabelEncoder()
label_encoder_occupation= LabelEncoder()
label_encoder_relationship= LabelEncoder()
label_encoder_race= LabelEncoder()
label_encoder_sex= LabelEncoder()
label_encoder_country= LabelEncoder()

x_census[:,1] = label_encoder_workclass.fit_transform(x_census[:,1])
x_census[:,3] = label_encoder_education.fit_transform(x_census[:,3])
x_census[:,5] = label_encoder_marital.fit_transform(x_census[:,5])
x_census[:,6] = label_encoder_occupation.fit_transform(x_census[:,6])
x_census[:,7] = label_encoder_relationship.fit_transform(x_census[:,7])
x_census[:,8] = label_encoder_race.fit_transform(x_census[:,8])
x_census[:,9] = label_encoder_sex.fit_transform(x_census[:,9])
x_census[:,13] = label_encoder_country.fit_transform(x_census[:,13])

print(x_census)

##entretando usando apenas o label encoder os algoritmos de IA iria categorizar errado
##temos q usar o onehotencoder(vai dividir uma coluna, exemplo workclass, em varias colunas(privado, local-gov, federal gov... (todas classes de trabalho)))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

OneHotEncoder_census= ColumnTransformer(transformers=[('onehot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
x_census=OneHotEncoder_census.fit_transform(x_census).toarray()
print(x_census)
print(x_census.shape)

##escalonando:
from sklearn.preprocessing import StandardScaler
scaler_credit= StandardScaler()
x_census=scaler_credit.fit_transform(x_census)
print(x_census)

##divsao bases de treinamento e teste

from sklearn.model_selection import train_test_split

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste= train_test_split(x_census, y_census, test_size=0.15, random_state=0)
print(x_census_treinamento.shape) 
print(y_census_treinamento.shape)
print(x_census_teste.shape, y_census_teste.shape)##4885 linhas e 108 colunas para usar no teste

#salvar base
import pickle
with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\PKLS\census.pkl', mode = 'wb') as f:
  pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)
