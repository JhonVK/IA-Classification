import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/census.pkl', 'rb') as f:
           x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)

from sklearn.neighbors import  KNeighborsClassifier 

print(x_census_teste.shape)
print(y_census_teste.shape)

knn_census = KNeighborsClassifier(n_neighbors=5)# nao precisa por os outros parametros, pois eles sao default, so coloquei no outro codigo para entender 

knn_census.fit(x_census_treinamento, y_census_treinamento)

previsoes= knn_census.predict(x_census_teste)

print(previsoes)
from sklearn.metrics import accuracy_score, classification_report

porcent= accuracy_score(y_census_teste, previsoes)

print(porcent)

print(classification_report(y_census_teste, previsoes))


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(knn_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
plt.show()
