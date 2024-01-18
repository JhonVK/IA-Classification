import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LogisticRegression

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/census.pkl', 'rb') as f:
           x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)

print(x_census_treinamento.shape, y_census_treinamento.shape)
print(x_census_teste.shape, y_census_teste.shape)

regressao_census=LogisticRegression(random_state=1)

regressao_census.fit(x_census_treinamento, y_census_treinamento)

previsoes=regressao_census.predict(x_census_teste)
print(previsoes)
print(y_census_teste) ##errou o ultimo 

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_census_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(regressao_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
plt.show()

print(classification_report(y_census_teste, previsoes))
