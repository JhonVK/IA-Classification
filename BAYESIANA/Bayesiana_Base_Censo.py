import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

import pickle
with open('C:\\Users\\joaov\\OneDrive\\Área de Trabalho\\Python\\census.pkl', 'rb') as f:
  x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)



  from sklearn.naive_bayes import GaussianNB
bayesiana_census=GaussianNB()
bayesiana_census.fit(x_census_treinamento, y_census_treinamento)

bayesiana_previsoes=bayesiana_census.predict(x_census_teste)
print(bayesiana_previsoes)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_census_teste, bayesiana_previsoes))## 47% de acertos (bem ruim)(nao exexutar escalonamento a gente chega a 70%)

from yellowbrick.classifier import ConfusionMatrix
cm=ConfusionMatrix(bayesiana_census)
cm.fit(x_census_treinamento, y_census_treinamento)
print(cm.score(x_census_teste, y_census_teste))
cm.show()

from sklearn.metrics import classification_report
print(classification_report(y_census_teste, bayesiana_previsoes))