import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/census.pkl', 'rb') as f:
           x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)

print(x_census_treinamento.shape, y_census_treinamento.shape)
print(x_census_teste.shape, y_census_teste.shape)

random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

random_forest.fit(x_census_treinamento, y_census_treinamento)

previsoes = random_forest.predict(x_census_teste)
print(previsoes)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_census_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
plt.show()

print(classification_report(y_census_teste, previsoes))