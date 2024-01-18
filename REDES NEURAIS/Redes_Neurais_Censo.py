import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.neural_network import MLPClassifier


import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/census.pkl', 'rb') as f:
           x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)


neural_census=MLPClassifier(hidden_layer_sizes=(51,51), max_iter=500, verbose=True, tol=0.000001, solver='adam', activation='relu')

neural_census.fit(x_census_treinamento, y_census_treinamento)

previsoes=neural_census.predict(x_census_teste)


from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_census_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(neural_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
plt.show()

print(classification_report(y_census_teste, previsoes))