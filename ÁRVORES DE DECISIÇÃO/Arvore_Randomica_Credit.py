import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)

print(x_credit_treinamento.shape, y_credit_treinamento.shape)
print(x_credit_teste.shape, y_credit_teste.shape)

random_forest = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

random_forest.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = random_forest.predict(x_credit_teste)
print(previsoes)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_credit_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()

print(classification_report(y_credit_teste, previsoes))