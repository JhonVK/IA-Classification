import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)

from sklearn.neighbors import  KNeighborsClassifier 

knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)

knn_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes= knn_credit.predict(x_credit_teste)

print(previsoes)

print(y_credit_teste)

from sklearn.metrics import accuracy_score, classification_report

porcent= accuracy_score(y_credit_teste, previsoes)

print(porcent)

print(classification_report(y_credit_teste, previsoes))


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(knn_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()
