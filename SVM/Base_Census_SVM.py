import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.svm import SVC

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/census.pkl', 'rb') as f:
           x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)

census_svm= SVC(kernel= 'linear', random_state=1, C=2.0)
census_svm.fit(x_census_treinamento, y_census_treinamento)

previsoes= census_svm.predict(x_census_teste)
print(previsoes)

from sklearn.metrics import accuracy_score, classification_report

print(y_census_teste)

from sklearn.metrics import accuracy_score, classification_report

porcent= accuracy_score(y_census_teste, previsoes)

print(porcent)

print(classification_report(y_census_teste, previsoes))


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(census_svm)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
plt.show()
