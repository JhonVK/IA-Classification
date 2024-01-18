import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

import pickle
with open('C:\\Users\\joaov\\OneDrive\\Área de Trabalho\\Python\\credit.pkl', 'rb') as f:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)


print(x_credit_treinamento)
from sklearn.naive_bayes import GaussianNB
bayesiana_credito=GaussianNB()
bayesiana_credito.fit(x_credit_treinamento, y_credit_treinamento.astype('int'))


bayesiana_previsoes= bayesiana_credito.predict(x_credit_teste)  
print(bayesiana_previsoes)##respostas do algoritmo
print(y_credit_teste)##respostas finais para comparar com a de cima
##para comparar automaticamente accuracy_score
from sklearn.metrics import accuracy_score

print(accuracy_score(y_credit_teste, bayesiana_previsoes))## 93% de acertos 

##visualização:

from yellowbrick.classifier import ConfusionMatrix
cm=ConfusionMatrix(bayesiana_credito)
cm.fit(x_credit_treinamento, y_credit_treinamento)
print(cm.score(x_credit_teste, y_credit_teste))
cm.show()
from sklearn.metrics import classification_report
print(classification_report(y_credit_teste, bayesiana_previsoes))
