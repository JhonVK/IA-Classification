import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.svm import SVC


import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)

print(x_credit_teste.shape)
print(y_credit_teste.shape)

print(x_credit_treinamento.shape)
print(y_credit_treinamento.shape)

svm_Credit= SVC(kernel= 'rbf', random_state=1, C=2.0)##rbf foi o melhor kernel nesse caso

svm_Credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes= svm_Credit.predict(x_credit_teste)
print(previsoes)

from sklearn.metrics import accuracy_score, classification_report

porcent= accuracy_score(y_credit_teste, previsoes)

print(porcent)


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(svm_Credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()

print(classification_report(y_credit_teste, previsoes))