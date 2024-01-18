import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.neural_network import MLPClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/I.A-Classificação/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)

rede_neural_credit= MLPClassifier(max_iter=2000, verbose=True, 
                                  tol=0.00001, solver='adam', activation='relu', 
                                  hidden_layer_sizes=(2,2)) ##(3+1)/2)
rede_neural_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes=rede_neural_credit.predict(x_credit_teste)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_credit_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(rede_neural_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()

print(classification_report(y_credit_teste, previsoes))


