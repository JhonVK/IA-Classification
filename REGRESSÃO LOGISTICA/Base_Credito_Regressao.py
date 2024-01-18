import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LogisticRegression

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)


logistic_credit=LogisticRegression(random_state=1)
logistic_credit.fit(x_credit_treinamento, y_credit_treinamento)

print(logistic_credit.intercept_)

print(logistic_credit.coef_)


previsoes1= logistic_credit.predict(x_credit_teste)
print(previsoes1)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_credit_teste, previsoes1))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(logistic_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()

print(classification_report(y_credit_teste, previsoes1))


