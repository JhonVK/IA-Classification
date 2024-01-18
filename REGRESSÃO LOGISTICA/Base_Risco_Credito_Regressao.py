import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LogisticRegression

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/risco_credito.pkl', 'rb') as f:
           x_risco_credito, y_risco_credito=pickle.load(f)
print(x_risco_credito)
print()
print(y_risco_credito)

#apagar os 'moderados' ( 2, 7, 11)

x_risco_credito=np.delete(x_risco_credito, [2, 7, 11], axis=0)
y_risco_credito=np.delete(y_risco_credito, [2, 7, 11], axis=0)
print(x_risco_credito)
print()
print(y_risco_credito)


logistic_risco_credito= LogisticRegression(random_state = 1)
logistic_risco_credito.fit(x_risco_credito, y_risco_credito)

print(logistic_risco_credito.intercept_)

print(logistic_risco_credito.coef_)

previsoes1= logistic_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
print(previsoes1)