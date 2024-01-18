import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/census.pkl', 'rb') as f:
           x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste=pickle.load(f)


print(np.unique(y_census_treinamento))

arvore_census= DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_census.fit(x_census_treinamento, y_census_treinamento)

previsoes= arvore_census.predict(x_census_teste)
print(previsoes)

print(y_census_teste)

from sklearn.metrics import accuracy_score, classification_report

acuracia= accuracy_score(y_census_teste, previsoes)
print(acuracia) ## da 81% de acertos


from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(arvore_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
plt.show()

print(classification_report(y_census_teste, previsoes))