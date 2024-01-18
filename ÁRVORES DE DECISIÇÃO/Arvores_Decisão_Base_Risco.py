import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/risco_credito.pkl', 'rb') as f:
          x_risco_credito, y_risco_credito = pickle.load(f)

print(x_risco_credito)
print(y_risco_credito)

arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(x_risco_credito, y_risco_credito)##treinamento

print(arvore_risco_credito.feature_importances_)##retorna a importancia de cada atributo, (por ordem)

from sklearn import tree
class_names= arvore_risco_credito.classes_.tolist()
previsores= ['história', 'dívida', 'garantias', 'renda']
figura, axes= plt.subplots(nrows=1, ncols=1, figsize=(10,10))
print(tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names= class_names, filled=True))##arvore
plt.show()

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
previsoes = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes)

