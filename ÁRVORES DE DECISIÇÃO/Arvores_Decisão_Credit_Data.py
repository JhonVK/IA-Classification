import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)


print(np.unique(y_credit_treinamento))

arvore_credit= DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes= arvore_credit.predict(x_credit_teste)
print(previsoes)

print(y_credit_teste)# comparando (varios acertos)

from sklearn.metrics import accuracy_score, classification_report

acuracia= accuracy_score(y_credit_teste, previsoes)
print(acuracia) ## da 98.2% de acertos



from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(arvore_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()



print(classification_report(y_credit_teste, previsoes))



from sklearn import tree
previsores = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20))
tree.plot_tree(arvore_credit, feature_names=previsores, class_names=['0','1'], filled=True);
fig.savefig('arvore_credit.png')

plt.show()
