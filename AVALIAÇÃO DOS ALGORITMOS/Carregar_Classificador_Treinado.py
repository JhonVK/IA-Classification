
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np


import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)



x_credit= np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
print(x_credit.shape)

y_credit= np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)
print(y_credit.shape)



with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\CLASSIFICADORES TREINADOS\Rede_neural_finalizado.sav', 'rb') as file:
    neural=pickle.load(file)

with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\CLASSIFICADORES TREINADOS\SVM_finalizado.sav', 'rb') as file:
    svm=pickle.load(file)

with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\CLASSIFICADORES TREINADOS\Arvore_finalizado.sav', 'rb') as file:
    arvore=pickle.load(file)


novo_reg=x_credit[1999]
novo_reg = novo_reg.reshape(1, -1)
print(neural.predict(novo_reg))


print(svm.predict(novo_reg))