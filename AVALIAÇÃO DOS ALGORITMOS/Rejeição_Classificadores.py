
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

novo_registro = x_credit[0]
novo_registro = novo_registro.reshape(1, -1)
novo_registro, novo_registro.shape


resposta_rede_neural = neural.predict(novo_registro)
resposta_arvore = arvore.predict(novo_registro)
resposta_svm = svm.predict(novo_registro)

probabilidade_Rede=neural.predict_proba(novo_registro)
print(probabilidade_Rede)
confinca_rede= probabilidade_Rede.max()
print(confinca_rede)

probabilidade_arvore=arvore.predict_proba(novo_registro)
confianca_arvore=probabilidade_arvore.max()
print(confianca_arvore)

probabilidade_svm=svm.predict_proba(novo_registro)
confianca_Svm=probabilidade_svm.max()
print(confianca_Svm)


paga = 0
nao_paga = 0
confianca_minima = 0.999999
algoritmos = 0

if confinca_rede >= confianca_minima:
  algoritmos += 1
  if resposta_rede_neural[0] == 1:
    nao_paga += 1   
  else:
    paga += 1

if confianca_arvore >= confianca_minima:
  algoritmos += 1
  if resposta_arvore[0] == 1:
    nao_paga += 1
  else:
    paga += 1

if confianca_Svm >= confianca_minima:
  algoritmos += 1
  if resposta_svm[0] == 1:
    nao_paga += 1
  else:
    paga += 1

if paga > nao_paga:
  print('Cliente pagará o empréstimo, baseado em {} algoritmos'.format(algoritmos))
elif paga == nao_paga:
  print('Empate, baseado em {} algoritmos'.format(algoritmos))
else:
  print('Cliente não pagará o empréstimo, baseado em {} algoritmos'.format(algoritmos))