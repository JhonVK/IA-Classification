
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

classificadorneural= MLPClassifier(activation='relu', batch_size= 56, solver='adam')
classificadorneural.fit(x_credit, y_credit)



classificadorarvore=DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
classificadorarvore.fit(x_credit, y_credit)


classificadorsvm=SVC(C=2.0, kernel='rbf', probability=True)
classificadorsvm.fit(x_credit, y_credit)


# Salvando o modelo de rede neural
with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\CLASSIFICADORES TREINADOS\Rede_neural_finalizado.sav', 'wb') as file:
    pickle.dump(classificadorneural, file)

# Salvando o modelo de árvore de decisão
with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\CLASSIFICADORES TREINADOS\Arvore_finalizado.sav', 'wb') as file:
    pickle.dump(classificadorarvore, file)

# Salvando o modelo de SVM
with open(r'C:\Users\joaov\OneDrive\Área de Trabalho\Python\CLASSIFICADORES TREINADOS\SVM_finalizado.sav', 'wb') as file:
    pickle.dump(classificadorsvm, file)
