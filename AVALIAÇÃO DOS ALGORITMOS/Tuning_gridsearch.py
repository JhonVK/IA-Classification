
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import numpy as np


import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)



print(x_credit_treinamento.shape, y_credit_treinamento.shape)
print(x_credit_teste.shape, y_credit_teste.shape)
##vamos concatenar para ter a base completa

x_credit= np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
print(x_credit.shape)

y_credit= np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)
print(y_credit.shape)


##arvore de decisao
parametros = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search= GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)
grid_search.fit(x_credit, y_credit)

melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_

print(melhores_parametros)
print(melhor_resultado)

##Random forests

parametros={'criterion': ['gini', 'entropy'],
            'n_estimators': [10, 40, 100, 150],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search= GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(x_credit, y_credit)

melhores_parametros=grid_search.best_params_
melhor_resultados= grid_search.best_score_

print(melhores_parametros)
print(melhor_resultado)

##KNN

parametros={'n_neighbors': [3, 5, 10, 20],
            'p': [1, 2]
        
}

grid_search= GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(x_credit, y_credit)

melhores_parametros=grid_search.best_params_
melhor_resultados= grid_search.best_score_

print(melhores_parametros)
print(melhor_resultado)
#regressao logistica 

parametros = {'tol': [0.0001, 0.00001, 0.000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga']}
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=parametros)
grid_search.fit(x_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)


##SVM

parametros={'C': [1.0, 2.0, 5.0, 10.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5, 1, 2, 10],
            'gamma': ['scale', 'auto']

}
grid_search= GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(x_credit, y_credit)

melhores_parametros=grid_search.best_params_
melhor_resultados= grid_search.best_score_

print(melhores_parametros)
print(melhor_resultados)

##redes neurais

parametros={'activation':['relu', 'identity', 'logistic', 'tanh'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'batch_size': [10, 56]}

grid_search=GridSearchCV(MLPClassifier(), param_grid=parametros)
grid_search.fit(x_credit, y_credit)

melhores= grid_search.best_score_
melhorespara=grid_search.best_params_

print(melhores)
print(melhorespara)
