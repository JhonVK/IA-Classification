import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


import pickle
with open('C:/Users/joaov/OneDrive/Área de Trabalho/Python/PKLS/credit.pkl', 'rb') as f:
           x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste=pickle.load(f)

##vamos concatenar para ter a base completa

x_credit= np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
print(x_credit.shape)

y_credit= np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)
print(y_credit.shape)

resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_logistica = []
resultados_svm = []
resultados_rede_neural = []

for i in range(30):
    
    kfold= KFold(n_splits=10, random_state=i, shuffle=True)
    arvore= DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    score=cross_val_score(arvore, x_credit, y_credit, cv=kfold)
    print(score)
    print(score.mean())
    print()
    resultados_arvore.append(score.mean())
    
    #random
    random=RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 1, min_samples_split=5, n_estimators = 10)
    score=cross_val_score(random, x_credit, y_credit, cv=kfold)
    print(score)
    print(score.mean())
    print()
    resultados_random_forest.append(score.mean())
    
    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, x_credit, y_credit, cv = kfold)
    resultados_knn.append(scores.mean())

    logistica = LogisticRegression(C = 1.0, solver = 'lbfgs', tol = 0.0001)
    scores = cross_val_score(logistica, x_credit, y_credit, cv = kfold)
    resultados_logistica.append(scores.mean())

    svm = SVC(kernel = 'rbf', C = 2.0)
    scores = cross_val_score(svm, x_credit, y_credit, cv = kfold)
    resultados_svm.append(scores.mean())

    rede_neural = MLPClassifier(activation = 'relu', batch_size = 56, solver = 'adam')
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv = kfold)
    resultados_rede_neural.append(scores.mean())



resultados = pd.DataFrame({'Arvore': resultados_arvore, 'Random forest': resultados_random_forest,
                           'KNN': resultados_knn, 'Logistica': resultados_logistica,
                           'SVM': resultados_svm, 'Rede neural': resultados_rede_neural})
print(resultados)
print(resultados.describe())
(resultados.std() / resultados.mean()) * 100 ##coef de variação


####teste de normalidade 


alpha = 0.05
from scipy.stats import shapiro

print(shapiro(resultados_arvore), shapiro(resultados_random_forest), shapiro(resultados_knn), shapiro(resultados_logistica), shapiro(resultados_svm), shapiro(resultados_rede_neural))

sns.displot(resultados_arvore, kind = 'kde');
plt.show()

sns.displot(resultados_random_forest, kind = 'kde');
plt.show()
sns.displot(resultados_knn, kind = 'kde');
plt.show()
sns.displot(resultados_logistica, kind = 'kde');
plt.show()
sns.displot(resultados_svm, kind = 'kde');
plt.show()
sns.displot(resultados_rede_neural, kind = 'kde');
plt.show()

##teste ANOVA E TUKEY(testa se tem diferença estatistica entre os algoritmos)

from scipy.stats import f_oneway

f, p = f_oneway(resultados_arvore, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural)

alpha = 0.05
if p <= alpha:
  print('Hipótese nula rejeitada. Dados são diferentes')
else:
  print('Hipótese alternativa rejeitada. Resultados são iguais')

##teste tukey(testar melhor algoritmo)
  
resultados_algoritmos= {'accuracy': np.concatenate([resultados_arvore, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural]),
                        
                       'algoritmo': ['arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore',
                          'random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest',
                          'knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn',
                          'logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica',
                          'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
                          'rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural']}

resultados_df = pd.DataFrame(resultados_algoritmos)
print(resultados_df)

from statsmodels.stats.multicomp import MultiComparison

compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])

teste_estatistico = compara_algoritmos.tukeyhsd()
print(teste_estatistico)

print(resultados.mean())

teste_estatistico.plot_simultaneous();
plt.show()