import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import Orange

base_credit= Orange.data.Table('C:/Users/joaov/OneDrive/Área de Trabalho/Python/CSV/credit_data_regras.csv')

print(base_credit.domain)

base_dividida = Orange.evaluation.testing.sample(base_credit, n = 0.25)

base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

print(len(base_treinamento), len(base_teste))


cn2=Orange.classification.rules.CN2Learner()
regras_credit = cn2(base_treinamento)

for regras in regras_credit.rule_list:
    print(regras)

previsoes= Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])
print(previsoes)

print(Orange.evaluation.CA(previsoes))