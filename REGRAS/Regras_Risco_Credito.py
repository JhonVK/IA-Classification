import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import Orange

base_risco_credito = Orange.data.Table('C:/Users/joaov/OneDrive/Área de Trabalho/Python/CSV/risco_credito_regras.csv')

print(base_risco_credito)
print()
print(base_risco_credito.domain)

cn2 = Orange.classification.rules.CN2Learner()

regras_risco_credito = cn2(base_risco_credito)

for regras in regras_risco_credito.rule_list:
  print(regras)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
print(previsoes)

print(base_risco_credito.domain.class_var.values)


for t in previsoes: 
    print(base_risco_credito.domain.class_var.values[t])
