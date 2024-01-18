import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import Orange

base_credit= Orange.data.Table('C:/Users/joaov/OneDrive/Área de Trabalho/Python/CSV/credit_data_regras.csv')

print(base_credit.domain)

majority= Orange.classification.MajorityLearner()

previsoes=Orange.evaluation.testing.TestOnTestData(base_credit, base_credit, [majority])
print(Orange.evaluation.CA(previsoes))

for registro in base_credit:
    print(registro.get_class())

from collections import Counter
print(Counter(str(registro.get_class()) for registro in base_credit))