import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import Orange

base_census= Orange.data.Table('C:/Users/joaov/OneDrive/Área de Trabalho/Python/CSV/census_regras.csv')

print(base_census.domain)

majority= Orange.classification.MajorityLearner()

previsoes=Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])
print(Orange.evaluation.CA(previsoes))


from collections import Counter
print(Counter(str(registro.get_class()) for registro in base_census))