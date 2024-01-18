import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
base_credit= pd.read_csv(r'C:\Users\joaov\Downloads\credit_data.csv')

print(base_credit)
print()
print(base_credit.describe())
print()
print(base_credit[base_credit['income']>=69995.685578])
print()
print()
print(base_credit[base_credit['loan']<=1.377630 ])
print()
print()
print(np.unique(base_credit['default'], return_counts=True))
print()
print()
sns.countplot(x=base_credit['default'])
plt.show()
plt.hist(x=base_credit['age'])
plt.show()
plt.hist(x=base_credit['income'])
plt.show()
plt.hist(x=base_credit['loan'])
plt.show()
grafico=px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()

