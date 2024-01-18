import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
base_credit= pd.read_csv(r'C:\Users\joaov\Downloads\credit_data.csv')



print(base_credit[base_credit['age']<0])

print()

base_credit2 = base_credit.drop('age', axis = 1)
print(base_credit2)

print()

base_credit3 = base_credit.drop(base_credit[base_credit['age']<0].index)
print(base_credit3)

print(base_credit3.loc[base_credit3['age']<0])