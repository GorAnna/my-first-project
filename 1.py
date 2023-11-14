import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("biostats.csv")

print(df.head())
print(df.info())

print(df.describe())

plt.hist(df['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()