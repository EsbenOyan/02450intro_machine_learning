import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np

data = pd.read_csv('./Data/Classification.data')

# Select the columns to plot
mean = data.iloc[:, 2:12]
sde = data.iloc[:, 12:22]
worst = data.iloc[:, 22:32]

# Set the x-axis tick labels
labels = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
          'compactness', 'concavity', 'concave points', 'symmetry',
          'fractal dimension']

# Create the boxplots
sns.set(style="ticks")
fig_mean = plt.figure(figsize=(8, 6))
sns.boxplot(data=mean, palette="pastel", orient="v")
sns.despine(trim=True)
plt.xticks(rotation=90)
plt.xticks(np.arange(len(labels)), labels)
plt.ylim(mean.min().min(), mean.max().max())
plt.title("Mean of attributes")
fig_mean.savefig('boxplot_mean.png', bbox_inches = 'tight')

sns.set(style="ticks")
fig_sde =plt.figure(figsize=(8, 6))
sns.boxplot(data=sde, palette="pastel", orient="v")
sns.despine(trim=True)
plt.xticks(rotation=90)
plt.xticks(np.arange(len(labels)), labels)
plt.ylim(sde.min().min(), sde.max().max())
plt.title("Standard error for attributes")
fig_sde.savefig('boxplot_sde.png', bbox_inches = 'tight')

sns.set(style="ticks")
fig_worst = plt.figure(figsize=(8,6))
sns.boxplot(data=worst, palette="pastel", orient="v")
sns.despine(trim=True)
plt.xticks(rotation=90)
plt.xticks(np.arange(len(labels)), labels)
plt.ylim(worst.min().min(), worst.max().max())
plt.title('"Worst" value for attributes')
fig_worst.savefig('boxplot_worst.png', bbox_inches = 'tight')


# Show the plot
# plt.show()