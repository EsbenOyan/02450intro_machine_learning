import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

data = pd.read_csv('./Data/Classification.data')
# Select the columns to plot
mean = data.iloc[:, 2:12]
sde = data.iloc[:, 12:22]
worst = data.iloc[:, 22:32]

#log transform the subdatasets
mean.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
mean = np.log2(mean.where(mean > 0))

sde.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
sde = np.log2(sde.where(sde > 0))

worst.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
worst = np.log2(worst.where(worst > 0))

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

################################################################

mean = data.iloc[:, 2:12]
sde = data.iloc[:, 12:22]
worst = data.iloc[:, 22:32]

