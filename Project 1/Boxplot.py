import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

from data_preprocessing import *
dfjoint, dfRec, dfClas = dataPreprocess()

colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()
mean = dfjoint.loc[:,colNamesMeans]
sde = dfjoint.loc[:,colNamesStd]
worst = dfjoint.loc[:,colNamesExt]
colNames = colNamesMeans + colNamesStd + colNamesExt + ["tumor size", "time"]

#log transform the subdatasets
# mean.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
mean = np.log2(mean.where(mean > 0))

# sde.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
sde = np.log2(sde.where(sde > 0))

# worst.replace([np.inf, -np.inf, np.nan], 0, inplace = True)
worst = np.log2(worst.where(worst > 0))

All = np.log2(dfjoint.loc[:,colNames].where(dfjoint.loc[:,colNames] > 0))

# Set the x-axis tick labels
labels = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
          'compactness', 'concavity', 'concave points', 'symmetry',
          'fractal dimension']

"""
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



sns.set(style="ticks")
fig_mean = plt.figure(figsize=(24, 9))
sns.boxplot(data=All, palette="pastel", orient="v")
sns.despine(trim=True)
plt.xticks(rotation=90)
plt.xticks(np.arange(len(colNames)), colNames)
plt.ylim(mean.min().min(), mean.max().max())
plt.title("All attributes")
fig_mean.savefig('boxplot_all.png', bbox_inches = 'tight')
"""

"""
log_values = np.log2(dfjoint.loc[:,colNames].where(dfjoint.loc[:,colNames] > 0))
index_sort = log_values.mean().sort_values(ascending = False) .index
# now applying the sorted indices to the data
df_sorted = log_values[index_sort]
df_norm = (df_sorted - df_sorted.mean())/df_sorted.std()
myFig = plt.figure()
boxplot_sort = df_sorted.boxplot(grid = False, rot = 90, fontsize = 12, figsize = (18,5))
myFig.savefig('boxplot_all_sort.png', bbox_inches = 'tight')
myFig.title('Boxplot of sorted log2 transformed values')

myFig = plt.figure()
boxplot_norm = df_norm.boxplot(grid = False, rot = 90, fontsize = 12, figsize = (18,5))
myFig.title('Boxplot of normalized log2 transformed values')
myFig.savefig('boxplot_all_norm.png', bbox_inches = 'tight')



log_values = np.log2(dfjoint.loc[:,colNames].where(dfjoint.loc[:,colNames] > 0))
index_sort = log_values.mean().sort_values(ascending = False).index
# now applying the sorted indices to the data
df_sorted = log_values[index_sort]
df_sorted = np.log2(df_sorted.loc[:,colNames].where(df_sorted.loc[:,colNames] > 0))


df_norm = (dfjoint - dfjoint.mean())/dfjoint.std()
df_norm = np.log2(df_norm.loc[:,colNames].where(df_norm.loc[:,colNames] > 0))





log_values = np.log2(dfjoint.loc[:,colNames].where(dfjoint.loc[:,colNames] > 0))
index_sort = log_values.mean().sort_values(ascending = False).index
# now applying the sorted indices to the data
df_sorted = log_values[index_sort]
df_norm = (df_sorted - df_sorted.mean())/df_sorted.std()

"""


log_values = np.log2(dfjoint.loc[:,colNames].where(dfjoint.loc[:,colNames] > 0))
index_sort = log_values.mean().sort_values(ascending = False).index
# now applying the sorted indices to the data
df_sorted = log_values[index_sort]



df_norm = (dfjoint - dfjoint.mean())/dfjoint.std()
log_values = np.log2(df_norm.loc[:,colNames].where(df_norm.loc[:,colNames] > 0))
# now applying the sorted indices to the data
df_norm = df_norm[index_sort]


fig, ax = plt.subplots(2,1, sharex = True, figsize = (18,12))
ax[0].set_title('Boxplot of sorted log2 transformed values', fontsize = 20)
ax[1].set_title('Boxplot of normalized values', fontsize = 20)

ax[0].set_xticklabels(index_sort, rotation = 90, fontsize = 20)
ax[1].set_xticklabels(index_sort, rotation = 90, fontsize = 20)

sns.despine(ax = ax[0], bottom = False)
sns.despine(ax = ax[1], bottom = False)

colorPalette = sns.light_palette("seagreen", reverse = True, n_colors=len(index_sort))

sns.boxplot(ax = ax[0], data=df_sorted, palette = colorPalette, orient="v")
sns.boxplot(ax = ax[1], data=df_norm, palette = colorPalette, orient="v")

fig.tight_layout()

plt.savefig('boxplot_compare.png', bbox_inches = 'tight')
################################################################
