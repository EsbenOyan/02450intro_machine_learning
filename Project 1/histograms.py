from data_preprocessing import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt

dfjoint, dfRec, dfClas = dataPreprocess()
dfjoint[dfjoint.loc[:,'lymph node status'] != 0]
colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()

"""
sns.set(style="white")
df = dfjoint.loc[:,colNamesExt[0:5] + ['Diagnosis']]
plot = sns.PairGrid(df, diag_sharey=False)
plot.map_lower(sns.kdeplot, cmap="Blues_d")
plot.map_upper(plt.scatter)
plot.map_diag(sns.histplot, lw=3)
plot.savefig('hist_Ext.png')

df = np.log2(dfjoint.loc[:,colNamesExt[0:5] + ['Diagnosis']])
plot = sns.PairGrid(df, diag_sharey=False)
plot.map_lower(sns.kdeplot, cmap="Blues_d")
plot.map_upper(plt.scatter)
plot.map_diag(sns.histplot, lw=3)
plot.savefig('hist_Ext_log2.png')

"""

dfjoint.columns = [column if 'extreme' not in column else column[:-5] for column in dfjoint.columns]
print(dfjoint.columns)
hist = dfjoint.hist(bins = 24, figsize = (24,15))
plt.show()
#plt.savefig('histograms.png', bbox_inches = 'tight')