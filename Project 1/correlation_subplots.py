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
df = dfjoint.loc[:,colNamesExt[0:7]]
plot = sns.PairGrid(df, diag_sharey=False)
plot.map_lower(sns.kdeplot, cmap="Blues_d")
plot.map_upper(plt.scatter)
plot.map_diag(sns.histplot, lw=3)
plot.savefig('hist_Ext.png', bbox_inches = 'tight')
"""
plt.rcParams["axes.labelsize"] = 17
df = np.log2(dfjoint.loc[:,colNamesMeans[0:7] + ['tumor size']])
plot = sns.PairGrid(df, diag_sharey=False, )
plot.map_lower(sns.kdeplot, cmap="Blues_d")
plot.map_upper(plt.scatter)
plot.map_diag(sns.histplot, lw=3)
for ax in plot.axes[:,0]:
    ax.get_yaxis().set_label_coords(-0.2,0.5)
plot.savefig('hist_Ext_log2.png', bbox_inches = 'tight')
