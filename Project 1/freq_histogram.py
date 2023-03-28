import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from scipy.stats import norm

data = pd.read_csv('./Data/Classification.data')
data = data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)
# Select the columns to plot
mean = data.iloc[:, 2:12]
sde = data.iloc[:, 12:22]
worst = data.iloc[:, 22:32]

def plot_histograms(df, title):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
    fig.suptitle(title, fontsize=16, y=0.98)
    for i, col in enumerate(df.columns):
        row, col = divmod(i, 5)
        # Plot histogram of the data
        sns.histplot(df[col], kde=False, bins=30, ax=axs[row, col])
        axs[row, col].set_title(col)
    plt.tight_layout()
    plt.savefig(title + '_histograms.png')
    plt.show()

# Call the function for each dataframe
plot_histograms(mean, 'Mean Features')
plot_histograms(sde, 'Standard Error Features')
plot_histograms(worst, 'Worst Features')