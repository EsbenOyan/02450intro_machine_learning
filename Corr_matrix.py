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

# Define a function to create correlation plot and save
def plot_corr_matrix(df):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(df.name + ' Correlation Matrix')
    ax.set_xticklabels(['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension'], rotation=45)
    ax.set_yticklabels(['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension'], rotation=0)
    plt.savefig(df.name + '_corrPlot.png')
    plt.close()

# Call the function for each dataframe
mean.name = 'Mean'
plot_corr_matrix(mean)

sde.name = 'Standard Error'
plot_corr_matrix(sde)

worst.name = 'Worst'
plot_corr_matrix(worst)