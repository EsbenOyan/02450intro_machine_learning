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
def plot_variance(data, title):
    # Select the columns to use in PCA
    features = data.iloc[:, 0:10]

    # Scale the features to have zero mean and unit variance
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Fit PCA to the scaled features
    pca = PCA()
    pca.fit(scaled_features)

    # Plot the explained variance ratio
    plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, 'bo-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'{title} - Explained Variance Ratio by Principal Component')
    plt.ylim([0,1])
    plt.xticks(range(1, pca.n_components_ + 1))
    plt.grid(True)

    # Plot the cumulative explained variance ratio
    cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, pca.n_components_ + 1), cumulative_var_ratio, 'ro-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'{title} - Cumulative Explained Variance Ratio by Principal Component')
    plt.ylim([0,1])
    plt.xticks(range(1, pca.n_components_ + 1))
    plt.grid(True)

    # Add a dashed line at 90% cumulative explained variance ratio
    plt.axhline(y=0.9, color='r', linestyle='--')

    # Add legend
    plt.legend(['Explained Variance Ratio', 'Cumulative Explained Variance Ratio', '90% Threshold'], loc='best')

    plt.savefig(title + '_variance_explained.png')
    plt.close()

# Plot variance for "mean" dataset
plot_variance(mean, 'Mean')

# Plot variance for "sde" dataset
plot_variance(sde, 'SDE')

# Plot variance for "worst" dataset
plot_variance(worst, 'Worst')