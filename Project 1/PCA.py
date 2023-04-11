import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from scipy.linalg import svd

from data_preprocessing import *
dfjoint, dfRec, dfClas = dataPreprocess()

colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()
mean = dfjoint.loc[:,colNamesMeans]
sde = dfjoint.loc[:,colNamesStd]
worst = dfjoint.loc[:,colNamesExt]
colNames = colNamesMeans + colNamesStd + colNamesExt

def plot_variance(features, title):
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
plot_variance(dfjoint.loc[:,colNames], 'PCA')

###############################################################################
                #2.1.6 Insertion#
###############################################################################

## Data preprocessing
attributeNames = colNames
X_s = dfClas.loc[:, colNames]

# X_s = np.empty((X_s.shape[0], X_s.shape[1]))
# for i, col_id in enumerate(range(1, len(attributeNames))):
#     X_s[:, i] = np.asarray(doc.col_values(col_id, X_s.shape[0], X_s.shape[1]))

N = X_s.shape[0]
classLabels = dfClas.loc[:,"Diagnosis"]
classNames = sorted(set(dfClas.loc[:,"Diagnosis"]))
classDict = dict(zip(classNames, range(2)))
                 
y = np.asarray([classDict[value] for value in classLabels])
###############################################################################
## Start of copied code

"""
r = np.arange(1,X_s.shape[1]+1)
plt.bar(r, np.std(X_s,0))
plt.xticks(r, attributeNames, rotation=45, ha='right')
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Breastcancer: attribute standard deviations')
plt.show()
"""

# Subtract the mean from the data
Y1 = X_s - np.ones((N, 1))*X_s.mean(0).to_numpy()

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X_s - np.ones((N, 1))*X_s.mean(0).to_numpy()
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
plt.title('Breastcancer: Effect of standardization')
nrows=1
ncols=1

for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplots(nrows, ncols)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.title('PCA analysis of breast cancer dataset')
    plt.legend(classNames)
    plt.axis('equal')
    plt.savefig('PCA_Diagnosis', bbox_inches = 'tight')
    
    """
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
    """     
    # Plot cumulative variance explained
    plt.subplots(1,1)
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title('Variance explained by principal components')
    plt.savefig('PCA_Diagnosis_explained_variance', bbox_inches = 'tight')

df = pd.DataFrame(V)
PC1 = df.iloc[:,0]
PC2 = df.iloc[:,1]
PC3 = df.iloc[:,2]
print(PC1)

fig, ax = plt.subplots(3,1, sharex = True, figsize = (18,12))
ax[0].set_title('PC1 weights', fontsize = 20)
ax[1].set_title('PC2 weights', fontsize = 20)
ax[2].set_title('PC3 weights', fontsize = 20)

ax[2].set_xticklabels(colNames, rotation = 90, fontsize = 20)

ax[0].set_ylabel('')
ax[1].set_ylabel('')
ax[2].set_ylabel('')

sns.despine(ax = ax[0])
sns.despine(ax = ax[1])
sns.despine(ax = ax[2])

colorPalette = sns.light_palette("seagreen", reverse = True, n_colors=len(colNames))

sns.barplot(ax = ax[0], x = colNames, y = PC1, palette = colorPalette)
sns.barplot(ax = ax[1], x = colNames, y = PC2, palette = colorPalette)
sns.barplot(ax = ax[2], x = colNames, y = PC3, palette = colorPalette)

fig.tight_layout()

plt.savefig('test.png', bbox_inches = 'tight')