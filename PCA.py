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

X = mean.values

# Create a PCA object with the number of components to keep
pca = PCA(n_components=10)

# Fit the PCA model to the data and transform the data
X_pca = pca.fit_transform(X)

# Print the explained variance ratio of each principal component
print(pca.explained_variance_ratio_)
print("The PC that explains the most is {} and explains {}".format(np.argmax(pca.explained_variance_ratio_), np.max(pca.explained_variance_ratio_)))
# Create a plot showing the principal components
plt.plot(X_pca)
plt.title('Principal Components')
plt.xlabel('Component')
plt.ylabel('Variance')
#plt.show()

Y = sde.values 
# Create a PCA object with the number of components to keep 
pca = PCA(n_components=10) 
# Fit the PCA model to the data and transform the data 
Y_pca = pca.fit_transform(Y) 
# Print the explained variance ratio of each principal component 
print(pca.explained_variance_ratio_)
print("The PC that explains the most is {} and explains {}".format(np.argmax(pca.explained_variance_ratio_), np.max(pca.explained_variance_ratio_)))
# Create a plot showing the principal components 
plt.plot(Y_pca) 
plt.title('Principal Components') 
plt.xlabel('Component') 
plt.ylabel('Variance') 
#plt.show()

Z = worst.values  
# Create a PCA object with the number of components to keep 
pca = PCA(n_components=10) 
# Fit the PCA model to the data and transform the data 
Z_pca = pca.fit_transform(Z) 
# Print the explained variance ratio of each principal component 
print(pca.explained_variance_ratio_)
print("The PC that explains the most is {} and explains {}".format(np.argmax(pca.explained_variance_ratio_), np.max(pca.explained_variance_ratio_)))
# Create a plot showing the principal components 
plt.plot(Z_pca) 
plt.title('Principal Components') 
plt.xlabel('Component') 
plt.ylabel('Variance') 
#plt.show()