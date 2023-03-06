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
# plt.show()

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
plt.show()


###############################################################################
                #2.1.6 Insertion#
###############################################################################

## Data preprocessing

from scipy.linalg import svd
from data_preprocessing import *
dfjoint, dfRec, dfClas = dataPreprocess()

colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()
dfmeanColumns = dfClas.loc[:,colNamesMeans]


## Select attribute type
attributeNames = colNamesMeans
X_s = dfmeanColumns

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

r = np.arange(1,X_s.shape[1]+1)
plt.bar(r, np.std(X_s,0))
plt.xticks(r, attributeNames, rotation=45, ha='right')
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Breastcancer: attribute standard deviations')
plt.show()


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
nrows=3
ncols=2
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
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames)
    plt.axis('equal')
    
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
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()


