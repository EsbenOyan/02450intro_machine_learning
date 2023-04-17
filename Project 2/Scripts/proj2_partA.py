from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot, hist)
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
import sklearn.linear_model as lm
from sklearn import model_selection
#import torch
import sys
sys.path.append('Project 2/Tools')
from toolbox_02450 import rlr_validate, correlated_ttest, train_neural_net, draw_neural_net

from data_preprocessing import *
dfjoint, dfRec, dfClas = dataPreprocess()

colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()
mean = dfjoint.loc[:,colNamesMeans]
sde = dfjoint.loc[:,colNamesStd]
worst = dfjoint.loc[:,colNamesExt]
colNames = colNamesMeans + colNamesStd + colNamesExt

#####################################
## Implement ex5_2_4.py

# Split dataset into features and target vector
colNames = dfjoint.columns
#time_idx = colNames.index('time')
y = dfjoint.loc[:,'time'].dropna()


#test = dfjoint.loc[:, 'tumor size']

#X_cols = list(range(0,time_idx)) + list(range(time_idx+1,len(attributeNames)))

# Columns to base regression on
X_cols = colNames.drop(['Diagnosis', 'time', 'outcome'])
#X_cols = list(range(0,time_idx)) + list(range(time_idx+1,len(colNames)))

X = dfjoint.loc[:, X_cols].dropna()
#X = x.values.reshape(-1,1)


# Feature transformation
X1 = X - np.outer(np.ones((len(y), 1)),X.mean(0))
X = X1*(1/np.std(X1,0))
X = np.concatenate((np.ones((X.shape[0],1)), X), 1)


# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Get model scores
r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# Introduce regression line
#m, b = np.polyfit(x, y, 1)

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('X label'); ylabel('Y label');
#subplot(2,1,2)
#hist(residual,40)

show()

