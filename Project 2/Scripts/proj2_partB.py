# Import necessary packages and functions
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
import sklearn.linear_model as lm
from sklearn import model_selection
import torch

# Add the path to the toolbox_02450 module
import sys
sys.path.append('Project 2/Tools')
from toolbox_02450 import rlr_validate, correlated_ttest, train_neural_net, draw_neural_net

# Import custom functions from data_preprocessing module
from data_preprocessing import *
dfjoint, dfRec, dfClas = dataPreprocess()
colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()

#######################################################################

# Extract data
mean = dfjoint.loc[:, colNamesMeans]
sde = dfjoint.loc[:, colNamesStd]
worst = dfjoint.loc[:, colNamesExt]
colNames = colNamesMeans + colNamesStd + colNamesExt

# Discretize time variable
time_discretized = []
for i in range(len(dfRec.iloc[:,0])):
    if dfRec['time'].iloc[i] <= 12:
        time_discretized.append('<1 years')
    elif dfRec['time'].iloc[i] <= 36:
        time_discretized.append('1-3 years')
    elif dfRec['time'].iloc[i] <= 72:
        time_discretized.append('>3-6 years')
    else:
        time_discretized.append('6+ years')

# Add discretized time variable to dataframe
dfRec['time_discretized'] = time_discretized

#######################################################################

# Prepare data for classification
attributeNames = [u'Offset'] + colNames
X = dfRec.loc[:, colNames]
classLabels = dfRec.loc[:,"time_discretized"]
classNames = sorted(set(dfRec.loc[:,"time_discretized"]))

# Create a dictionary mapping class names to integer labels
classDict = dict(zip(classNames, range(4)))
y = np.asarray([classDict[value] for value in classLabels])

# Define data dimensions
N = X.shape[0]
M = len(attributeNames)
# M = M + 1
C = len(classNames)

# Add bias to input data
X = np.concatenate((np.ones((X.shape[0],1)), X), 1)


#######################################################################
# 2 level cross validation, linear regression model

# Define number of folds for cross-validation
K = 10

# Create cross-validation partition for evaluation
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Define lambda values to be tested
lambdas = np.power(10.,range(-5,9))

# Initialize variables to store errors and weights
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
errors_lin_reg = np.empty((K,1))

# Loop over each fold in the cross-validation
k = 0
print('\n-----------------------')
print('Linear regression model:\n')

for train_index, test_index in CV.split(X, y):

    # Extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Define the number of internal folds for cross-validation
    internal_cross_validation = 10

    # Validate the performance of the linear regression model using ridge regression
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize the data based on the training set and save the mean and standard deviation
    # since they're part of the model
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    # Save the generalization error for later statistics
    errors_lin_reg[k] = opt_val_err

    k += 1

    # Print the results of the current fold
    print('Outer fold {0}'.format(k))
    print('has optimal lambda: {0}'.format(opt_lambda))
    print('and optimal generalization error: {0}'.format(opt_val_err))

#######################################################################
# 2 level Cross validation, baseline

# Define number of outer and inner folds for cross-validation
K = 10
K_i = 10

# Create cross-validation partition for evaluation
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Initialize variables to store results
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
error_i = np.empty((K_i, 1))
error_o = np.empty((K, 1))

print('\n-----------------------')
print('Baseline model:\n')

# Loop over outer cross-validation folds
k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    
    CV_i = model_selection.KFold(K_i, shuffle=True)
    
    k_i = 0
    for train_index, test_index in CV_i.split(X_train,y_train):
        
        # extract training and test set for current inner CV fold
        X_train_i = X_train[train_index]
        y_train_i = y_train[train_index]
        X_test_i = X_train[test_index]
        y_test_i = y_train[test_index]
        
        # calculate mean squared error for inner CV fold
        mean_i = np.mean(y_train_i)
        error_i[k_i]=(1/len(y_test_i))*np.sum((y_test_i-mean_i)**2)
        #Error_i[k_i] = (1/len(y_test_i)*
        
        k_i+=1
    # find minimum error for inner CV folds
    error_o[k] = min(error_i)
        
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    print('Outer fold {} generalization error: {}'.format(k+1,error_o[k]))

    k+=1

#######################################################################
# ANN

# Create crossvalidation partition for evaluation
# Set the number of folds for cross-validation
K = 10
# Set the number of folds for inner cross-validation
K_i = 10
# Create K-fold cross-validation partitions for evaluation
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Initialize variables for storing results
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
error_i = np.empty((K_i, 1))
minerror = np.empty((K, 1))
min_ind = np.empty((K, 1))

# Loop over the K folds for cross-validation
k_o = 1
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]


    # Parameters for neural network classifier
    # K-fold crossvalidation
    CV = model_selection.KFold(K_i, shuffle=True)
    
    #print('Training model of type:\n\n{}\n'.format(str(model())))
    # Initialize a list for storing generalizaition error in each loop
    errors = np.empty((K_i, 1))

    # Loop over the K_i folds for the inner cross-validation
    for (k, (train_index, test_index)) in enumerate(CV.split(X_train,y_train)): 
        #print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K_i))  
        
        # Set the number of hidden units for the neural network
        n_hidden_units = k + 1

        # Set the number of networks trained in each k-fold
        n_replicates = 1

        # Set the maximum number of iterations
        max_iter = 10000
        
        # Define the model using a lambda function
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        # Set the loss function
        loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train_i = torch.Tensor(X_train[train_index,:])
        y_train_i = torch.Tensor(y_train[train_index])
        X_test_i = torch.Tensor(X_train[test_index,:])
        y_test_i = torch.Tensor(y_train[test_index])
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_i,
                                                           y=y_train_i,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        # Print the best loss for the current cross-validation fold
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_test_est = net(X_test_i)
        
        # Store error rate for current CV fold
        errors[k] = final_loss
    
    # Store the minimum error rate and its index for the current outer fold
    minerror[k_o-1]=min(errors)
    min_ind[k_o-1] = np.argmin(errors)
    
    # Increment the outer fold counter
    k_o += 1


print('\n-----------------------')
print('ANN model:\n')  

for ii in range(10):
    print('For outer fold {} best generalization error is {} for h = {} '.format(ii+1,minerror[ii],min_ind[ii]+1))

#######################################################################
# Statistical test for method setup II

# MSE are saved in variable:
    # errors_lin_reg for linear regression
    # error_o for baseline
    # minerror for ANN

# Generalization error differences
r_lin_base = errors_lin_reg - error_o # Calculate difference between linear regression and baseline errors
r_lin_ANN = minerror - errors_lin_reg # Calculate difference between ANN and linear regression errors
r_ANN_base = minerror - error_o # Calculate difference between ANN and baseline errors

# significance level
alpha = 0.05

rho = 1/K

print(' ')
# Test lin reg vs baseline
print('Correlated t-test for linear regression and baseline')
p_val, conf_int = correlated_ttest(r_lin_base, rho, alpha) # Perform correlated t-test between linear regression and baseline errors
print('p-value {} and confidence interval {}'.format(p_val, conf_int))
print(' ')

# Test ANN vs baseline
print('Correlated t-test for ANN and baseline')
p_val, conf_int = correlated_ttest(r_ANN_base, rho, alpha) # Perform correlated t-test between ANN and baseline errors
print('p-value {} and confidence interval {}'.format(p_val, conf_int))
print(' ')

# Test lin reg vs ANN
print('Correlated t-test for linear regression and ANN')
p_val, conf_int = correlated_ttest(r_lin_ANN, rho, alpha) # Perform correlated t-test between linear regression and ANN errors
print('p-value {} and confidence interval {}'.format(p_val, conf_int))
