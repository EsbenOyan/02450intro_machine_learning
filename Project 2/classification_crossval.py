import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from data_preprocessing import *
from sklearn.metrics import accuracy_score
import scipy.stats
import matplotlib.pyplot as plt

def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21), "\n")

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p


# Load the dataset
dfjoint, dfRec, dfClas = dataPreprocess()
colNamesMeans, colNamesStd, colNamesExt, colNamesOther = getSpecificColNames()

# convert malign/benign to binary values
mapping = {'B': 0, 'M': 1}
dfClas = dfClas.replace({'Diagnosis': mapping})
colNames = colNamesMeans + colNamesStd + colNamesExt

X = dfClas.loc[:, dfClas.columns != 'Diagnosis']
X = (X - X.mean())/X.std()
y = dfClas['Diagnosis']

logreg_num_params = 50
knn_num_params = 20
k_inner = 10
k_outer = 5

# Initialize the result arrays
logreg_scores = np.zeros((k_outer, logreg_num_params))
knn_scores = np.zeros((k_outer, knn_num_params))
baseline_acc = np.zeros((k_outer, k_inner))
baseline_err = np.zeros((k_outer, k_inner))

# Set up the parameter grids for logistic regression and KNN
logreg_params = {'C': 1/np.logspace(-4, 2, logreg_num_params)}
knn_params = {'n_neighbors': np.arange(1, 1 + knn_num_params)}

# Set up the outer and inner cross-validation loops
outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=1)
logreg_outer_results = list()
knn_outer_results = list()

logreg_yhat_best = []
knn_yhat_best = []
baseline_yhat_best = []
y_true = []

# Outer cross-validation loop
for i, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    print(f"Fold {i+1}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    y_true.append(y_test)
    inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=1)

    # Inner cross-validation loop for logistic regression
    logreg_grid = GridSearchCV(LogisticRegression(penalty='l2', max_iter=1000), logreg_params,n_jobs = -1, cv=inner_cv, refit = True)
    logreg_result = logreg_grid.fit(X_train, y_train)
    best_model = logreg_result.best_estimator_
    logreg_scores[i, :] = logreg_grid.cv_results_['mean_test_score']
    logreg_yhat = best_model.predict(X_test)
    logreg_yhat_best.append(logreg_yhat)
    acc = accuracy_score(y_test, logreg_yhat)
    logreg_Etest = len(logreg_yhat[logreg_yhat != y_test]) / len(y_test)
    logreg_outer_results.append(acc)
    print('Logreg:   >acc=%.3f, Etest=%.3f, est=%.3f, cfg=%s' % (acc, logreg_Etest, logreg_result.best_score_, 1/logreg_result.best_params_['C']))

    # Inner cross-validation loop for KNN
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, n_jobs=-1, cv=inner_cv, refit=True)
    knn_result = knn_grid.fit(X_train, y_train)
    best_model = knn_result.best_estimator_
    knn_scores[i, :] = knn_grid.cv_results_['mean_test_score']
    knn_yhat = best_model.predict(X_test)
    knn_yhat_best.append(knn_yhat)
    acc = accuracy_score(y_test, knn_yhat)
    knn_Etest = len(knn_yhat[knn_yhat != y_test]) / len(y_test)
    knn_outer_results.append(acc)
    print('KNN:      >acc=%.3f, Etest=%.3f, est=%.3f, cfg=%s' % (acc, knn_Etest, knn_result.best_score_, knn_result.best_params_))

    # inner cross-validation loop for baseline model
    for v, (train_idx, test_idx) in enumerate(inner_cv.split(X_train, y_train)):
        X_train_, X_test_ = X.iloc[train_idx], X.iloc[test_idx]
        y_train_, y_test_ = y.iloc[train_idx], y.iloc[test_idx]

        max_class = max(sum(y_train_), len(y_train_) - sum(y_train_))
        if sum(y_train_) == max_class:
            class_to_predict = 1
            baseline_yhat = np.ones(len(y_test_))
        else: 
            class_to_predict = 0
            baseline_yhat = np.zeros(len(y_test_))
    
        acc = accuracy_score(y_test_, baseline_yhat)
        acc = len(baseline_yhat[baseline_yhat == y_test_]) / len(y_test_)
        baseline_Etest = len(baseline_yhat[baseline_yhat != y_test_]) / len(y_test_)
        baseline_acc[i, v] = acc
        baseline_err[i, v] = baseline_Etest
    
    baseline_yhat_best.append(baseline_yhat)


    acc_idx = max(range(len(baseline_acc[i, :])), key=baseline_acc[i, :].__getitem__)

    print('Baseline: >acc=%.3f, Etest=%.3f' % (baseline_acc[i, acc_idx], baseline_err[i, acc_idx]))
    print('\n')

    
    

############################################################################
############################ Compute statistics ############################
############################################################################



logreg_yhat_best = np.concatenate(logreg_yhat_best)
knn_yhat_best = np.concatenate(knn_yhat_best)
baseline_yhat_best = np.zeros(len(knn_yhat_best))
y_true = np.concatenate(y_true)


# Compute statistics for KNN vs logistic regression
print("\n\nmcnemars test for KNN vs logreg")
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, knn_yhat_best, logreg_yhat_best, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# Compute statistics for KNN vs baseline
print("\n\nmcnemars test for KNN vs baseline")
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, knn_yhat_best, baseline_yhat_best, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# # Compute statistics for logistic regression vs baseline
print("\n\nmcnemars test for logreg vs baseline")
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, logreg_yhat_best, baseline_yhat_best, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)





############################################################################
############### Fit a logistic regression model with lambda = 0.6 ##########
############################################################################



from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
features = X.columns

X = X.to_numpy()
y = y.to_numpy()

model = lm.LogisticRegression()
model = model.fit(X,y)

# print weights with features
for weight, feature in zip(model.coef_[0], features):
    print(feature, ': ', round(weight,3))

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_white_prob = model.predict_proba(X)[:, 0] 

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))


f = figure()
print(type(y))
class0_ids = list(np.nonzero(y==0))
plot(class0_ids, y_est_white_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_white_prob[class1_ids], '.r')
xlabel('Tumors')
ylabel('Predicted prob. of a Malignant tumor')
legend(['Malignant', 'Benign'])
ylim(-0.01,1.5)
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('#ccb787')
leg.legendHandles[1].set_color('red')

show()
