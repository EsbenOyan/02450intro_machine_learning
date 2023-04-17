from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
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

