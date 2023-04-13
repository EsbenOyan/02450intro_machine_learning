import numpy as np
import pandas as pd
import os

def dataPreprocess() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """**********************************************************"""
    """******************** DEFINE VARIABLES ********************"""
    """**********************************************************"""

    # create paths to data independent of operating system
    pathRec = os.path.join('Data', 'Recurrence.data')
    pathClas = os.path.join('Data', 'Classification.data')

    # assign column names in order of appearence in datafile
    recHeader = ['patient ID', 'outcome', 'time', 'radius mean', 'texture mean', 'perimeter mean', 'area mean',
                'smoothness mean', 'compactness mean', 'concavity mean', 'concave points mean', 'symmetry mean',
                'fractal dimension mean', 'radius std.', 'texture std.', 'perimeter std.', 'area std.', 
                'smoothness std.', 'compactness std.', 'concavity std.', 'concave points std.', 'symmetry std.', 
                'fratal dimension std.', 'radius extreme mean', 'texture extreme mean', 'perimeter extreme mean', 
                'area extreme mean', 'smoothness extreme mean', 'compactness extreme mean',
                'concavity extreme mean', 'concave points extreme mean', 'symmetry extreme mean', 
                'fractal dimension extreme mean', 'tumor size', 'lymph node status']

    clasHeader = ['patient ID', 'Diagnosis', 'radius mean', 'texture mean', 'perimeter mean', 'area mean',
                'smoothness mean', 'compactness mean', 'concavity mean', 'concave points mean', 'symmetry mean',
                'fractal dimension mean', 'radius std.', 'texture std.', 'perimeter std.', 'area std.', 
                'smoothness std.', 'compactness std.', 'concavity std.', 'concave points std.', 'symmetry std.', 
                'fratal dimension std.', 'radius extreme mean', 'texture extreme mean',
                'perimeter extreme mean', 'area extreme mean', 'smoothness extreme mean', 'compactness extreme mean',
                'concavity extreme mean', 'concave points extreme mean', 'symmetry extreme mean',
                'fractal dimension extreme mean']



    """**********************************************************"""
    """*********** LOAD DATA AND REMOVE MISSING VALUES **********"""
    """**********************************************************"""

    # load data
    dfRec = pd.read_csv(pathRec, header = None, names = recHeader, index_col = 'patient ID')
    dfClas = pd.read_csv(pathClas, header = None, names = clasHeader, index_col = 'patient ID')

    # join datasets keeping all columns from both sets (each column is additionally labeled with 'clas' or 'rec' 
    # depending on its origin). Only patients which are common between the datasets are added onto the classification dataset (139 observations).
    dfjoint = dfClas.join(dfRec, how = 'left', lsuffix = ' clas', rsuffix = ' rec')

    # find patient IDs with missing values and remove those patients from the dataset
    for column in dfjoint.columns:
        missingvalues = dfjoint.loc[:,column][dfjoint.loc[:,column] == '?']
        if missingvalues.count() != 0:
            dfjoint = dfjoint.drop(index = missingvalues.index.values)

    """**********************************************************"""
    """*** REMOVE DATA THAT IS INCONSISTENT BETWEEN DATASETS ****"""
    """**********************************************************"""

    # some patient IDs/observation contain information that is different between classification and recurrence datasets
    # as this is assumed to be human error when noting the data, these observations are removed

    # first, find column names that are common in the two datasets
    sharedColumns = [column for column in recHeader if column in clasHeader]
    sharedColumns.remove('patient ID')

    # second, find the names of columns after the join (we added either ' rec' or ' clas') that originated in the recurrence dataset
    recColNames = [column for column in dfjoint.columns if 'rec' not in column and column not in clasHeader]

    # third, find patient IDs of data originating from rec dataset, so we can isolate these
    # (if we use the full joint dataset, we will compare values in classification with NaN values in recurrence, which
    # would show up as a difference, but we are not interested in that)
    dfRecindices = dfjoint.loc[:, recColNames].dropna().index
    dfjointOriginRec = dfjoint[dfjoint.index.isin(dfRecindices)]

    # fourth, for each shared column check if values are identical between the datasets and save patients that have inconsistent data
    # in a variable remove_patients
    remove_patients = []
    for column in sharedColumns:
        wrongObservations = dfjointOriginRec[dfjointOriginRec.loc[:, column + ' clas'] != dfjointOriginRec.loc[:, column + ' rec']].index
        for observation in wrongObservations:
            if not observation in remove_patients:
                remove_patients.append(observation)

    # now that we have a list (remove_patients) with observations that contains conflicting information between classification and recurrence
    # datasets, we can remove those observations from the full joint dataframe. After removing conflicting information and missing values we
    # have 554 observations for classification and 124 observations for regression
    dfjoint = dfjoint.drop(remove_patients)


    """**********************************************************"""
    """**************** FINAL DATA PREPROCESSING ****************"""
    """**********************************************************"""

    # as the collected data is the same between classification and recurrence we can keep just one version of the "clas" and "rec" columns
    sharedColumnsRec = [column + ' rec' for column in sharedColumns]
    dfjoint = dfjoint.drop(sharedColumnsRec, axis = 1)
    jointColNames = [column if ' clas' not in column else column[:-5] for column in dfjoint.columns]
    dfjoint.columns = jointColNames
    dfjoint['lymph node status'] = dfjoint['lymph node status'].apply(pd.to_numeric)

    # we can create a subdataframe containing only columns found in the classification dataset
    clasHeader.remove('patient ID')
    dfClas = dfjoint.loc[:, clasHeader].dropna()

    # and we can do the same with the rec dataset
    recHeader.remove('patient ID')
    dfRec = dfjoint.loc[:, recHeader].dropna()

    return dfjoint, dfRec, dfClas

def getSpecificColNames() -> tuple[list, list, list, list]:

    # you can use the following column name groupings to select either mean, stdev, extreme mean, or other columns
    colNamesMeans = ['radius mean', 'texture mean', 'perimeter mean', 'area mean', 'smoothness mean', 'compactness mean', 
                    'concavity mean', 'concave points mean', 'symmetry mean', 'fractal dimension mean']
    colNamesStd = ['radius std.', 'texture std.', 'perimeter std.', 'area std.', 'smoothness std.', 'compactness std.', 
                'concavity std.', 'concave points std.', 'symmetry std.', 'fratal dimension std.']
    colNamesExt = ['radius extreme mean', 'texture extreme mean', 'perimeter extreme mean', 'area extreme mean', 
                'smoothness extreme mean', 'compactness extreme mean','concavity extreme mean', 'concave points extreme mean', 
                'symmetry extreme mean', 'fractal dimension extreme mean']
    colNamesOther = ['outcome', 'time', 'tumor size', 'lymph node status']

    return colNamesMeans, colNamesStd, colNamesExt, colNamesOther