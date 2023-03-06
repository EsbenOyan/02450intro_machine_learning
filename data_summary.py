from data_attributes import *

dfjoint, dfRec, dfClas = dataPreprocess()

def summarizeData(dfjoint: pd.DataFrame, printToConsole = True, save = False) -> pd.DataFrame:
    summarizedData = dfjoint.describe()
    if save:
        summarizedData.to_csv('data_summary.csv')
    if printToConsole:
        print(summarizedData)

    return summarizedData