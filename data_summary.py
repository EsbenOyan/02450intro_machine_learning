from data_preprocessing import *

def summarizeData(dfjoint: pd.DataFrame, printToConsole = True, save = False) -> pd.DataFrame:
    summarizedData = dfjoint.describe()
    if save:
        summarizedData.to_csv('data_summary.csv')
    if printToConsole:
        print(summarizedData)

    return summarizedData


if __name__ ==  '__main__':
    dfjoint, dfRec, dfClas = dataPreprocess()
    summarizeData(dfjoint, save = True)