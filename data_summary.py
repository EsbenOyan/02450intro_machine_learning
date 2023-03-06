from data_preprocessing import *

def summarizeData(dfjoint: pd.DataFrame, printToConsole = True, save = False) -> pd.DataFrame:
    # summarizes and prints/saves the data depending on function arguments
    summarizedData = dfjoint.describe()
    if save:
        summarizedData.to_csv('data_summary.csv')
    if printToConsole:
        print(summarizedData)

    return summarizedData

# if you run this file itself this section will be run. If you import this file from another script this 
# will not be run
if __name__ ==  '__main__':
    dfjoint, dfRec, dfClas = dataPreprocess()
    summarizeData(dfjoint, save = True)