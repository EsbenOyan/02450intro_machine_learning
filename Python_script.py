import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('./Data/Classification.data')

"""virker ikke

data2 = data
# Create a scaler object
scaler = MinMaxScaler()

# Normalize the second column of the DataFrame
data2.iloc[:,2:] = scaler.fit_transform(data.iloc[:, 2:].values.reshape(-1, 1))
"""
data_subset_mean = data.iloc[:, 2:12]

data_subset_mean = data_subset_mean.astype(int)

# Create a boxplot of the selected columns
plt.boxplot(data_subset_mean)
plt.title('Boxplot of means')
plt.xlabel('Means')
plt.ylabel('Value')
plt.show()
print(data_subset_mean.head())
print('work1')

data_subset_SD = data.iloc[:, 12:22]

data_subset_SD = data_subset_SD.astype(int)

# Create a boxplot of the selected columns
plt.boxplot(data_subset_SD)
plt.title('Boxplot of standard deviation')
plt.xlabel('Means')
plt.ylabel('Value')
plt.show()
print('work2')

data_subset_worst = data.iloc[:, 22:32]

data_subset_worst = data_subset_worst.astype(int)

# Create a boxplot of the selected columns
plt.boxplot(data_subset_worst)
plt.title('Boxplot of worst values')
plt.xlabel('Means')
plt.ylabel('Value')
plt.show()
print('work3')