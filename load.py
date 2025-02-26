import pandas as pd
import numpy as np

#Making pandas dataframe
url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"

df = pd.read_csv(url, sep=",", header=0, index_col=0)

#Converting to np
raw_data = df.values

#taking all columns except the last
cols = range(0, len(raw_data[0]) - 1)
X = raw_data[:, cols]
X[:, 4] = np.where(X[:, 4] == "Present", 1, 0)

#Getting attributenames
attributeNames = np.asarray(df.columns[cols])
classNames = np.array(['Negative', 'Positive'])
classLabels = raw_data[:, -1]

N, M = X.shape
C = len(classNames)