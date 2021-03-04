import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

trainfile = pd.read_csv("train.csv")
testfile = pd.read_csv("test.csv")
samplefile = pd.read_csv("sample.csv")

xtrain = trainfile.loc[:, 'x1':'x10']
ytrain = trainfile.loc[:,'y']
xtest  = testfile.loc[:, 'x1':'x10']

#print(trainfile.head())
#print(ytrain.head())
#print(xtrain.head())

model = LinearRegression().fit(xtrain, ytrain)
ypredraw = model.predict(xtest)

ypred = pd.DataFrame(ypredraw, columns=['y'])
ypred.index += 10000

ypred.to_csv("result.csv")
#Don't forget to add Id manually