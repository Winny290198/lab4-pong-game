import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('pongdata.csv').dropna()

x = data.iloc[:,:4]
y = data.iloc[:,5]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.25, random_state=0)
model = LinearRegression()
model.fit(xtrain, ytrain)
ymodel = model.predict(xtest)

from joblib import dump
dump(model, 'mymodel.joblib')