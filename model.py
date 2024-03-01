import pandas as pd
import numpy as np
import pickle

data = pd.read_excel("iris.xls")

from sklearn.model_selection import train_test_split

x=data.drop(['Classification'],axis=1)
y=data.drop(['SL', 'SW', 'PL', 'PW'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#fitting model

model.fit(x_train,y_train)

#saving the model to disk

pickle.dump(model,open('model.pkl','wb'))

#pickle for converting to byte stream.serialising and deserialization















