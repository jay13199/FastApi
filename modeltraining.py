import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
### Implement Random Forest classifier
classifier=RandomForestClassifier()

df=pd.read_csv('BankNote_Authentication.csv')
### Independent and Dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

classifier.fit(X_train,y_train)
### Check Accuracy
y_pred=classifier.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)

### Create a Pickle file using serialization 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()