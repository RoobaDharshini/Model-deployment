import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv('iris.csv')

print(df.head())

x=df.iloc[:,:4]
y=df['Class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(y_train.shape)

std=StandardScaler()
x_train=std.fit_transform(x_train)
x_test=std.transform(x_test)

rf_model=RandomForestClassifier()
rf_model.fit(x_train,y_train)

# making pickel file
import pickle

pickle.dump(rf_model,open('model.pkl','wb'))




