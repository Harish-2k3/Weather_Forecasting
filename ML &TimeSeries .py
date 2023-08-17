#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[ ]:


data = pd.read_csv('3363652.csv')


# In[ ]:


null_pct = data.apply(pd.isnull).sum()/data.shape[0]
null_pct


# In[ ]:


valid_col = data.columns[null_pct < .05]
valid_col


# In[ ]:


data = data[valid_col].copy()
data


# In[ ]:


print(data.head())


# In[ ]:


print(data.tail())


# In[ ]:


print(data.info())


# In[ ]:


print(data.describe())


# In[ ]:


print(data['SNWD'].value_counts().head())


# In[ ]:


print(data['TMIN'].unique())


# In[ ]:


print(data.dtypes)


# In[ ]:


data = data.dropna()


# # Visualizations 

# In[ ]:


plt.figure(figsize=(10, 6))
data['TMAX'].value_counts().head().plot(kind='bar')
plt.xlabel('Maximum temperature')
plt.ylabel('Count')
plt.title('Distribution of Categories')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
data['TMAX'].value_counts().tail(10).plot(kind='bar')
plt.xlabel('Maximum temperature')
plt.ylabel('Count')
plt.title('Distribution of Categories')
plt.show()


# In[ ]:


correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter3d(
    x=data['TMIN'],
    y=data['TMAX'],
    z=data['PRCP'],
    mode='markers',
    marker=dict(
        size=5,
        color=data['SNOW'],
        colorscale='Viridis',
        opacity=0.8
    )
))
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z'
))
fig.show()


# # KNN

# In[ ]:


#importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#read the data
data = pd.read_csv('3363652.csv',index_col="DATE")
data.head()

#find null percentage
null_pct = data.apply(pd.isnull).sum()/data.shape[0]
null_pct

#Select data
valid_col = data.columns[null_pct < .05]
valid_col

#copy data
data = data[valid_col].copy()
data

#Delete COlumns
column_to_remove = ['STATION','NAME','LATITUDE','LONGITUDE']
data = data.drop(column_to_remove, axis=1)
data

#Assign value
x = data.iloc[:,:-1]
x

y = data.iloc[:,-1]
y

#Find NULL value
data.isnull().sum()

#split data
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0) 

#Scaling the data
from sklearn.preprocessing import StandardScaler  
st_x = StandardScaler()  
x_train = st_x.fit_transform(x_train)  
x_test = st_x.transform(x_test)

#Change data type
index = pd.to_datetime(data.index)
index

#Find square root
import math
math.sqrt(len(y_test))

#KNN
classifier = KNeighborsClassifier(n_neighbors = 61, p=2, metric='euclidean')
classifier.fit(x_train, y_train)
classifier.score(x_test,y_test)
print(classifier.predict(x_test))
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# # ANN

# In[ ]:


#import the library
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt 

#read the data
df = pd.read_csv("3363652.csv" , index_col="DATE")

#Separate features and target variable
x = df.iloc[:,0:9].values
y = df.iloc[:,-1].values

#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0, test_size = 0.20)

#scale the feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#build the ANN model
model = Sequential()
model.add(Dense(units=6, activation = 'relu',input_dim=X_train.shape[1]))
model.add(Dense(units=6, activation = 'relu'))
model.add(Dense(units=1, activation = 'sigmoid'))

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

#Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Plot the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# # ARIMA

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# In[ ]:


data = pd.read_csv('3363652.csv')


# In[ ]:


data['DATE'] = pd.to_datetime(data['DATE'])


# In[ ]:


data.set_index('DATE', inplace=True)


# In[ ]:


null_pct = data.apply(pd.isnull).sum()/data.shape[0]
null_pct


# In[ ]:


valid_col = data.columns[null_pct < .05]
valid_col


# In[ ]:


data = data[valid_col].copy()


# In[ ]:


column_to_remove = ['STATION','NAME','LATITUDE','LONGITUDE','ELEVATION']
data = data.drop(column_to_remove, axis=1)


# In[ ]:


print('Shape of the data :',data.shape)
data


# In[ ]:


data['TMIN'].plot(figsize=(12,5))


# In[ ]:


from statsmodels.tsa.stattools import adfuller

def ad_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print("1. ADF:",dftest[0])
    print("2. P-Value:",dftest[1])
    print("3. Num of Lags:",dftest[2])
    print("4. Num of Observation used for ADF regression and critical value calculation:",dftest[3])
    print("5. Critical Value:")
    
    for key, val in dftest[4].items():
        print("\t",key,": ",val)


# In[ ]:


ad_test(data['TMIN'])


# In[ ]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


stepwise_fit = auto_arima(data['TMIN'], trace = True,
                          suppress_warnings = True)
stepwise_fit.summary()


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


print(data.shape)
train = data.iloc[:-30]
test = data.iloc[-30:]
print(train.shape,test.shape)


# In[ ]:


model = ARIMA(train['TMIN'],order=(4,0,3))
model = model.fit()
model.summary()


# In[ ]:


start = len(train)
end = len(train)+len(test)-1
pred = model.predict(start = start, end = end, type = 'levels')
print(pred)


# In[ ]:


pred.plot(legend = True)
test['TMIN'].plot(legend = True)


# In[ ]:


test['TMIN'].mean()


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(pred,test['TMIN']))
print(rmse)


# In[ ]:


model2 = ARIMA(data['TMIN'],order=(4,0,3))
model2=model2.fit()
data.tail()


# In[ ]:


index_feature_dates = pd.date_range(start = '2023-06-12', end = '2023-07-13')
print(index_feature_dates)


# In[ ]:


pred = model2.predict(start= len(data),end =len(data)+30,typ='levels').rename('ARIMA Predictions')
print(pred)


# In[ ]:


pred.plot(figsize=(12,6),legend = True)


# In[ ]:




