#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import sqrt

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Bidirectional, LSTM, concatenate
from tensorflow.keras.models import Model
# from sklearn import preprocessing
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# from tensorflow.keras import Input
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.models import load_model, Model

# import datetime
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from matplotlib import pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, BatchNormalization
# from keras.layers.convolutional import Conv1D    
# from keras.layers import InputLayer, Conv1D, Dense, Flatten, MaxPooling1D,Bidirectional

# from keras.callbacks import TensorBoard
# from keras.callbacks import ModelCheckpoint
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from tensorflow.keras.optimizers import SGD
# from matplotlib import pyplot

df_14, df_15, df_16, df_17, df_18, df_19, df_20 = [pd.read_csv(fr"../../../Taipei_{i}.csv") 
                                                   for i in range(14,21)]


# In[3]:


def get_X_and_Y(table, station_name = 'Banqiao'):
    table = table[table.SiteEngName =='Banqiao']
    features=table[['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5','RAINFALL', 'RH']]
    #print(f'The shape of the input table is {features.shape}')
    data = np.array(features.values.reshape((-1,10*1)))
    timestep=24
    x_build = []
    for i in range(data.shape[0] - timestep - timestep):
        x_build.append(data[i:i+timestep])
    train_x = np.array(x_build)
    #print(f'The shape of the input train_x is {train_x.shape}')


    data = np.array(table['PM2.5'].values.reshape((-1, 1)))
    y_build = []
    for i in range(timestep, data.shape[0] - timestep):
        y_build.append(data[i:i+timestep])
    train_y = np.array(y_build)
    print(f'The shape of the input train_y is {train_y.shape}')
    print('-'*50)
    return (train_x, train_y)

train14_x, train14_y = get_X_and_Y(df_14, station_name = 'Banqiao')
train15_x, train15_y = get_X_and_Y(df_15, station_name = 'Banqiao')
train16_x, train16_y = get_X_and_Y(df_16, station_name = 'Banqiao')
train17_x, train17_y = get_X_and_Y(df_17, station_name = 'Banqiao')
train18_x, train18_y = get_X_and_Y(df_18, station_name = 'Banqiao')
train19_x, train19_y = get_X_and_Y(df_19, station_name = 'Banqiao')
train20_x, train20_y = get_X_and_Y(df_20, station_name = 'Banqiao')


# In[16]:


train_X=np.concatenate((train14_x,train15_x,train16_x,train17_x,train18_x,train19_x),axis=0)
#print(train_X.shape)
train_y=np.concatenate((train14_y,train15_y,train16_y,train17_y,train18_y,train19_y),axis=0)
#print(train_y.shape)
test_X=train20_x;test_y=train20_y
#print(train20_x.shape,train20_y.shape)


from sklearn.model_selection import train_test_split
# create dataset
X, y = test_X,test_y

# split into train test sets
valid_x,test_x,valid_y,test_y = train_test_split(X, y, test_size=0.5)
print(valid_x.shape, valid_y.shape,test_x.shape, test_y.shape)

print("train_X.shape, train_y.shape, test_x.shape, test_y.shape") 
print(train_X.shape, train_y.shape, test_x.shape, test_y.shape) 

####################################
######################################


# Define the input layers
input1 = Input(shape=(train_X.shape[1], train_X.shape[2]))
x1 = Conv1D(filters=100, kernel_size=1, activation='relu')(input1)
x1 = MaxPooling1D(pool_size=1)(x1)
lstm1 = Bidirectional(LSTM(50, return_sequences=True))(x1)
output1 = Dense(50, activation='relu')(lstm1)
out=Dense(1, activation='relu')(output1)
###########################
model = Model(inputs=[input1], outputs=out)

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

model.summary()

# # Compile the model
# model.compile(loss='mse',optimizer='adam',metrics=['mae'])

# # Print model summary
# model.summary()


# In[2]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np
checkpoint_path = "Banqiao24.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback   = [
      EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]

#Train the model with the new callback
history = model.fit((train_X) ,
                    (train_y), 
                    validation_data=((valid_x),
                                    (valid_y)),
                    epochs =1, batch_size=19, verbose = 2, shuffle = True,
                    callbacks=[cp_callback])


# In[ ]:


model.load_weights(checkpoint_path)


# In[ ]:


scores = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', scores)
print('Test accuracy:', scores)
predict_ary = model.predict([test_x.astype("float")], batch_size = 190)
print (predict_ary.shape)
print (test_y.shape)
#print (X_valid.shape)
rmse_score20 = np.sqrt(np.mean(np.square(predict_ary - test_y.astype("float"))))
mae_score = np.mean(np.abs(predict_ary - test_y.astype("float")))
#mape_score = mean_absolute_percentage_error(y_valid_c.astype("float"),predict_ary)
#mae2 = mean_absolute_error(predict_ary, validation_Y[:-3])
print('this is rmse ',rmse_score20)
#print('this is mape ',mape_score)
print('this is mae ',mae_score)

y_pred_flat = predict_ary.reshape(-1)
y_true_flat = test_y.astype("float").reshape(-1)

# Calculate metrics
r2 = r2_score(y_true_flat, y_pred_flat)
mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat)
print('this is r2 ',r2)
print('this is mape ',mape)
# In[ ]:


# In[3]:


#################
import csv
data = [[24,rmse_score20,mae_score,mape,r2,"Banqiao"]]
file = open('Banqiao.csv', 'a+', newline ='')

# writing the data into the file
with file:    
    write = csv.writer(file)
    write.writerows(data)
dfa47= pd.read_csv("Banqiao.csv")
dfa47







# In[ ]:





# In[ ]:




