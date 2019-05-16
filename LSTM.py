import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

X = read_csv('Training_25jours_st.csv',usecols=[2],engine='python')
Y = read_csv('Training_25jours_st.csv',usecols=[3],engine='python')

X=np.array(X)
X_Data=X.reshape(-1,1) 

Y=np.array(Y)
Y_Data=Y.reshape(-1,1)

df=np.stack((X_Data,Y_Data),axis=1)

shift_days = 2
shift_steps = shift_days *4

df_targets = np.roll(df,-shift_steps)

x_data =df[:]
y_data = df_targets
y_data=y_data
num_data = len(x_data)
train_split = 0.9

num_train = int(train_split * num_data)
print("num_train",num_train)


num_test = num_data - num_train
print("num_test",num_test)
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
print("x length train and test",len(x_train) , len(x_test))
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
#len(y_train) + len(y_test)
num_input_signals = x_data[0].size
print("x number of inputs ",num_input_signals,"len test y",len(y_test))
num_label_signals = y_data.shape[1] #ici probleme
print("num label y",num_label_signals)
print("Min:", np.min(x_train))
print("Max:", np.max(x_train))

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()



x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, 2))
x_test_scaled = x_scaler.transform(x_test.reshape(-1, 2))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 2))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 2))

print("num_label_signals" ,num_label_signals)
print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))

print("Min y:", np.min(y_train_scaled),np.min(y_train))
print("Max y:", np.max(y_train_scaled),np.max(y_train))


def batch_generator(batch_size, sequence_length):

    Generator function for creating random batches of training-data.

    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_input_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_label_signals)
        #print("tr",y_shape)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        # Fill the batch with random sequences of data.

       

        for i in range(batch_size):
            idx = int((random.randint(1,5519))*20)
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

x_train_scaled[0:19].shape


import random
batch_size =  # add batch length 
sequence_length = # add sequence length 
print("sequence length", len(x_train))
print ("y len",len(y_train))

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

x_batch, y_batch = next(generator)


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

model = Sequential()
model.add(LSTM (4,input_shape=(None, num_input_signals,),return_sequences=True))
#model.add(LSTM(4, batch_input_shape=(256, x_train_scaled.shape, x_train_scaled.shape), stateful=True))
model.add(Dense(2))
   
model.add(Dense(num_label_signals, activation='sigmoid'))
if False:
    from tensorflow.python.keras.initializers import RandomUniform
    init = RandomUniform(minval=-0.05, maxval=0.05)
    model.add(Dense(num_label_signals,
                    activation='linear',
                    kernel_initializer=init))
optimizer = RMSprop(lr=1e-5)
#model.compile(optimizer='rmsprop',
#              loss='mse')
#model.compile(loss=loss_mse_warmup, optimizer=optimizer)

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
path_checkpoint = '2hedi_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)



callback_tensorboard = TensorBoard(log_dir='./2hedi_logs/',
                                   histogram_freq=0,
                                  write_graph=True)
        
#callbacks.TensorBoard(log_dir='./2hedi_logs/', histogram_freq=0,write_graph=True, write_images=True)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


model.fit_generator(generator=generator,
                    epochs= # add epochs,
                    steps_per_epoch=# add steps per epoch,
                    validation_data=validation_data,
                    callbacks=callbacks)

