import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.metrics import f1_score
from sklearn import preprocessing

import pandas as pd
# # add header to csv file
# df = pd.read_csv('filename.csv', header=None)
# header = ['column1', 'column2', 'column3', 'column4', 'column5']
# df.columns = header
# df.to_csv('new_filename.csv', index=False)


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


input_timestep = 5


anomaly_blocked_df = pd.read_csv('../trajectory/ramp/ghost_vehicle_attack/with_attack/anomaly_blocked_attack.csv')
# header = ["vid", "time","position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
#                                     "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
#                                     "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
#                                     "self_position", "self_speed", "self_acceleration", "blocking_vid"]
# anomaly_blocked_df.columns = header                          
anomaly_blocked_df = anomaly_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
anomaly_blocked = anomaly_blocked_df.values.tolist()
anomaly_blocked = np.array(anomaly_blocked)

anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]//input_timestep, 60))
print(anomaly_blocked.shape)
#remove the data that has nan in it
delete_list = [] 
for i in range(anomaly_blocked.shape[0]):
    if np.isnan(anomaly_blocked[i]).any():
        # print(np.where(np.isnan(anomaly_blocked[i]).any()))
        delete_list.append(i)
anomaly_blocked = np.delete(anomaly_blocked, delete_list, axis=0)       

# anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]*(10/input_timestep), input_timestep, 12))
print(anomaly_blocked.shape)
# label 1 for anomaly_blocked
anomaly_blocked_label = [1]*anomaly_blocked.shape[0]
#-------------------------------------------------------------------------------#

normal_blocked_df = pd.read_csv('../trajectory/ramp/ghost_vehicle_attack/with_attack/normal_blocked_attack.csv')
# normal_blocked_df.columns = header 
normal_blocked_df = normal_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
normal_blocked = normal_blocked_df.values.tolist()
normal_blocked = np.array(normal_blocked)


# normal_blocked = normal_blocked[:4020]
# normal_blocked = preprocessing.StandardScaler().fit(normal_blocked).transform(normal_blocked)
normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]//input_timestep, 60))
print(normal_blocked.shape)
#remove the data that has nan in it
delete_list = [] 
for i in range(normal_blocked.shape[0]):
    if np.isnan(normal_blocked[i]).any():
        delete_list.append(i)
normal_blocked = np.delete(normal_blocked, delete_list, axis=0)       

# normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]*(10/input_timestep), input_timestep, 12))
# label 0 for normal_blocked
normal_blocked_label = [0]*normal_blocked.shape[0]
print(normal_blocked.shape)

#--------------------------------------------------------------------------------------#

from sklearn.model_selection import train_test_split
# X = np.concatenate(([anomaly_blocked, normal_blocked, lane_keep]), axis=0)
X = np.concatenate(([anomaly_blocked, normal_blocked]), axis=0)
X = np.reshape(X, (X.shape[0]*input_timestep, 12))
X = preprocessing.StandardScaler().fit(X).transform(X)
X = np.reshape(X, (X.shape[0]//input_timestep, 60))

# y = np.concatenate(([anomaly_blocked_label, normal_blocked_label, lane_keep_label]), axis=0)
y = np.concatenate(([anomaly_blocked_label, normal_blocked_label]), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
import tensorflow as tf 
from keras import optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


model = Sequential()
model.add(Dense(units=20,  activation="sigmoid"))
model.add(Dropout(0.1))
model.add(Dense(units=100,  activation="sigmoid"))
model.add(Dropout(0.1))
model.add(Dense(units=200,  activation="sigmoid"))
# model.add(Dropout(0.1))
# model.add(Dense(units=500,  activation="sigmoid"))

model.add(Dropout(0.1))
model.add(Dense(units=2,  activation="sigmoid"))

opt = tf.keras.optimizers.Nadam(learning_rate=0.01) ## nice optimizer
class_weight = {0: 1, 1: 1}
model.compile(loss='SparseCategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])


history = model.fit(X, y, verbose = 1, epochs = 200, validation_data=(X_test, y_test), class_weight=class_weight)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(model.evaluate(X_test, y_test))
y_pred = []
# print(model.predict(X_test))
for i in model.predict(X_test):
    y_pred.append(np.argmax(i))
print(y_pred[:100])


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
# print("f1 score: ", f1_score(y_test, y_pred, average='macro'))

print(f1_score(y_test, y_pred, average='macro'))

print("anomaly blocked cases: ", len(anomaly_blocked_label))
print("normal blocked cases: ", len(normal_blocked_label))
# print("lane keep cases: ", len(lane_keep_label))

#----------------------------------------------------------------------#

with open('./model/rnn.pkl','wb') as f:
    pickle.dump(model,f)

