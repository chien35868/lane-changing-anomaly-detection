import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import f1_score
from sklearn import preprocessing
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
import tensorflow as tf 
from keras import optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score



# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

anomaly_blocked_df = pd.read_csv('../trajectory/ghost_vehicle_attack_2/with_attack/anomaly_blocked_attack.csv')
anomaly_blocked_df = anomaly_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
                                "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
                                "self_position", "self_speed", "self_acceleration"]]
anomaly_blocked = anomaly_blocked_df.values.tolist()
anomaly_blocked = np.array(anomaly_blocked)


anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]//10, 120))
#remove the data that has nan in it
delete_list = [] 
for i in range(anomaly_blocked.shape[0]):
    if np.isnan(anomaly_blocked[i]).any():
        delete_list.append(i)
anomaly_blocked = np.delete(anomaly_blocked, delete_list, axis=0)       
print(anomaly_blocked.shape)

anomaly_blocked_true_label = [1]*anomaly_blocked.shape[0]
anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]*10, 12))
# label 1 for anomaly_blocked
anomaly_blocked_label = [1]*anomaly_blocked.shape[0]
#-------------------------------------------------------------------------------#

normal_blocked_df = pd.read_csv('../trajectory/ghost_vehicle_attack_2/with_attack/normal_blocked_attack.csv')
normal_blocked_df = normal_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
                                "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
                                "self_position", "self_speed", "self_acceleration"]]
normal_blocked = normal_blocked_df.values.tolist()
normal_blocked = np.array(normal_blocked)

print(normal_blocked.shape)
print(normal_blocked.shape[0])
# normal_blocked = normal_blocked[:4020]
# normal_blocked = preprocessing.StandardScaler().fit(normal_blocked).transform(normal_blocked)
normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]//10, 120))
print(normal_blocked.shape)
#remove the data that has nan in it
delete_list = [] 
for i in range(normal_blocked.shape[0]):
    if np.isnan(normal_blocked[i]).any():
        delete_list.append(i)
normal_blocked = np.delete(normal_blocked, delete_list, axis=0)       
print(normal_blocked.shape)

normal_blocked_true_label = [0]*normal_blocked.shape[0]
normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]*10, 12))
# label 0 for normal_blocked
normal_blocked_label = [0]*normal_blocked.shape[0]

#-------------------------------------------------------------------#

# lane_keep_df = pd.read_csv('../trajectory/ghost_vehicle_attack/with_attack/lane_keep.csv')

# lane_keep = lane_keep_df.values.tolist()
# lane_keep = np.array(lane_keep)
# lane_keep = lane_keep[:12000]
# print(lane_keep.shape)
# print(lane_keep.shape[0])

# lane_keep = np.reshape(lane_keep, (lane_keep.shape[0]//10, 10, 12))
# print(lane_keep.shape)
# #remove the data that has nan in it
# delete_list = [] 
# for i in range(lane_keep.shape[0]):
#     if np.isnan(lane_keep[i]).any():
#         delete_list.append(i)
# lane_keep = np.delete(lane_keep, delete_list, axis=0)       
# print(lane_keep.shape)

# # label 2 for lane keep
# lane_keep_label = [2]*lane_keep.shape[0]

#-------------------------------------------------------------------#

# right_change_df = pd.read_csv('../trajectory/bigger_attack/right_turn_attack.csv')
# right_change = right_change_df.values.tolist()
# left_change_df = pd.read_csv('../trajectory/bigger_attack/left_turn_attack.csv')
# left_change = left_change_df.values.tolist()

# lane_change = right_change + left_change
# lane_change = np.array(lane_change)
# lane_change = np.reshape(lane_change, (lane_change.shape[0]//10, 10, 12))
# print(lane_change.shape)

# delete_list = [] 
# for i in range(lane_change.shape[0]):
#     if np.isnan(lane_change[i]).any():
#         delete_list.append(i)
# lane_change = np.delete(lane_change, delete_list, axis=0)       
# print(lane_change.shape)

# # label 1 for lane keep data
# lane_change_label = [1]*lane_change.shape[0]

#-------------------------------------------------------------------#

# anomaly_blocked_df = pd.read_csv('../trajectory/bigger_attack/anomaly_blocked_attack.csv')
# anomaly_blocked_df = anomaly_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
#                                 "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
#                                 "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
#                                 "self_position", "self_speed", "self_acceleration"]]
# anomaly_blocked = anomaly_blocked_df.values.tolist()
# anomaly_blocked = np.array(anomaly_blocked)

# anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]//10, 10, 12))
# #remove the data that has nan in it
# delete_list = [] 
# for i in range(anomaly_blocked.shape[0]):
#     if np.isnan(anomaly_blocked[i]).any():
#         delete_list.append(i)
# anomaly_blocked = np.delete(anomaly_blocked, delete_list, axis=0)       
# print(anomaly_blocked.shape)


# normal_blocked_df = pd.read_csv('../trajectory/bigger_attack/normal_blocked_attack.csv')
# normal_blocked_df = normal_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
#                                 "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
#                                 "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
#                                 "self_position", "self_speed", "self_acceleration"]]
# normal_blocked = normal_blocked_df.values.tolist()
# normal_blocked = np.array(normal_blocked)

# print(normal_blocked.shape)
# print(normal_blocked.shape[0])
# # normal_blocked = normal_blocked[:4020]
# normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]//10, 10, 12))
# print(normal_blocked.shape)
# #remove the data that has nan in it
# delete_list = [] 
# for i in range(normal_blocked.shape[0]):
#     if np.isnan(normal_blocked[i]).any():
#         delete_list.append(i)
# normal_blocked = np.delete(normal_blocked, delete_list, axis=0)       
# print(normal_blocked.shape)

# blocked = np.concatenate(([anomaly_blocked, normal_blocked]), axis=0)

# blocked_label = [0]*blocked.shape[0]


#-------------------------------------------------------------------#

from sklearn.model_selection import train_test_split
# X = np.concatenate(([anomaly_blocked, normal_blocked, lane_keep]), axis=0)
X = np.concatenate(([anomaly_blocked, normal_blocked]), axis=0)
# X = np.reshape(X, (X.shape[0]*10, 12))
X = preprocessing.StandardScaler().fit(X).transform(X)
# X = np.reshape(X, (X.shape[0]//10, 10, 12))
# y = np.concatenate(([anomaly_blocked_label, normal_blocked_label, lane_keep_label]), axis=0)
y = np.concatenate(([anomaly_blocked_label, normal_blocked_label]), axis=0)
y_true = np.concatenate(([anomaly_blocked_true_label, normal_blocked_true_label]), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


import tensorflow as tf 
from keras import optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


with open("./model/dnn.pkl", 'rb') as f:
    model = pickle.load(f)

# model = Sequential()


# model.add(LSTM(20, input_shape=(12, 1), return_sequences=True))
# model.add(Dropout(0.2))
# # model.add(LSTM(50, return_sequences=True))
# # model.add(Dropout(0.2))
# # model.add(LSTM(100, return_sequences=True))
# # model.add(Dropout(0.2))
# model.add(LSTM(200))
# model.add(Dropout(0.2))
# # model.add(Dense(2, activation="sigmoid"))
# model.add(Dense(2, activation="sigmoid"))

# # opt = tf.keras.optimizers.SGD(learning_rate=0.001)
# # opt = tf.keras.optimizers.Nadam(learning_rate=0.0005) ## nice optimizer
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# # opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
# # opt = tf.keras.optimizers.Adagrad(learning_rate=0.001)
# class_weight = {0: 0.2, 1: 1}
# model.compile(loss='SparseCategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])



print(model.evaluate(X, y))
y_pred = []
# print(model.predict(X_test))
for i in model.predict(X):
    # if i >= 0.5:
    #     y_pred.append(1)
    # else:
    #     y_pred.append(0) 
    y_pred.append(np.argmax(i))

y_pred_true = []

cnt = 0
cnt_0 = 0
cnt_1 = 0
# print(y_pred[:400])
for i in range(len(y_pred)):
    cnt += 1
    if y_pred[i] == 0:
        cnt_0 += 1
    else:
        cnt_1 += 1
    # print(cnt)
    if cnt == 10:
        if cnt_0 >= cnt_1:
            y_pred_true.append(0)
        else:
            y_pred_true.append(1)
        cnt = 0
        cnt_0 = 0
        cnt_1 = 0


# print("rnn accuracy:", accuracy_score(y, y_pred))
print("dnn f1 score:", f1_score(y_true, y_pred_true, average='macro'))
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_true).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")

