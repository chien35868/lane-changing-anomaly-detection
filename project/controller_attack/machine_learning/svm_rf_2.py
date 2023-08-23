import numpy as np
import pandas as pd
from sklearn import svm
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

input_length = 5

anomaly_blocked_df = pd.read_csv('../trajectory/ramp/ghost_vehicle_attack/with_attack/anomaly_blocked_attack.csv')
anomaly_blocked_df = anomaly_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
anomaly_blocked = anomaly_blocked_df.values.tolist()
anomaly_blocked = np.array(anomaly_blocked)
# print(anomaly_blocked.shape)



anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]//input_length, 12*input_length))

#remove the data that has nan in it
delete_list = [] 
for i in range(anomaly_blocked.shape[0]):
    if np.isnan(anomaly_blocked[i]).any():
        delete_list.append(i)
anomaly_blocked = np.delete(anomaly_blocked, delete_list, axis=0)       
print(anomaly_blocked.shape)

anomaly_blocked_true_label = [1]*anomaly_blocked.shape[0]
# anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]*5, 24))

# label 1 for anomaly_blocked
anomaly_blocked_label = [1]*anomaly_blocked.shape[0]
#-------------------------------------------------------------------------------#

normal_blocked_df = pd.read_csv('../trajectory/ramp/ghost_vehicle_attack/with_attack/normal_blocked_attack.csv')
normal_blocked_df = normal_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
normal_blocked = normal_blocked_df.values.tolist()
normal_blocked = np.array(normal_blocked)


normal_blocked = normal_blocked[:5000]
normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]//input_length, 12*input_length))

#remove the data that has nan in it
delete_list = [] 
for i in range(normal_blocked.shape[0]):
    if np.isnan(normal_blocked[i]).any():
        delete_list.append(i)
normal_blocked = np.delete(normal_blocked, delete_list, axis=0)       
print(normal_blocked.shape)

normal_blocked_true_label = [0]*normal_blocked.shape[0]
# normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]*5, 24))

# label 0 for normal_blocked
normal_blocked_label = [0]*normal_blocked.shape[0]
print(normal_blocked.shape)
#------------------------------------------------------------#

# lane_keep_df = pd.read_csv('../trajectory/ghost_vehicle_attack/with_attack/lane_keep.csv')

# lane_keep = lane_keep_df.values.tolist()
# lane_keep = np.array(lane_keep)
# lane_keep = lane_keep[:12000]
# print(lane_keep.shape)
# print(lane_keep.shape[0])

# lane_keep = np.reshape(lane_keep, (lane_keep.shape[0]//10, 120))
# print(lane_keep.shape)
# #remove the data that has nan in it
# delete_list = [] 
# for i in range(lane_keep.shape[0]):
#     if np.isnan(lane_keep[i]).any():
#         delete_list.append(i)
# lane_keep = np.delete(lane_keep, delete_list, axis=0)       
# print(lane_keep.shape)

# # label 2 for lane keep
# lane_keep_label = [1]*lane_keep.shape[0]

#------------------------------------------------------------#

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
X = np.concatenate(([anomaly_blocked, normal_blocked]), axis=0)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = np.concatenate(([anomaly_blocked_label, normal_blocked_label]), axis=0)
y_true = np.concatenate(([anomaly_blocked_true_label, normal_blocked_true_label]), axis=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#------------------------------------------------------------#

for model in ["svm", "rf", "ada"]:

    # load
    with open("./model/"+model+".pkl", 'rb') as f:
        clf = pickle.load(f)


    # print(X.shape, y.shape)
    y_pred = []
    y_pred = clf.predict(X)

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
        if cnt == 5:
            if cnt_0 >= cnt_1:
                y_pred_true.append(0)
            else:
                y_pred_true.append(1)
            cnt = 0
            cnt_0 = 0
            cnt_1 = 0



    print(model+" accuracy:", accuracy_score(y, y_pred))
    print(model+" f1 score:", f1_score(y, y_pred, average='micro'))
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")

    print(len(y_true), len(y_pred_true)) 
    print(model+ " accuracy:", accuracy_score(y_true, y_pred_true))
    print(model+ " f1 score:", f1_score(y_true, y_pred_true, average='macro'))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_true).ravel()
    print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")




