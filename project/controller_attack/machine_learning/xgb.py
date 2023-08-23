import numpy as np
import pandas as pd
import pickle
from sklearn import svm

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

anomaly_blocked_df = pd.read_csv('../trajectory/ghost_vehicle_attack/with_attack/anomaly_blocked_attack.csv')
anomaly_blocked_df = anomaly_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
                                "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
                                "self_position", "self_speed", "self_acceleration"]]
anomaly_blocked = anomaly_blocked_df.values.tolist()
anomaly_blocked = np.array(anomaly_blocked)
# print(anomaly_blocked.shape)

anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]//10, 120))

#remove the data that has nan in it
delete_list = [] 
for i in range(anomaly_blocked.shape[0]):
    if np.isnan(anomaly_blocked[i]).any():
        delete_list.append(i)
anomaly_blocked = np.delete(anomaly_blocked, delete_list, axis=0)       


anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]*5, 24))

# label 1 for anomaly_blocked
anomaly_blocked_label = [1]*anomaly_blocked.shape[0]
#-------------------------------------------------------------------------------#

normal_blocked_df = pd.read_csv('../trajectory/ghost_vehicle_attack/with_attack/normal_blocked_attack.csv')
normal_blocked_df = normal_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
                                "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
                                "self_position", "self_speed", "self_acceleration"]]
normal_blocked = normal_blocked_df.values.tolist()
normal_blocked = np.array(normal_blocked)


normal_blocked = normal_blocked[:5000]
normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]//10, 120))

#remove the data that has nan in it
delete_list = [] 
for i in range(normal_blocked.shape[0]):
    if np.isnan(normal_blocked[i]).any():
        delete_list.append(i)
normal_blocked = np.delete(normal_blocked, delete_list, axis=0)       


normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]*5, 24))

# label 0 for normal_blocked
normal_blocked_label = [0]*normal_blocked.shape[0]

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
from xgboost import XGBClassifier
print(anomaly_blocked.shape)
print(normal_blocked.shape)
X = np.concatenate(([anomaly_blocked, normal_blocked]), axis=0)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = np.concatenate(([anomaly_blocked_label, normal_blocked_label]), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)



xgboostModel = XGBClassifier(n_estimators=2000, learning_rate=0.2)
xgboostModel.fit(X_train, y_train)
y_pred = []
y_pred = xgboostModel.predict(X_test)
print("xgboost accuracy:", accuracy_score(y_test, y_pred))
print("xgboost f1 score:", f1_score(y_test, y_pred, average='macro'))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")

y_pred = xgboostModel.predict(X)
print("xgboost accuracy:", accuracy_score(y, y_pred))
print("xgboost f1 score:", f1_score(y, y_pred, average='macro'))
with open('./model/xgb.pkl','wb') as f:
    pickle.dump(xgboostModel,f)