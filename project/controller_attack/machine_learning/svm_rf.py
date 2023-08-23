import numpy as np
import pandas as pd
from sklearn import svm
import pickle
import time

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
input_shape=5

anomaly_blocked_df = pd.read_csv('../trajectory/roundabout_6/ghost_vehicle_attack/with_attack/anomaly_blocked_attack.csv')
anomaly_blocked_df = anomaly_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
anomaly_blocked = anomaly_blocked_df.values.tolist()
anomaly_blocked = np.array(anomaly_blocked)
# print(anomaly_blocked.shape)

anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]//input_shape, 12*input_shape))

#remove the data that has nan in it
delete_list = [] 
for i in range(anomaly_blocked.shape[0]):
    if np.isnan(anomaly_blocked[i]).any():
        delete_list.append(i)
anomaly_blocked = np.delete(anomaly_blocked, delete_list, axis=0)       
print(anomaly_blocked.shape)

# anomaly_blocked = np.reshape(anomaly_blocked, (anomaly_blocked.shape[0]*5, 24))

# label 1 for anomaly_blocked
anomaly_blocked_label = [1]*anomaly_blocked.shape[0]
#-------------------------------------------------------------------------------#

normal_blocked_df = pd.read_csv('../trajectory/roundabout_6/ghost_vehicle_attack/with_attack/normal_blocked_attack.csv')
normal_blocked_df = normal_blocked_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
normal_blocked = normal_blocked_df.values.tolist()
normal_blocked = np.array(normal_blocked)


# normal_blocked = normal_blocked[:5000]
normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]//input_shape, 12*input_shape))

#remove the data that has nan in it
delete_list = [] 
for i in range(normal_blocked.shape[0]):
    if np.isnan(normal_blocked[i]).any():
        delete_list.append(i)
normal_blocked = np.delete(normal_blocked, delete_list, axis=0)       
print(normal_blocked.shape)

# normal_blocked = np.reshape(normal_blocked, (normal_blocked.shape[0]*5, 24))

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
from sklearn.utils import shuffle
X = np.concatenate(([anomaly_blocked, normal_blocked]), axis=0)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = np.concatenate(([anomaly_blocked_label, normal_blocked_label]), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#------------------------------------------------------------#

clf = svm.SVC()
print ("training_time_start:", time.time() )
clf.fit(X_train, y_train)
print ("training_time_end:", time.time() )

print("testing_data length:", len(X_test))
print ("testing_time_start:", time.time() )
y_pred = []
y_pred = clf.predict(X_test)
print ("testing_time_end:", time.time() )
# for i in clf.predict(X_test):
#     if i > 0.5:
#         y_pred.append(1)
#     else:
#         y_pred.append(0)
# print(y_pred[:400])
print("svm accuracy:", accuracy_score(y_test, y_pred))
print("svm f1 score:", f1_score(y_test, y_pred, average='macro'))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")

with open('./model/svm.pkl','wb') as f:
    pickle.dump(clf,f)
#--------------------------------------------------------#


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=100, random_state=21)
print ("training_time_start:", time.time() )
clf.fit(X_train, y_train)
print ("training_time_end:", time.time() )
y_pred = []
print("testing_data length:", len(X_test))
print ("testing_time_start:", time.time() )
y_pred = clf.predict(X_test)
print ("testing_time_end:", time.time() )
# print(y_pred[:400])
print("rf accuracy:", accuracy_score(y_test, y_pred))
print("rf f1 score:", f1_score(y_test, y_pred, average='macro'))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")


with open('./model/rf.pkl','wb') as f:
    pickle.dump(clf,f)

#--------------------------------------------------------#
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=30, random_state=60)
clf.fit(X_train, y_train)
y_pred = []
y_pred = clf.predict(X_test)

print("adaboost accuracy:", accuracy_score(y_test, y_pred))
print("adaboost f1 score:", f1_score(y_test, y_pred, average='macro'))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: " , tp, "\n")

with open('./model/ada.pkl','wb') as f:
    pickle.dump(clf,f)