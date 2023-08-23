import pandas as pd
import numpy as np

# import tensorflow as tf


lane_keep_df = pd.read_csv('../trajectory/lane_keep_attack.csv')
lane_keep = lane_keep_df.values.tolist()
lane_keep = np.array(lane_keep)


lane_keep = np.reshape(lane_keep, (lane_keep.shape[0]//10, 10, 12))
print(lane_keep.shape)
#remove the data that has nan in it
delete_list = [] 
for i in range(lane_keep.shape[0]):
    if np.isnan(lane_keep[i]).any():
        delete_list.append(i)
lane_keep = np.delete(lane_keep, delete_list, axis=0)       
print(lane_keep.shape)

# label 0 for lane keep data
lane_keep_label = [0]*lane_keep.shape[0]

#-------------------------------------------------------------------#

right_change_df = pd.read_csv('../trajectory/right_turn_attack.csv')
right_change = right_change_df.values.tolist()
left_change_df = pd.read_csv('../trajectory/left_turn_attack.csv')
left_change = left_change_df.values.tolist()

lane_change = right_change + left_change
lane_change = np.array(lane_change)
lane_change = np.reshape(lane_change, (lane_change.shape[0]//10, 10, 12))
print(lane_change.shape)

delete_list = [] 
for i in range(lane_change.shape[0]):
    if np.isnan(lane_change[i]).any():
        delete_list.append(i)
lane_change = np.delete(lane_change, delete_list, axis=0)       
print(lane_change.shape)

# label 1 for lane keep data
lane_change_label = [1]*lane_change.shape[0]

#-------------------------------------------------------------------#

from sklearn.model_selection import train_test_split
X = np.concatenate((lane_keep, lane_change), axis=0)
y = np.concatenate((lane_keep_label, lane_change_label), axis=0)
print(X.shape, y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#-------------------------------------------------------------------#

v_max, a_max = -9999, -9999
v_min, a_min = 9999, 9999
p_old, p_new = 0, 0
v_old, v_new = 0, 0
a_old, a_new = 0, 0
epsilon_p = 0.0001
epsilon_v = 0.0001
t = 0.1


error = 0
for x in X:
    for veh in [0, 3, 6, 9]: ## lead, side lead, side follow, self
        for step in x:
            if veh == 9:
                if step is x[0]:
                    p_old = step[veh]
                    v_old = step[veh+1]
                    a_old = step[veh+2]
                else:
                    p_old = p_new
                    v_old = v_new
                    a_old = a_new

                p_new = step[veh]
                v_new = step[veh+1]
                a_new = step[veh+2]

                v_max = max(v_old, v_new)
                v_min = min(v_old, v_new)
                a_max = max(a_old, a_new)
                a_min = min(a_old, a_new)

                condition_1 = (v_min*t + 0.5*a_min*(t**2) - epsilon_p <= p_new - p_old)
                condition_2 = (p_new - p_old <= v_max*t + 0.5*a_max*(t**2) + epsilon_p)
                condition_3 = ((a_min*t - epsilon_v <= v_new - v_old) and (v_new - v_old <= a_max*t +epsilon_v))

                if condition_1 and condition_2 and condition_3 == 0:
                    error += 1

            else:
                if step is x[0]:
                    p_old = step[veh]+step[9]
                    v_old = step[veh+1]+step[10]
                    a_old = step[veh+2]+step[11]
                else:
                    p_old = p_new
                    v_old = v_new
                    a_old = a_new

                p_new = step[veh]+step[9]
                v_new = step[veh+1]+step[10]
                a_new = step[veh+2]+step[11]

                v_max = max(v_old, v_new)
                v_min = min(v_old, v_new)
                a_max = max(a_old, a_new)
                a_min = min(a_old, a_new)

                condition_1 = (v_min*t + 0.5*a_min*(t**2) - epsilon_p <= p_new - p_old)
                condition_2 = (p_new - p_old <= v_max*t + 0.5*a_max*(t**2) + epsilon_p)
                condition_3 = ((a_min*t - epsilon_v <= v_new - v_old) and (v_new - v_old <= a_max*t +epsilon_v))

                if condition_1 and condition_2 and condition_3 == 0:
                    error += 1             

print(1 - (error/X.shape[0]))
