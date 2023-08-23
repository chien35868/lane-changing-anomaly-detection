import csv
import pandas
import re
import sys

# Case 1: Doesn’t want to change the lane 
# Case 2: Want to change the lane but blocked by normal vehicle
# Case 3: Want to change the lane but blocked by anomaly vehicle


pandas.options.display.max_rows = 1000
pandas.options.display.max_columns = 1000
lane_change_duration = 1
lane_change_time = 0
lane_keep_time = 10


header_list = ["vid", "time", "position", "speed", "acceleration", "right_lanechange_state", "left_lanechange_state", "lane",
                "lead_vid", "lead_position", "lead_speed", "lead_acceleration",
                "right_lead_vid", "right_lead_position", "right_lead_speed", "right_lead_acceleration",
                "right_follow_vid", "right_follow_position", "right_follow_speed", "right_follow_acceleration",
                "left_lead_vid", "left_lead_position", "left_lead_speed", "left_lead_acceleration",
                "left_follow_vid", "left_follow_position", "left_follow_speed", "left_follow_acceleration"]

# df = pandas.read_csv("./trajectory/bigger_attack/raw_trajectory.csv", on_bad_lines='skip')
df = pandas.read_csv("./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/raw_trajectory.csv", on_bad_lines='skip')


df.columns = header_list

df = df.sort_values(['vid', 'time'], ascending=[True, True])
df.to_csv("./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/raw_trajectory.csv", mode='w+', index=False, header=True)

#---------------------------------------------------------------------------#
## This part is to get left and right turn data(with blocking situation)

blocked_df = pandas.DataFrame()
left_follower_blocked_df = pandas.DataFrame()
left_leader_blocked_df = pandas.DataFrame()
right_follower_blocked_df = pandas.DataFrame()
right_leader_blocked_df = pandas.DataFrame()

left_leader_anomaly_blocked_df = pandas.DataFrame()
left_leader_normal_blocked_df = pandas.DataFrame()
left_follower_anomaly_blocked_df = pandas.DataFrame()
left_follower_normal_blocked_df = pandas.DataFrame()
right_leader_anomaly_blocked_df = pandas.DataFrame()
right_leader_normal_blocked_df = pandas.DataFrame()
right_follower_anomaly_blocked_df = pandas.DataFrame()
right_follower_normal_blocked_df = pandas.DataFrame()

cnt = 0
key = 0
total = len(df['vid'].unique())
for vid in df['vid'].unique():
    cnt+=1
    print("phase_1", cnt/total)
    

    target_df = df[df['vid'] == vid]
    lanechange_df = target_df[target_df["left_lanechange_state"].str.contains("speedGain")] #for freeway
    blocked_time = lanechange_df.time[lanechange_df["left_lanechange_state"].str.contains("blocked by left leader|blocked by left follower")]
    min_blocked_time = min(blocked_time) if len(blocked_time) != 0 else -1


    if min_blocked_time == -1: continue
    temp_df = target_df[(target_df["time"] <= min_blocked_time) & (target_df["time"] > min_blocked_time-1)]
    temp_df = temp_df[temp_df["position"]>400]
    if len(temp_df) == 10:
        if temp_df[temp_df["time"] == min_blocked_time]["left_lanechange_state"].str.contains("blocked by left leader").any():
            left_leader_blocked_df = pandas.concat([left_leader_blocked_df, temp_df])
            if temp_df[temp_df["time"] == min_blocked_time]["left_lead_vid"].str.contains("anomaly").any():
                left_leader_anomaly_blocked_df = pandas.concat([left_leader_anomaly_blocked_df, temp_df])
            if temp_df[temp_df["time"] == min_blocked_time]["left_lead_vid"].str.contains("normal").any():
                left_leader_normal_blocked_df = pandas.concat([left_leader_normal_blocked_df, temp_df])

        if temp_df[temp_df["time"] == min_blocked_time]["left_lanechange_state"].str.contains("blocked by left follower").any():
            left_follower_blocked_df = pandas.concat([left_follower_blocked_df, temp_df])

            if temp_df[temp_df["time"] == min_blocked_time]["left_follow_vid"].str.contains("anomaly").any():
                left_follower_anomaly_blocked_df = pandas.concat([left_follower_anomaly_blocked_df, temp_df])
            else:
                left_follower_normal_blocked_df = pandas.concat([left_follower_normal_blocked_df, temp_df])


dataframe_set = [left_leader_blocked_df, left_follower_blocked_df]
print(len(left_leader_blocked_df), len(left_follower_blocked_df), len(right_follower_blocked_df), len(right_follower_blocked_df))
store_key = 1
for dataframe in dataframe_set:
    store_key = 1 if dataframe.equals(left_leader_blocked_df) else 0
    dataframe["position_gap_lead"] = dataframe["lead_position"] - dataframe["position"]
    dataframe["speed_gap_lead"] = dataframe["lead_speed"] - dataframe["speed"]
    dataframe["acceleration_gap_lead"] = dataframe["lead_acceleration"] - dataframe["acceleration"]
    dataframe["position_gap_left_lead"] = dataframe["left_lead_position"] - dataframe["position"]
    dataframe["speed_gap_left_lead"] = dataframe["left_lead_speed"] - dataframe["speed"]
    dataframe["acceleration_gap_left_lead"] = dataframe["left_lead_acceleration"] - dataframe["acceleration"]
    dataframe["position_gap_left_follow"] = dataframe["left_follow_position"] - dataframe["position"]
    dataframe["speed_gap_left_follow"] = dataframe["left_follow_speed"] - dataframe["speed"]
    dataframe["acceleration_gap_left_follow"] = dataframe["left_follow_acceleration"] - dataframe["acceleration"]

    if dataframe.equals(left_leader_blocked_df):
        dataframe["blocking_vid"] = dataframe["left_lead_vid"] 
    else:
        dataframe["blocking_vid"] = dataframe["left_follow_vid"] 

    dataframe["self_position"] = dataframe["position"]
    dataframe["self_speed"] = dataframe["speed"]
    dataframe["self_acceleration"] = dataframe["acceleration"]

    dataframe = dataframe[["vid", "time","position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
                                "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
                                "self_position", "self_speed", "self_acceleration", "blocking_vid"]]

    if store_key == 1:
        dataframe.to_csv('./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/left_leader_blocked_attack.csv', mode='w+', index=False, header=True) 
    else:
        dataframe.to_csv('./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/left_follower_blocked_attack.csv', mode='w+', index=False, header=True)
    # dataframe.to_csv('./trajectory/bigger_attack/'+str(dataframe)+'_attack.csv', mode='w+', index=False, header=True)


## The code of right change blocking has not been modified yet, the left change part has been modified.

# cnt = 0
# for vid in df['vid'].unique():
#     cnt+=1
#     print("phase_2", cnt/total)

#     target_df = df[df['vid'] == vid]
    
#     blocked_time = target_df.time[target_df["right_lanechange_state"].str.contains("blocked by right leader|blocked by right follower")]
#     min_blocked_time = min(blocked_time) if len(blocked_time) != 0 else -1

#     if min_blocked_time == -1: continue
#     temp_df = target_df[(target_df["time"] <= min_blocked_time) & (target_df["time"] > min_blocked_time-1)]
#     temp_df = temp_df[temp_df["position"]>400]
#     if len(temp_df) == 10:
#         if temp_df["right_lanechange_state"].str.contains("blocked by right leader").any():
#             right_leader_blocked_df = pandas.concat([right_leader_blocked_df, temp_df])

#             if temp_df[temp_df["time"] == min_blocked_time]["right_lead_vid"].str.contains("anomaly").any():
#                 right_leader_anomaly_blocked_df = pandas.concat([right_leader_anomaly_blocked_df, temp_df])
#             else:
#                 right_leader_normal_blocked_df = pandas.concat([right_leader_normal_blocked_df, temp_df])

#         if temp_df["right_lanechange_state"].str.contains("blocked by right follower").any():
#             right_follower_blocked_df = pandas.concat([right_follower_blocked_df, temp_df])

#             if temp_df[temp_df["time"] == min_blocked_time]["right_follow_vid"].str.contains("anomaly").any():
#                 right_follower_anomaly_blocked_df = pandas.concat([right_follower_anomaly_blocked_df, temp_df])
#             else:
#                 right_follower_normal_blocked_df = pandas.concat([right_follower_normal_blocked_df, temp_df])


# dataframe_set = [right_leader_blocked_df, right_follower_blocked_df]
# for dataframe in dataframe_set:
#     store_key = 1 if dataframe.equals(right_leader_blocked_df) else 0
#     dataframe["position_gap_lead"] = dataframe["lead_position"] - dataframe["position"]
#     dataframe["speed_gap_lead"] = dataframe["lead_speed"] - dataframe["speed"]
#     dataframe["acceleration_gap_lead"] = dataframe["lead_acceleration"] - dataframe["acceleration"]
#     dataframe["position_gap_right_lead"] = dataframe["right_lead_position"] - dataframe["position"]
#     dataframe["speed_gap_right_lead"] = dataframe["right_lead_speed"] - dataframe["speed"]
#     dataframe["acceleration_gap_right_lead"] = dataframe["right_lead_acceleration"] - dataframe["acceleration"]
#     dataframe["position_gap_right_follow"] = dataframe["right_follow_position"] - dataframe["position"]
#     dataframe["speed_gap_right_follow"] = dataframe["right_follow_speed"] - dataframe["speed"]
#     dataframe["acceleration_gap_right_follow"] = dataframe["right_follow_acceleration"] - dataframe["acceleration"]

#     if dataframe.equals(right_leader_blocked_df) or dataframe.equals(right_leader_anomaly_blocked_df) or dataframe.equals(right_leader_normal_blocked_df):
#         dataframe["blocking_vid"] = dataframe["right_lead_vid"] 
#     else:
#         dataframe["blocking_vid"] = dataframe["right_follow_vid"]

#     dataframe["self_position"] = dataframe["position"]
#     dataframe["self_speed"] = dataframe["speed"]
#     dataframe["self_acceleration"] = dataframe["acceleration"]

#     dataframe = dataframe[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
#                                 "position_gap_right_lead", "speed_gap_right_lead", "acceleration_gap_right_lead",
#                                 "position_gap_right_follow", "speed_gap_right_follow", "acceleration_gap_right_follow",
#                                 "self_position", "self_speed", "self_acceleration", "blocking_vid"]]

#     if store_key == 1:
#         dataframe.to_csv('./trajectory/bigger_attack/right_leader_blocked_attack.csv', mode='w+', index=False, header=True) 
#     else:
#         dataframe.to_csv('./trajectory/bigger_attack/right_follower_blocked_attack.csv', mode='w+', index=False, header=True)


# dataframe.to_csv('./trajectory/bigger_attack/right_leader_blocked_attack.csv', mode='w+', index=False, header=True) 



#--------------------------------------------------------------------------------------#

## This part is combine the anomaly blocker dataframes and normal blokcers dataframes


# anomaly_blocked_df = pandas.DataFrame()
# normal_blocked_df = pandas.DataFrame()

# anomaly_blocked_df = pandas.concat([pandas.concat([left_leader_anomaly_blocked_df, left_follower_anomaly_blocked_df, right_leader_anomaly_blocked_df, right_follower_anomaly_blocked_df])])
# normal_blocked_df = pandas.concat([pandas.concat([left_leader_normal_blocked_df, left_follower_normal_blocked_df, right_leader_normal_blocked_df, right_follower_normal_blocked_df])])
# anomaly_blocked_df = pandas.concat([pandas.concat([left_leader_anomaly_blocked_df, left_follower_anomaly_blocked_df])])
# normal_blocked_df = pandas.concat([pandas.concat([left_leader_normal_blocked_df, left_follower_normal_blocked_df])])


dataframe_set = [left_leader_anomaly_blocked_df, left_follower_anomaly_blocked_df, left_leader_normal_blocked_df, left_follower_normal_blocked_df]

print(len(left_leader_anomaly_blocked_df.index))
print(len(left_follower_anomaly_blocked_df.index))
print(len(left_leader_normal_blocked_df.index))
print(len(left_follower_normal_blocked_df.index))

for dataframe in dataframe_set:
    store_key = 1 if dataframe.equals(left_leader_anomaly_blocked_df) or dataframe.equals(left_follower_anomaly_blocked_df) else 0

    dataframe["position_gap_lead"] = dataframe["lead_position"] - dataframe["position"]
    dataframe["speed_gap_lead"] = dataframe["lead_speed"] - dataframe["speed"]
    dataframe["acceleration_gap_lead"] = dataframe["lead_acceleration"] - dataframe["acceleration"]
    dataframe["position_gap_left_lead"] = dataframe["left_lead_position"] - dataframe["position"]
    dataframe["speed_gap_left_lead"] = dataframe["left_lead_speed"] - dataframe["speed"]
    dataframe["acceleration_gap_left_lead"] = dataframe["left_lead_acceleration"] - dataframe["acceleration"]
    dataframe["position_gap_left_follow"] = dataframe["left_follow_position"] - dataframe["position"]
    dataframe["speed_gap_left_follow"] = dataframe["left_follow_speed"] - dataframe["speed"]
    dataframe["acceleration_gap_left_follow"] = dataframe["left_follow_acceleration"] - dataframe["acceleration"]

    if dataframe.equals(left_leader_anomaly_blocked_df) or dataframe.equals(left_leader_normal_blocked_df):
        dataframe["blocking_vid"] = dataframe["left_lead_vid"] 
    else:
        dataframe["blocking_vid"] = dataframe["left_follow_vid"] 

    dataframe["self_position"] = dataframe["position"]
    dataframe["self_speed"] = dataframe["speed"]
    dataframe["self_acceleration"] = dataframe["acceleration"]

    dataframe = dataframe[["vid", "time","position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
                                "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
                                "self_position", "self_speed", "self_acceleration", "blocking_vid"]]
    if store_key == 1:
        dataframe.to_csv('./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/anomaly_blocked_attack.csv', mode='w+', index=False, header=True) 
    else:
        dataframe.to_csv('./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/normal_blocked_attack.csv', mode='w+', index=False, header=True) 
    

# anomaly_blocked_df = pandas.concat([pandas.concat([left_leader_anomaly_blocked_df, left_follower_anomaly_blocked_df])])
# normal_blocked_df = pandas.concat([pandas.concat([left_leader_normal_blocked_df, left_follower_normal_blocked_df])])

# anomaly_blocked_df.to_csv('./trajectory/bigger_attack/anomaly_blocked_attack.csv', mode='w+', index=False, header=True) 
# normal_blocked_df.to_csv('./trajectory/bigger_attack/normal_blocked_attack.csv', mode='w+', index=False, header=True) 


#--------------------------------------------------------------------------------------#

## This part is to get left and right turn data(without blocking situation)

# left_turn_df = pandas.DataFrame()
# right_turn_df = pandas.DataFrame()

# for vid in df['vid'].unique():
#     target_df = df[df['vid'] == vid]
  
#     ## diff = 1, left turn
#     if target_df.loc[target_df['lane'].diff() == 1].empty == 0:
#         lane_change_timelist = target_df.loc[target_df['lane'].diff() == 1]['time'].values
#         # print(target_df.loc[target_df['lane'].diff() == 1])
#         for time in lane_change_timelist:
#             temp_df = target_df[(target_df["time"] < time - lane_change_time) & (target_df["time"] >= time - lane_change_time - lane_change_duration)]
#             temp_df = temp_df[temp_df["position"]>400]
#             if len(temp_df) == 10:
#                 left_turn_df = pandas.concat([left_turn_df, temp_df])

#     ## diff = -1, right turn
#     if target_df.loc[target_df['lane'].diff() == -1].empty == 0:
#         lane_change_timelist = target_df.loc[target_df['lane'].diff() == -1]['time'].values  
#         # print(target_df.loc[target_df['lane'].diff() == -1])
#         for time in lane_change_timelist:
#             temp_df = target_df[(target_df["time"] < time - lane_change_time) & (target_df["time"] >= time - lane_change_time - lane_change_duration)]
#             temp_df = temp_df[temp_df["position"]>400]
#             if len(temp_df) == 10:
#                 right_turn_df = pandas.concat([right_turn_df, temp_df])

# # print(left_turn_df)
# # print(right_turn_df)
    
# left_turn_df["position_gap_lead"] = left_turn_df["lead_position"] - left_turn_df["position"]
# left_turn_df["speed_gap_lead"] = left_turn_df["lead_speed"] - left_turn_df["speed"]
# left_turn_df["acceleration_gap_lead"] = left_turn_df["lead_acceleration"] - left_turn_df["acceleration"]
# left_turn_df["position_gap_left_lead"] = left_turn_df["left_lead_position"] - left_turn_df["position"]
# left_turn_df["speed_gap_left_lead"] = left_turn_df["left_lead_speed"] - left_turn_df["speed"]
# left_turn_df["acceleration_gap_left_lead"] = left_turn_df["left_lead_acceleration"] - left_turn_df["acceleration"]
# left_turn_df["position_gap_left_follow"] = left_turn_df["left_follow_position"] - left_turn_df["position"]
# left_turn_df["speed_gap_left_follow"] = left_turn_df["left_follow_speed"] - left_turn_df["speed"]
# left_turn_df["acceleration_gap_left_follow"] = left_turn_df["left_follow_acceleration"] - left_turn_df["acceleration"]

# left_turn_df["self_position"] = left_turn_df["position"]
# left_turn_df["self_speed"] = left_turn_df["speed"]
# left_turn_df["self_acceleration"] = left_turn_df["acceleration"]


# left_turn_df = left_turn_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
#                             "position_gap_left_lead", "speed_gap_left_lead", "acceleration_gap_left_lead",
#                             "position_gap_left_follow", "speed_gap_left_follow", "acceleration_gap_left_follow",
#                             "self_position", "self_speed", "self_acceleration"]]
# # print(left_turn_df)


# right_turn_df["position_gap_lead"] = right_turn_df["lead_position"] - right_turn_df["position"]
# right_turn_df["speed_gap_lead"] = right_turn_df["lead_speed"] - right_turn_df["speed"]
# right_turn_df["acceleration_gap_lead"] = right_turn_df["lead_acceleration"] - right_turn_df["acceleration"]
# right_turn_df["position_gap_right_lead"] = right_turn_df["right_lead_position"] - right_turn_df["position"]
# right_turn_df["speed_gap_right_lead"] = right_turn_df["right_lead_speed"] - right_turn_df["speed"]
# right_turn_df["acceleration_gap_right_lead"] = right_turn_df["right_lead_acceleration"] - right_turn_df["acceleration"]
# right_turn_df["position_gap_right_follow"] = right_turn_df["right_follow_position"] - right_turn_df["position"]
# right_turn_df["speed_gap_right_follow"] = right_turn_df["right_follow_speed"] - right_turn_df["speed"]
# right_turn_df["acceleration_gap_right_follow"] = right_turn_df["right_follow_acceleration"] - right_turn_df["acceleration"]


# right_turn_df["self_position"] = right_turn_df["position"]
# right_turn_df["self_speed"] = right_turn_df["speed"]
# right_turn_df["self_acceleration"] = right_turn_df["acceleration"]


# right_turn_df = right_turn_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
#                                 "position_gap_right_lead", "speed_gap_right_lead", "acceleration_gap_right_lead",
#                                 "position_gap_right_follow", "speed_gap_right_follow", "acceleration_gap_right_follow",
#                                 "self_position", "self_speed", "self_acceleration"]]
# # print(right_turn_df)

# left_turn_df.to_csv('./trajectory/senpai_attack_test/left_turn_attack.csv', mode='w+', index=False, header=True)
# right_turn_df.to_csv('./trajectory/senpai_attack_test/right_turn_attack.csv', mode='w+', index=False, header=True)

# print("phase1 done")

# ##-----------------------------------------------------------------------------------------##

# This part is to get "lanekeep" data

left_turn_df = pandas.DataFrame()
right_turn_df = pandas.DataFrame()
lane_keep_df = pandas.DataFrame()

cnt = 0
for vid in df['vid'].unique():
    cnt+=1
    print("phase_2", cnt/total)

    target_df = df.loc[df['vid'] == vid]
    ## diff = 1, left turn
    if target_df.loc[target_df['lane'].diff() == 1].empty == 0:
        lane_change_timelist = target_df.loc[target_df['lane'].diff() == 1]['time'].values  
        # print(target_df.loc[target_df['lane'].diff() == 1])
        for time in lane_change_timelist:
            temp_df = target_df[(target_df["time"] < time - lane_keep_time) & (target_df["time"] >= time - lane_keep_time - lane_change_duration)]
            temp_df = temp_df[temp_df["position"]>400]
            if len(temp_df) == 10:
                left_turn_df = pandas.concat([left_turn_df, temp_df])

    ## diff = -1, right turn
    if target_df.loc[target_df['lane'].diff() == -1].empty == 0:
        lane_change_timelist = target_df.loc[target_df['lane'].diff() == -1]['time'].values  
        # print(target_df.loc[target_df['lane'].diff() == -1])
        for time in lane_change_timelist:
            temp_df = target_df[(target_df["time"] < time - lane_keep_time) & (target_df["time"] >= time - lane_keep_time - lane_change_duration)]
            temp_df = temp_df[temp_df["position"]>400]
            if len(temp_df) == 10:
                right_turn_df = pandas.concat([right_turn_df, temp_df])

# print(left_turn_df)
# print(right_turn_df)
    
left_turn_df["position_gap_lead"] = left_turn_df["lead_position"] - left_turn_df["position"]
left_turn_df["speed_gap_lead"] = left_turn_df["lead_speed"] - left_turn_df["speed"]
left_turn_df["acceleration_gap_lead"] = left_turn_df["lead_acceleration"] - left_turn_df["acceleration"]
left_turn_df["position_gap_side_lead"] = left_turn_df["left_lead_position"] - left_turn_df["position"]
left_turn_df["speed_gap_side_lead"] = left_turn_df["left_lead_speed"] - left_turn_df["speed"]
left_turn_df["acceleration_gap_side_lead"] = left_turn_df["left_lead_acceleration"] - left_turn_df["acceleration"]
left_turn_df["position_gap_side_follow"] = left_turn_df["left_follow_position"] - left_turn_df["position"]
left_turn_df["speed_gap_side_follow"] = left_turn_df["left_follow_speed"] - left_turn_df["speed"]
left_turn_df["acceleration_gap_side_follow"] = left_turn_df["left_follow_acceleration"] - left_turn_df["acceleration"]

left_turn_df["self_position"] = left_turn_df["position"]
left_turn_df["self_speed"] = left_turn_df["speed"]
left_turn_df["self_acceleration"] = left_turn_df["acceleration"]


left_turn_df = left_turn_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                            "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                            "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                            "self_position", "self_speed", "self_acceleration"]]
# print(left_turn_df)


right_turn_df["position_gap_lead"] = right_turn_df["lead_position"] - right_turn_df["position"]
right_turn_df["speed_gap_lead"] = right_turn_df["lead_speed"] - right_turn_df["speed"]
right_turn_df["acceleration_gap_lead"] = right_turn_df["lead_acceleration"] - right_turn_df["acceleration"]
right_turn_df["position_gap_side_lead"] = right_turn_df["right_lead_position"] - right_turn_df["position"]
right_turn_df["speed_gap_side_lead"] = right_turn_df["right_lead_speed"] - right_turn_df["speed"]
right_turn_df["acceleration_gap_side_lead"] = right_turn_df["right_lead_acceleration"] - right_turn_df["acceleration"]
right_turn_df["position_gap_side_follow"] = right_turn_df["right_follow_position"] - right_turn_df["position"]
right_turn_df["speed_gap_side_follow"] = right_turn_df["right_follow_speed"] - right_turn_df["speed"]
right_turn_df["acceleration_gap_side_follow"] = right_turn_df["right_follow_acceleration"] - right_turn_df["acceleration"]


right_turn_df["self_position"] = right_turn_df["position"]
right_turn_df["self_speed"] = right_turn_df["speed"]
right_turn_df["self_acceleration"] = right_turn_df["acceleration"]


right_turn_df = right_turn_df[["position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                "self_position", "self_speed", "self_acceleration"]]
# print(right_turn_df)

lane_keep_df = pandas.concat([lane_keep_df, left_turn_df])
lane_keep_df = pandas.concat([lane_keep_df, right_turn_df])

lane_keep_df.to_csv('./trajectory/roundabout_2/ghost_vehicle_attack/with_attack/lane_keep.csv', mode='w+', index=False, header=True)
# print(lane_keep_df)


##-----------------------------------------------------------------------------------------##





##-----------------------------------------------------------------------------------------##
## The part below is the work I had done before, and it is not related to the work in the upper section

# collision_vehicle = []
# collision_record = open('./trajectory/lc_collision_record.txt', 'r')
# for line in collision_record.readlines():
#     line = line.split("Teleporting vehicle ",1)[-1].split()[0]
#     line = line.replace("'", "")
#     line = line.replace(";", "")
#     collision_vehicle.append(line)

# c_v = pandas.DataFrame()
# for vid in collision_vehicle:
#     c_v = pandas.concat([c_v, df[df['vid'].str.contains(vid)][-20:]])
# c_v.to_csv('./trajectory/collision_with_attack.csv', mode='w+', index=False, header=True)
