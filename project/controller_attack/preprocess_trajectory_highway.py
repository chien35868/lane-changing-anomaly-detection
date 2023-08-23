import csv
import pandas
import re
import sys

# Case 1: Doesn’t want to change the lane 
# Case 2: Want to change the lane but blocked by normal vehicle
# Case 3: Want to change the lane but blocked by anomaly vehicle


pandas.options.display.max_rows = 1000
pandas.options.display.max_columns = 1000
lane_change_duration = 0.5
lane_change_time = 0
lane_keep_time = 10


header_list = ["vid", "time", "position", "speed", "acceleration", "right_lanechange_state", "left_lanechange_state", "lane",
                "lead_vid", "lead_position", "lead_speed", "lead_acceleration",
                "right_lead_vid", "right_lead_position", "right_lead_speed", "right_lead_acceleration",
                "right_follow_vid", "right_follow_position", "right_follow_speed", "right_follow_acceleration",
                "left_lead_vid", "left_lead_position", "left_lead_speed", "left_lead_acceleration",
                "left_follow_vid", "left_follow_position", "left_follow_speed", "left_follow_acceleration", "lane_id"]

# df = pandas.read_csv("./trajectory/bigger_attack/raw_trajectory.csv", on_bad_lines='skip')
df = pandas.read_csv("./trajectory/ramp/acceleration_vehicle_attack/with_attack/raw_trajectory.csv", on_bad_lines='skip')
df.columns = header_list
df = df.sort_values(['vid', 'time'], ascending=[True, True])
df.to_csv("./trajectory/ramp/acceleration_vehicle_attack/with_attack/raw_trajectory.csv", mode='w+', index=False, header=True)

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
    print("phase_left", cnt/total)
    

    target_df = df[df['vid'] == vid]
    lanechange_df = target_df[(target_df["left_lanechange_state"].apply(lambda x: "'left'" in x))] 
    # print(lanechange_df["vid"], lanechange_df["time"])
    # lanechange_df = target_df[target_df["left_lanechange_state"].str.contains("speedgain")] #for freeway
    # print(lanechange_df["left_lanechange_state"])

    blocked_time = lanechange_df.time[lanechange_df["left_lanechange_state"].str.contains("blocked by left leader|blocked by left follower")]
    min_blocked_time = min(blocked_time) if len(blocked_time) != 0 else -1


    # print(lanechange_df["vid"], len(lanechange_df))

    if min_blocked_time == -1: continue
    temp_df = target_df[(target_df["time"] <= min_blocked_time) & (target_df["time"] > min_blocked_time-lane_change_duration)]
    # print(temp_df.time)
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



## The code of right change blocking has not been modified yet, the left change part has been modified.


cnt = 0
key = 0
total = len(df['vid'].unique())
for vid in df['vid'].unique():
    cnt+=1
    print("phase_right", cnt/total)
    

    target_df = df[df['vid'] == vid]


    lanechange_df = target_df[(target_df["right_lanechange_state"].apply(lambda x: "'right'" in x))] #for ramp

    # print(len(lanechange_df))
    blocked_time = lanechange_df.time[lanechange_df["right_lanechange_state"].str.contains("blocked by right leader|blocked by right follower")]
    min_blocked_time = min(blocked_time) if len(blocked_time) != 0 else -1


    if min_blocked_time == -1: continue
    temp_df = target_df[(target_df["time"] <= min_blocked_time) & (target_df["time"] > min_blocked_time-lane_change_duration)]
    # temp_df = temp_df[temp_df["position"]>400]
    if len(temp_df) == 10:
        if temp_df[temp_df["time"] == min_blocked_time]["right_lanechange_state"].str.contains("blocked by right leader").any():
            right_leader_blocked_df = pandas.concat([right_leader_blocked_df, temp_df])
            if temp_df[temp_df["time"] == min_blocked_time]["right_lead_vid"].str.contains("anomaly").any():
                right_leader_anomaly_blocked_df = pandas.concat([right_leader_anomaly_blocked_df, temp_df])
            if temp_df[temp_df["time"] == min_blocked_time]["right_lead_vid"].str.contains("normal").any():
                right_leader_normal_blocked_df = pandas.concat([right_leader_normal_blocked_df, temp_df])

        if temp_df[temp_df["time"] == min_blocked_time]["right_lanechange_state"].str.contains("blocked by right follower").any():
            right_follower_blocked_df = pandas.concat([right_follower_blocked_df, temp_df])

            if temp_df[temp_df["time"] == min_blocked_time]["right_follow_vid"].str.contains("anomaly").any():
                right_follower_anomaly_blocked_df = pandas.concat([right_follower_anomaly_blocked_df, temp_df])
            else:
                right_follower_normal_blocked_df = pandas.concat([right_follower_normal_blocked_df, temp_df])


print(len(left_leader_anomaly_blocked_df), len(left_follower_anomaly_blocked_df), len(left_leader_normal_blocked_df),len(left_follower_normal_blocked_df),
len(right_leader_anomaly_blocked_df),len(right_follower_anomaly_blocked_df),len(right_leader_normal_blocked_df),len(right_follower_normal_blocked_df))
#--------------------------------------------------------------------------------------#



# dataframe_set = [left_leader_anomaly_blocked_df, left_follower_anomaly_blocked_df, left_leader_normal_blocked_df, left_follower_normal_blocked_df,
#                 right_leader_anomaly_blocked_df, right_follower_anomaly_blocked_df, right_leader_normal_blocked_df, right_follower_normal_blocked_df]
dataframe_set = [right_leader_anomaly_blocked_df, right_leader_normal_blocked_df, right_follower_normal_blocked_df]

anomaly_blocked_df = pandas.DataFrame()
normal_blocked_df = pandas.DataFrame()

store_key = -1

for dataframe in dataframe_set:
    if dataframe.equals(left_leader_anomaly_blocked_df) or dataframe.equals(left_follower_anomaly_blocked_df) or dataframe.equals(right_leader_anomaly_blocked_df) or dataframe.equals(right_follower_anomaly_blocked_df):
        store_key = 0
    else:
        store_key = 1
    print(dataframe.columns)
    if dataframe.equals(left_leader_anomaly_blocked_df) or dataframe.equals(left_follower_anomaly_blocked_df) or dataframe.equals(left_leader_normal_blocked_df) or dataframe.equals(left_follower_normal_blocked_df):
        dataframe["position_gap_lead"] = dataframe["lead_position"] - dataframe["position"]
        dataframe["speed_gap_lead"] = dataframe["lead_speed"] - dataframe["speed"]
        dataframe["acceleration_gap_lead"] = dataframe["lead_acceleration"] - dataframe["acceleration"]
        dataframe["position_gap_side_lead"] = dataframe["left_lead_position"] - dataframe["position"]
        dataframe["speed_gap_side_lead"] = dataframe["left_lead_speed"] - dataframe["speed"]
        dataframe["acceleration_gap_side_lead"] = dataframe["left_lead_acceleration"] - dataframe["acceleration"]
        dataframe["position_gap_side_follow"] = dataframe["left_follow_position"] - dataframe["position"]
        dataframe["speed_gap_side_follow"] = dataframe["left_follow_speed"] - dataframe["speed"]
        dataframe["acceleration_gap_side_follow"] = dataframe["left_follow_acceleration"] - dataframe["acceleration"]

        dataframe["self_position"] = dataframe["position"]
        dataframe["self_speed"] = dataframe["speed"]
        dataframe["self_acceleration"] = dataframe["acceleration"]
        if dataframe.equals(left_leader_anomaly_blocked_df) or dataframe.equals(left_leader_normal_blocked_df):
            dataframe["blocking_vid"] = dataframe["left_lead_vid"] 
        else:
            dataframe["blocking_vid"] = dataframe["left_follow_vid"]
        
        dataframe = dataframe[["vid", "time","position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                    "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                    "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                    "self_position", "self_speed", "self_acceleration", "blocking_vid"]]
        if store_key == 0:
            dataframe.to_csv('./trajectory/ramp/acceleration_vehicle_attack/with_attack/length_5/anomaly_blocked_attack.csv', mode='a+', index=False, header=False) 
        else:
            dataframe.to_csv('./trajectory/ramp/acceleration_vehicle_attack/with_attack/length_5/normal_blocked_attack.csv', mode='a+', index=False, header=False) 


    elif dataframe.equals(right_leader_anomaly_blocked_df) or dataframe.equals(right_follower_anomaly_blocked_df) or dataframe.equals(right_leader_normal_blocked_df) or dataframe.equals(right_follower_normal_blocked_df):
        dataframe["position_gap_lead"] = dataframe["lead_position"] - dataframe["position"]
        dataframe["speed_gap_lead"] = dataframe["lead_speed"] - dataframe["speed"]
        dataframe["acceleration_gap_lead"] = dataframe["lead_acceleration"] - dataframe["acceleration"]
        dataframe["position_gap_side_lead"] = dataframe["right_lead_position"] - dataframe["position"]
        dataframe["speed_gap_side_lead"] = dataframe["right_lead_speed"] - dataframe["speed"]
        dataframe["acceleration_gap_side_lead"] = dataframe["right_lead_acceleration"] - dataframe["acceleration"]
        dataframe["position_gap_side_follow"] = dataframe["right_follow_position"] - dataframe["position"]
        dataframe["speed_gap_side_follow"] = dataframe["right_follow_speed"] - dataframe["speed"]
        dataframe["acceleration_gap_side_follow"] = dataframe["right_follow_acceleration"] - dataframe["acceleration"]\

        dataframe["self_position"] = dataframe["position"]
        dataframe["self_speed"] = dataframe["speed"]
        dataframe["self_acceleration"] = dataframe["acceleration"]
        if dataframe.equals(right_leader_anomaly_blocked_df) or dataframe.equals(right_leader_normal_blocked_df):
            dataframe["blocking_vid"] = dataframe["right_lead_vid"] 
        else:
            dataframe["blocking_vid"] = dataframe["right_follow_vid"]
        
        dataframe = dataframe[["vid", "time","position_gap_lead", "speed_gap_lead", "acceleration_gap_lead",
                                    "position_gap_side_lead", "speed_gap_side_lead", "acceleration_gap_side_lead",
                                    "position_gap_side_follow", "speed_gap_side_follow", "acceleration_gap_side_follow",
                                    "self_position", "self_speed", "self_acceleration", "blocking_vid"]]

        if store_key == 1:
            dataframe.to_csv('./trajectory/ramp/acceleration_vehicle_attack/with_attack/length_5/anomaly_blocked_attack.csv', mode='a+', index=False, header=False) 
        else:
            dataframe.to_csv('./trajectory/ramp/acceleration_vehicle_attack/with_attack/length_5/normal_blocked_attack.csv', mode='a+', index=False, header=False) 
    

#-----------------------------------------------------------------------------------------------#

