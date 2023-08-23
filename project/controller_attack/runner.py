import os, sys
import csv
from sumolib import checkBinary
import traci
import random
import pandas

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
step_length = 0.1    
sumoBinary = checkBinary('sumo-gui')
# sumoCmd = [sumoBinary, "-c", "anomaly.sumocfg", "--step-length", "0.2", "--lanechange.duration", "2"]
sumoCmd = [sumoBinary, "-c", "controller.sumocfg", "--step-length", f'{step_length}', "--seed", "76", "--time-to-teleport", "30" ]

## seed: ramp 76, overtake 76, roundabout 76(accel attack not done)

traci.start(sumoCmd)


f = open('./trajectory/test/ghost_vehicle_attack/without_attack/raw_trajectory.csv', 'w+')# modify
writer = csv.writer(f)

# modify
# attack_pos = open('./attack_file/attack_pos.txt', 'r')
# attack_pos = attack_pos.readlines()
# attack_pos_list = attack_pos[0].split(" ")
# attack_vel = open('./attack_file/attack_vel.txt', 'r')
# attack_vel = attack_vel.readlines()
# attack_vel_list = attack_vel[0].split(" ")
# attack_acc = open('./attack_file/attack_acc.txt', 'r')
# attack_acc = attack_acc.readlines()
# attack_acc_list = attack_acc[0].split(" ")

# modify
# ghost_attack = open('./attack_file/ghost_attack.txt')
# ghost_attack_list = {}
# lines = [line.rstrip() for line in ghost_attack]

# ghost_attack_list = {}
# with open("./attack_file/ghost_attack.txt") as f_:
#     for line in f_:
#         if len(line.strip()) == 0:
#             continue
        
#         item = line.split()
#         # print(type(item[0]), type(item[1]))
#         if item[0] not in ghost_attack_list:
#             ghost_attack_list[item[0]] = {}
#         ghost_attack_list[item[0]][item[1]] = [float(item[2]), float(item[3]), float(item[4])]
# f_.close()


offset = 0.1
def get_feature(vid, step):
    pos = traci.vehicle.getLanePosition(vid)
    vel = traci.vehicle.getSpeed(vid)
    acc = traci.vehicle.getAcceleration(vid)
    
    # modify
    # if "anomaly" in str(vid):
    #     pos += float(attack_pos_list[int(step)%200])
    #     vel += float(attack_vel_list[int(step)%200])
    #     acc += float(attack_acc_list[int(step)%200])

    # modify
    # if "anomaly" in str(vid):
    #     step = float(step)
    #     # print(step, str(round(step/10, 1)))
        
    #     if str(round(step/10, 1)-offset) in ghost_attack_list[str(vid)]:
    #         pos = ghost_attack_list[str(vid)][str(round(step/10, 1)-offset)][0]
    #         vel = ghost_attack_list[str(vid)][str(round(step/10, 1)-offset)][1]
    #         acc = ghost_attack_list[str(vid)][str(round(step/10, 1)-offset)][2]
    #     else:
    #         pos = traci.vehicle.getLanePosition(vid)
    #         vel = traci.vehicle.getSpeed(vid)
    #         acc = traci.vehicle.getAcceleration(vid)

    # else:
    #     pos = traci.vehicle.getLanePosition(vid)
    #     vel = traci.vehicle.getSpeed(vid)
    #     acc = traci.vehicle.getAcceleration(vid)        

    return pos, vel, acc





step = 0       
while step<500000:
    traci.simulationStep()
    for vid in traci.vehicle.getIDList():
        

        follow_vid = None
        lead_vid = lead_pos = lead_vel = lead_acc = None
        r_lead_vid = r_lead_pos = r_lead_vel = r_lead_acc = r_follow_vid = r_follow_pos = r_follow_vel = r_follow_acc = None
        l_lead_vid = l_lead_pos = l_lead_vel = l_lead_acc = l_follow_vid = l_follow_pos = l_follow_vel = l_follow_acc = None

        if traci.vehicle.getFollower(vid):
            follow_vid = traci.vehicle.getFollower(vid)[0]

        if traci.vehicle.getLeader(vid):
            lead_vid = traci.vehicle.getLeader(vid)[0]
            lead_pos, lead_vel, lead_acc = get_feature(lead_vid, step)

            # lead_pos = traci.vehicle.getLanePosition(lead_vid)
            # lead_vel = traci.vehicle.getSpeed(lead_vid)
            # lead_acc = traci.vehicle.getAcceleration(lead_vid)
        if traci.vehicle.getRightFollowers(vid):
            r_follow_vid = traci.vehicle.getRightFollowers(vid)[0][0]
            r_follow_pos, r_follow_vel, r_follow_acc = get_feature(r_follow_vid, step)

            # r_follow_pos = traci.vehicle.getLanePosition(r_follow_vid)
            # r_follow_vel = traci.vehicle.getSpeed(r_follow_vid)
            # r_follow_acc = traci.vehicle.getAcceleration(r_follow_vid)        
        if traci.vehicle.getRightLeaders(vid):
            r_lead_vid = traci.vehicle.getRightLeaders(vid)[0][0]
            r_lead_pos, r_lead_vel, r_lead_acc = get_feature(r_lead_vid, step)

            # r_lead_pos = traci.vehicle.getLanePosition(r_lead_vid)
            # r_lead_vel = traci.vehicle.getSpeed(r_lead_vid)
            # r_lead_acc = traci.vehicle.getAcceleration(r_lead_vid)
        if traci.vehicle.getLeftLeaders(vid):
            l_lead_vid = traci.vehicle.getLeftLeaders(vid)[0][0]
            l_lead_pos, l_lead_vel, l_lead_acc = get_feature(l_lead_vid, step)
            
            # l_lead_pos = traci.vehicle.getLanePosition(l_lead_vid)
            # l_lead_vel = traci.vehicle.getSpeed(l_lead_vid)
            # l_lead_acc = traci.vehicle.getAcceleration(l_lead_vid)
        if traci.vehicle.getLeftFollowers(vid):
            l_follow_vid = traci.vehicle.getLeftFollowers(vid)[0][0]
            l_follow_pos, l_follow_vel, l_follow_acc = get_feature(l_follow_vid, step)

            # l_follow_pos = traci.vehicle.getLanePosition(l_follow_vid)
            # l_follow_vel = traci.vehicle.getSpeed(l_follow_vid)
            # l_follow_acc = traci.vehicle.getAcceleration(l_follow_vid)

        # print("step:", step, "vid:", vid, "lead:", lead_vid, "follow:", follow_vid, "r_lead:", r_lead_vid, "r_follow:", r_follow_vid, "l_lead:", l_lead_vid, "l_follow:", l_follow_vid)
        # trajectory = [vid, round(traci.simulation.getTime()-step_length, 1), traci.vehicle.getLanePosition(vid), traci.vehicle.getSpeed(vid),
        # traci.vehicle.getAcceleration(vid), traci.vehicle.getLaneChangeStatePretty(vid, -1)[1], traci.vehicle.getLaneChangeStatePretty(vid, 1)[1], traci.vehicle.getLaneIndex(vid), ## -1 right change, 1 left change
        # lead_vid, lead_pos, lead_vel, lead_acc,
        # r_lead_vid, r_lead_pos, r_lead_vel, r_lead_acc, r_follow_vid, r_follow_pos, r_follow_vel, r_follow_acc,
        # l_lead_vid, l_lead_pos, l_lead_vel, l_lead_acc, l_follow_vid, l_follow_pos, l_follow_vel, l_follow_acc,
        # traci.vehicle.getLaneID(vid)]

        # modify
        trajectory = [vid, round(traci.simulation.getTime()-step_length, 1), 
        traci.vehicle.getLanePosition(vid), traci.vehicle.getSpeed(vid),
        traci.vehicle.getAcceleration(vid)]


        writer.writerow(trajectory)
        
        # if "emergency" in str(vid):
        #     writer.writerow(trajectory)
        #     if traci.vehicle.getLaneID(vid) == "-E0_1" and lead_vid is not None:
        #         print("lead", step, lead_vid)
        #     if traci.vehicle.getLaneID(vid) == "-E0_1" and len(follow_vid) != 0:
        #         print("follow", step, follow_vid)

    step+=1
    
f.close()
traci.close()
