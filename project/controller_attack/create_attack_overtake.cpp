#include <iostream>
#include <fstream>
#include <cmath>
#include <string.h>
#include <cstdio>
#include <stdio.h>
#include <vector>
#include <sstream>
using namespace std;


// int main(){
//     //find some way that can store these value(ofstream)
//     float acceleration[400], velocity[400], position[400];
//     int magnitude = 3;
//     float frequency = 0.2;
//     float vel_accumulate = 0;
//     float pos_accumulate = 0;

//     for(int time=0; time<400; time++){ //200 time steps
//         acceleration[time] = magnitude * sin(frequency * 0.1 * time + M_PI/2);
//     }
//     for(int time=0; time<400; time++){ //200 time steps
//         vel_accumulate += 0.1*acceleration[time];
//         velocity[time] = vel_accumulate;
//     }
//     for(int time=0; time<400; time++){ //200 time steps
//         pos_accumulate += 0.1*velocity[time];
//         position[time] = pos_accumulate;
//     }

//     ofstream myfile;
//     myfile.open("./attack_file/attack_acc.txt");
//     for(int i=0; i<400; i++){
//         myfile << acceleration[i] << " ";
//     }
//     myfile.close();

//     myfile.open("./attack_file/attack_vel.txt");
//     for(int i=0; i<400; i++){
//         myfile << velocity[i] << " ";        
//     }
//     myfile.close();

//     myfile.open("./attack_file/attack_pos.txt");
//     for(int i=0; i<400; i++){
//         myfile << position[i] << " ";        
//     }
//     myfile.close();

//     return 0;


// }

int main()
{   
    FILE *file;
    file = fopen("/home/r10922a06/Desktop/sumo2/sumo/project/controller_attack/trajectory/overtake/ghost_vehicle_attack/without_attack/raw_trajectory.csv", "r");
    char line[100];
    char* element;

    vector<vector<string>> record;
    int cnt = 0;
    int num = 0;

    record.push_back(vector<string>());
    while (fgets(line, sizeof(line), file)){
        // cout << line << endl;
        element = strtok(line,",");
        while (element != NULL){
            char *newline = strchr(element, '\n');
            if (newline){ //eliminate '\n'
                *newline = 0;
            }                

            if(record[record.size()-1].size() == 5){
                record.push_back(vector<string>());
            }
            record[record.size()-1].push_back(element);
            element = strtok(NULL, ",");
        }
    }
    // for (int i = 0; i < record.size(); i++){
    //     for (int j = 0; j < record[i].size(); j++){
    //         cout << record[i][j] << "\n";
    //     }0
    // }
    fclose(file);

    
    ofstream o_file;
    o_file.open ("/home/r10922a06/Desktop/sumo2/sumo/project/controller_attack/attack_file/ghost_attack.txt");

    for (int i = 0; i < record.size(); i++){
        for (int j = 0; j < record[i].size(); j++){
            if (j == record[i].size()-1){
                o_file << record[i][j];
            }else{
                o_file << record[i][j] << " ";
            }
        }
        // o_file << "\n";
    }
    o_file.close();

    return 0;
}



    
// int main(){
//     //find some way that can store these value(ofstream)
//     float delta = 0.1;
//     float gamma = 0.002;
//     int time = 200;
//     float o_p[time];
//     float o_v[time];
//     float o_a[time];

//     int offset = 20;
//     for(int i=offset;i<time+offset;i++){
//         o_p[i-offset] = delta * delta * gamma * (((0.167)*(i*i*i)) + ((0.25)*(i*i)) + ((0.0833)*(i)));
//     }
//     for(int i=offset;i<time+offset;i++){
//         o_v[i-offset] = delta * gamma  * i * i / 2;
//     }
//     for(int i=offset;i<time+offset;i++){
//         o_a[i-offset] = gamma * i;
//     }


//     ofstream myfile;
//     myfile.open("./attack_file/attack_acc.txt");
//     for(int i=0; i<time; i++){
//         myfile << o_a[i] << " ";
//     }
//     myfile.close();

//     myfile.open("./attack_file/attack_vel.txt");
//     for(int i=0; i<time; i++){
//         myfile << o_v[i] << " ";        
//     }
//     myfile.close();

//     myfile.open("./attack_file/attack_pos.txt");
//     for(int i=0; i<time; i++){
//         myfile << o_p[i] << " ";        
//     }
//     myfile.close();

//     return 0;


// }



