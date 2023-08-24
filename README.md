這一份project的主體基本上是SUMO的source code，不過在source code上也有不少的修改。  
lane changing的模擬和anomaly detection的code都在project/controller_attack中。  
模擬的部分由執行runner.py即可，模擬設定則由controller.sumocfg決定，.sumocfg檔中須至少包含一個.rou.xml檔和.net.xml檔才能正常進行模擬。  
模擬後的得到的資料可經由preprocess_trajectory.py進行處理，處理的資料即可做為machine learning model的input。  
關於如何修改SUMO的source code可參考附檔的論文內的Chapter 4.2 Attack Deployment。  
如有任何使用上的問題請寄信至r10922a06@ntu.edu.tw。
