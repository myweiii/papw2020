from haversine import haversine
import pandas as pd
import numpy as np
import pickle


def get_graph():
    # Test haversine
    loc_1 = (30.684437, 114.14762)
    loc_2 = (30.684437, 114.16711)
    print(haversine(loc_1, loc_2))


    location_data = pd.read_csv("./papw/location.csv")
    print(location_data)

    graph = np.zeros((len(location_data), len(location_data))) + np.inf
    print(graph.shape)

    # Use haversine to calculate distance between each location
    for loc_1_idx in range(0, len(location_data)):
        for loc_2_idx in range(loc_1_idx, len(location_data)):
            loc_1 = (location_data.loc[loc_1_idx, "lat"], location_data.loc[loc_1_idx, "lng"])
            loc_2 = (location_data.loc[loc_2_idx, "lat"], location_data.loc[loc_2_idx, "lng"])
            
            dist = haversine(loc_1, loc_2)
            print(f"loc{loc_1_idx} - loc{loc_2_idx}: {dist}")
            graph[loc_1_idx, loc_2_idx] = dist
            graph[loc_2_idx, loc_1_idx] = dist


    # Transfer distance to weight by applying a math method with a thershold
    std = np.std(graph)
    graph = np.e ** (-(graph/std)**2)

    #graph = np.where(graph > 0.001, graph, 0)
    print(graph)

    np.save("./papw/graph.npy", graph)
        

def get_features():
    data = pd.read_csv("./papw/area_cnt.csv")
    features = ["CurrentSusceptible", "CurrentIncubation", "CurrentDiscovered",
                "CurrentCritical", "CurrentRecovered", "CurrentInfected"]
    all_data = []
    for day in range(0, 60):
        daily_data = data[data["day"]==day][features]
        all_data.append(np.array(daily_data))
    all_data = np.stack(all_data)
    print(all_data.shape)
    
    horizon = 15
    pred_period = 1
    
    x = []
    y = []
    for sld_win in range(0, all_data.shape[0]):
        if sld_win+horizon+pred_period > all_data.shape[0]:
            break
        x_win = all_data[sld_win:sld_win+horizon]
        y_win = all_data[sld_win+horizon:sld_win+horizon+pred_period]
        x.append(x_win)
        y.append(y_win)
    
    x = np.stack(x)[:, :, :, :]/2000
    y = np.stack(y)[:, :, :, -1]/2000
    print(x.shape)
    print(y.shape)
    
    data = {"x": x, "y": y, "all_data": all_data}
    with open("./papw/data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    '''
    with open("./papw/data.pkl", "rb") as f:
        data = pickle.load(f)
    print(data["x"].shape)
    print(data["y"].shape)
    '''

def get_all_period_sld_window():
    all_data = np.load("./papw/all_1_5.npy")
    '''
    new_all_data = []
    for row in all_data:
        row = np.append(row, np.sum(row))
        new_all_data.append(row)
    all_data = np.stack(new_all_data)
    '''
    print(all_data.shape)
    horizon = 5
    pred_period = 5
    
    x = []
    y = []
    for sld_win in range(0, all_data.shape[0]):
        
        if sld_win+horizon+pred_period > all_data.shape[0]:
            break
        
        if 0 < (sld_win+horizon+pred_period) % 60 < horizon+pred_period:
            print(sld_win)
            #print(sld_win+horizon+pred_period)
            continue
        x_win = all_data[sld_win:sld_win+horizon]
        y_win = all_data[sld_win+horizon:sld_win+horizon+pred_period]
        x.append(x_win)
        y.append(y_win)
    
    
    x = np.stack(x)/3000
    y = np.stack(y)/3000
    print(x.shape)
    print(y.shape)
    
    data = {"x": x, "y": y, "all_data": all_data}
    with open("./papw/all_1_5.pkl", "wb") as f:
        pickle.dump(data, f)
        

def get_new_sld_window():
    #all = np.load("./papw/all_day_repeat20_stage1_5/all_day_repeat20_stage1_5.npy")
    all = np.load("./papw/total_infect/total_infect.npy")
    
    #padding = all[:, :, -3:, :]
    
    #padding =  np.tile(all[:, :, 0, :][:, :, np.newaxis, :], (1, 1, 3, 1))
    
    #padding = all[:, :, 0:2 , :]
    
    #all = np.concatenate((padding[::-1], all), axis=2)
    
    #all = np.pad(all, ((0, 0), (0, 0), (3, 0), (0, 0)))
    print(all.shape)
    horizon = 10
    pred_period = 1
    
    
    
    repeats = all.shape[0]
    stages = all.shape[1]
    
    all_repeat_x = []
    all_repeat_y = []
    
    for repeat in range(repeats):
        all_stage = all[repeat]
        all_stage_x = []
        all_stage_y = []
        for stage in range(stages):
            if stage == 2:
                continue
            all_data = all_stage[stage]
            x = []
            y = []
            for sld_win in range(0, all_data.shape[0]):
                
                if sld_win+horizon+pred_period > all_data.shape[0]:
                    break
                
                x_win = all_data[sld_win:sld_win+horizon]
                y_win = all_data[sld_win+horizon:sld_win+horizon+pred_period]
                x.append(x_win)
                y.append(y_win)
            x = np.stack(x)
            y = np.stack(y)
            #print(f"repeat {repeat + 1} - stage {stage + 1}")
            #print(x.shape)
            #print(y.shape)
            
            all_stage_x.append(x)
            all_stage_y.append(y)
        all_stage_x = np.stack(all_stage_x, axis=0)
        all_stage_y = np.stack(all_stage_y, axis=0)
        
        #print(all_stage_x.shape)
        #print(all_stage_y.shape)
            
        all_repeat_x.append(all_stage_x)
        all_repeat_y.append(all_stage_y)
        
    all_repeat_x = np.stack(all_repeat_x, axis=0)
    all_repeat_y = np.stack(all_repeat_y, axis=0)
        
    print(all_repeat_x.shape)
    print(all_repeat_y.shape) 
    # repears, stages, sld, timesteps, areas
    data = {"x": all_repeat_x, "y": all_repeat_y, "all_data": all}
    #with open("./papw/all_day_repeat20_stage1_5/total_infect_repeat20_stage1_5.pkl", "wb") as f:
    with open("./papw/total_infect/total_infect.pkl", "wb") as f:
        pickle.dump(data, f)
 
    
if __name__ == "__main__":
    #get_graph()
    #get_features()
    #get_all_period_sld_window()
    get_new_sld_window()