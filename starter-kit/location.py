import h5py
import pandas as pd
import numpy as np
from haversine import haversine


location = {"id": [], "type": []}

for stage in range(3, 4):
    id_list = []
    type_list = []
    with h5py.File(f'./results/area_cates_scenario{stage}.h5', 'r') as file:
        #print("Keys: %s" % file.keys())
        keys = list(file.keys())

        for key in keys:
            type = file[key][...]
            id_list.append(int(key))
            type_list.append(int(type))

    print(f"Data from {id_list}:")
    print(type_list)
    
    location["id"] = id_list
    location["type"] = type_list

with h5py.File(f'./results/location_type.h5', 'w') as hdf:
    for key, value in location.items():
        hdf.create_dataset(key, data=value)


loc_data = pd.read_csv("./results/hex_cnt_scenario3.txt", header=None)
print(loc_data)

location_data = []
for idx, row in loc_data.iterrows():
    if idx > 97:
        break
    print(row)
    loc_id = row[1]
    lon = row[2]
    lat = row[3]
    location_data.append(np.array([loc_id, lon, lat]))
    
    
location_data = np.stack(location_data)
    
print(location_data.shape)
graph = np.zeros((location_data.shape[0], location_data.shape[0]))

for loc_1_idx in range(0, len(location_data)):
    for loc_2_idx in range(loc_1_idx, len(location_data)):
        loc_1 = (location_data[loc_1_idx, 2], location_data[loc_1_idx, 1])
        loc_2 = (location_data[loc_2_idx, 2], location_data[loc_2_idx, 1])
        
        dist = haversine(loc_1, loc_2)
        print(f"loc{loc_1_idx} - loc{loc_2_idx}: {dist}")
        graph[loc_1_idx, loc_2_idx] = dist
        graph[loc_2_idx, loc_1_idx] = dist

# Transfer distance to weight by applying a math method with a thershold
std = np.std(graph)
graph = np.e ** (-(graph/std)**2)

#graph = np.where(graph > 0.001, graph, 0)
print(graph.shape)

np.save("./results/location_graph_98.npy", graph)