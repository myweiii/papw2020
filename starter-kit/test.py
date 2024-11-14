import simulator
import numpy as np

period = 840



all_repeat = []

for repeat in range(20):
    all_stage = [] 
    
    for stage in range(1, 6):
        all_day = []
        
        engine = simulator.Engine(thread_num=1, write_mode="write", specified_run_name=f"scenario{stage}", scenario=f"scenario{stage}") #scenario{stage}")
        engine.reset()
        
        for i in range(period):
            engine.next_step()
            engine.get_current_time()
            #print(f"i: {i} - {engine.get_area_infected_cnt(1)}")
            
            if i % 14 == 0:
                areas = []
                for idx in range(0, 98):
                    area_cnt = engine.get_area_infected_cnt(idx)
                    areas.append(area_cnt)
                #areas = np.stack(areas)
                print(f"{repeat+1} - {stage}: {i} - {sum(areas)} \n {areas}")

                
                all_day.append(np.append(areas, np.sum(areas)))
        #all_period = np.stack(all_period)
        #import ipdb;ipdb.set_trace()
        
        '''
        area_cates = engine.get_all_area_category()
        with h5py.File(f'./results/area_cates_scenario{stage}.h5', 'w') as hdf:
            for key, value in area_cates.items():
                hdf.create_dataset(str(key), data=value)
        '''
        
        
        
        '''
        multi_dim_index = np.unravel_index(np.argmax(all_period), all_period.shape)
        print(np.argmax(all_period))
        print(multi_dim_index)
        print(np.max(all_period))
        print(all_period[multi_dim_index])
        #np.save("./results/all_period_data.npy", all_period)
        '''


        print("-----------")
        print(engine.get_individual_visited_history(0))
        print(engine.get_all_area_category())
        
        print(f"Senario {stage} Ends!")
        print("="*30)
        del engine
        
        
        all_day = np.stack(all_day, axis=0)
        
        
        all_stage.append(all_day)
        
    all_stage = np.stack(all_stage, axis=0)
    all_repeat.append(all_stage)

#import ipdb;ipdb.set_trace()
print(all_day.shape)
print(all_stage.shape)

all_repeat = np.stack(all_repeat, axis=0)
print(all_repeat.shape)


np.save("./results/all_day_repeat20_stage1_5.npy", all_repeat)
