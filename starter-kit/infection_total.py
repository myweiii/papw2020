import simulator
import numpy as np

period = 840



all_repeat = []

for repeat in range(30):
    all_stage = [] 
    
    for stage in range(1, 6):
        all_day = []
        
        engine = simulator.Engine(thread_num=1, write_mode="write", specified_run_name=f"scenario{stage}", scenario=f"scenario{stage}") #scenario{stage}")
        engine.reset()
        
        
        idv_states = np.zeros(10000)
        infect_num = []
        for i in range(period):
            engine.next_step()
            engine.get_current_time()
            #print(f"i: {i} - {engine.get_area_infected_cnt(1)}")
            
            if i % 14 == 0:
                
                idv_states = np.zeros(10000)  # current total infection
                for idn in range(0, 10000):
                    if engine.get_individual_infection_state(idn) == 3 or engine.get_individual_infection_state(idn) == 4:
                        idv_states[idn] = 1
                        
                area_contained_ind = engine.get_area_contained_individual()
                daily_infect = []
                for area in range(0, 98):
                    if area in area_contained_ind.keys():
                        daily_infect.append(idv_states[area_contained_ind[area]].sum())
                    else:
                        daily_infect.append(0)
                daily_infect_total = sum(daily_infect)
                daily_infect.append(daily_infect_total)
                    
                infect_num.append(daily_infect)
                #print(daily_infect)
                #print(idv_states.sum())


        print("-----------")
        print(engine.get_individual_visited_history(0))
        print(engine.get_all_area_category())
        
        print(f"Repeat {repeat} - Senario {stage} Ends!")
        print("="*30)
        del engine
        
        
        all_day = np.stack(infect_num, axis=0)
        
        
        all_stage.append(all_day)
        
    all_stage = np.stack(all_stage, axis=0)
    all_repeat.append(all_stage)

#import ipdb;ipdb.set_trace()
print(all_day.shape)
print(all_stage.shape)

all_repeat = np.stack(all_repeat, axis=0)
print(all_repeat.shape)


np.save("./results/total_infect.npy", all_repeat)
