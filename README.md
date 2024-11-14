### This is a code sample for [papw 2020](https://hzw77-demo.readthedocs.io/en/round2/index.html).

# preprocess.py
### get_graph()
Use the longitude and lantitude of the 11 nodes to calculate distance in real world, generating adjacency matrix. (For stage 3, the node number is 98)

### get_new_sld_window()
The input shape is (30, 5, 60, 99), in which 30 is the repeated simulator running time. 5 is the stage number in each repeated simulation, with each stage simulated for 60 days. The 99 denotes that there are 98 nodes and 1 sum column with all nodes. Currently only the first 11 nodes were used to simulate stage 1, 2, 4, and 5. The difference between the 5 stages is shown in [papw 2020 - scenario](https://hzw77-demo.readthedocs.io/en/round2/scenario.html).

For each 60 days, use a sliding window with horizon size of 10 days to predict the next 1 day. There are overlaps between each sliding window.

# papw_simulator.py
This file serves as the surrogate model to mimic the simulated data with sliding windows obtained from preprocess. `ColaGNN` serves as the surrogate model. The first 2 repeats are test set and the rest is divided into training set and validation set using `train_test_split` from `sklearn`. The training set and the validation set are shuffled randomly and standardized by the mean and std value of training set.

In the training process, use all the overlapped sliding window data. In the test process, only use the first 10 days from the original data, and then apply the sliding window to the original data with newly predicted data. For example, first use the 1-10 days to predict the 11st day, and next use the 2-10 days along with the predicted 11st day to predict the 12nd day, so forth.

# papw_pipeline.py
This is the attack model which is trying to decrease the output of the surrogate model. It use the previous 10 days with overlapped sliding window to predict the next day's intervention, outputing a graph mask. Then use the graph mask to get the modified graph and obtain surrogate model output with the 10 days and a modified graph. The loss function is designed to only have the infection number for now. The loss is updated every sliding window. In the test set, the process is similar with the surrogate model test process, only using the first 10 days from the original data, and then apply the sliding window to the original data with newly predicted data. The attack model also adopts newly predicted data.

# How to obtain the original simulated data
You should first follow the instruction of [papw 2020 - try it](https://hzw77-demo.readthedocs.io/en/round2/try.html) to download the docker file, and then use the command as follows:

`docker run -it -v /path/starter-kit:/test episim2020/simulator`

Replace the path to your path, and the enter the test file. Run `infection_total.py`. 

The simulator provided some APIs to obtain the fetuares for each day, shown in [papw 2020 - api](https://hzw77-demo.readthedocs.io/en/round2/api.html). The simulator will run 14 periods for each day, in my experiment, I found that the data inside the 14 periods each day remains the same.

In this implementation, 30 repeats were run, with obtaining current individual infected states for 98 nodes. For stage 1, 2, 4, 5, only the first 11 nodes are used. Besides, the location for each node is obtained through `location.py`. 

`test.py` is trying to obtain the newly discovered infection numbers for each node. The rest `.py` files are provided by papw.