# Similarity + Relevance Score Setting

### Dependencies: 
---
```
 1. python 3
 2. PyTorch
 3. Numpy
 4. Gym
```

### Dataset 
---
The algorithm was tested on the benchmark LETOR Datasets <br>
[LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)

### Algorithm
---
![Flowchart](https://github.com/sans-sehgal/DRDO-Interns-2022/blob/main/SS_RS_DQ-AC-Rank/Flowchart.jpg)

### Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of epochs(-e): `Required`
4. Gamma(-g): `1`
5. Learning rate actor(-lr_actor): `0.0001`
6. Learning rate critic(-lr_critic): `0.0002`
7. Mini-batch-size(-batch_size): `256`

### Runs DQ-AC on SS+RS setting for selected hyperparameters on 70-30 Train Test Split
---
To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -nf num_features -e num_epochs -g gamma -lr_actor actorlearningrate -lr_critic criticlearningrate -hnodes nodes_hiddenlayer
-episode_length max_episode_length -eps_end min_value_epsilon -eps_dec decay_epsilon -batch_size mini-batch-size -max_mem_size size_memory 
-replace_target steps_replace_target -seed seed `<br>

### Example: 
---
1. Running on default hyperparameters for 70-30 split and 50 epochs (MQ2008): <br> `$ python main.py -d ./data/MQ2008/all_0,1,2 -e 25 -nf 2`
2. Running with given hyperparameters: <br> `$ python main.py -d ./data/MQ2008/all_0,1,2 -e 25 -nf 2 -g 1 -lr_actor 0.0001 -lr_critic 0.0002 -batch_size 256`

### Results:
---
All results, graphs, models are saved in the Folder - Results.



