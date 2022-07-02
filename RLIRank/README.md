## RLIRank: Learning to Rank with Reinforcement Learning for Dynamic Search Implementation



### Dependencies: 
---
```
 1. python 3
 2. PyTorch
 3. Numpy
 4. Gym
```
### Flowchart
---
![Flowchart](https://github.com/sans-sehgal/DRDO-Interns-2022/blob/main/RLIRank/RLIFlowChart.PNG)

### Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of epochs(-e): `Required`
4. Gamma(-g): `1`
5. Learning rate actor(-alpha): `0.0001`
6. Learning rate critic(-beta): `0.0002`
7. Episode Length(-episode_length): `256`

### Runs RLIRank on MQ2008 using given hyperparameters
---
To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -nf num_features -e num_epochs -g gamma -alpha actorlearningrate -beta criticlearningrate -episode_length max_episode_length`<br>

### Example: 
---
1. Running on given hyperparameters: <br> `$ python main3.py -d ./data/MQ2008/all_0,1,2 -nf 46 -alpha 0.0001 -beta 0.00002 -g 1 -e 35`
---

### Results:
---
All results, graphs, models are saved in the Folder - Results.

### Citation:

The code for the paper can be found here
[RLIRank](https://arxiv.org/abs/2105.10124)

