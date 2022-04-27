### Runs DQN on MDPRank setting for selected hyperparameters on 70-30 Train Test Split

---


To run the algorithm, enter the following code:<br>
`$ python main.py -d data_directory -nf num_features -e num_epochs -g gamma -lr learningrate -hnodes nodes_hiddenlayer
-episode_length max_episode_length -eps_end min_value_epsilon -eps_dec decay_epsilon -batch_size mini-batch-size -max_mem_size size_memory 
-replace_target steps_replace_target -seed seed `<br>

<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of epochs(-e): `Required`
4. Gamma(-g): `1`
5. Learning rate(-lr): `0.00005`
6. Mini-batch-size(-batch_size): `128`
7. Hidden layer Nodes(-hnodes): `32`
8. Episode Length(-episode_length): `50`
9. Epsilon end value(-eps_end): `0.03`
10. Epsilon linear decay value(-eps_dec): `0.00005`
11. Memory size(-max_mem_size): `50000`
12. Replace target interval(-replace_target): `2000`
13. seed(-seed):`3`


<br>Example: 
---
1. Running on default hyperparameters for 70-30 split and 50 epochs (MQ2007): <br> `$ python main.py -d ./data/MQ2007_All_DIC_0,1,2 -e 50 -nf 46`
2. Running with given hyperparameters: <br> `$ python main.py -d ./Data/MQ2007_All_DIC_0,1,2 -e 50 -nf 46 -g 1 -lr 0.00005 -hnodes 32 -episode_length 20 -eps_end 0.02 -eps_dec 0.00001 -batch_size 128 -max_mem_size 60000 -replace_target 3000 -seed 3`



<br>
This runs DQRank for given hyperparameters and saves all results,graphs,models in the Folder, Results.

---


---
<br>DQN Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/124238944-cbde6a00-db36-11eb-9dcc-cd32df656077.png" width="700" height="450">

<br>DQRank Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/125194380-b1ab3700-e26e-11eb-9af3-8183be88d2bc.png" width="700" height="500">

<br>

---
