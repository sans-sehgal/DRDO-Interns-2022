### Runs PPO on GNN setting for selected hyperparameters and 70-30 Train Test Split

---


To run the algorithm, enter the following code:<br>
``` 
python train.py -d data_directory -nf num_features -i num_iterations -g gamma -lr_actor actor_learningrate -lr_critic critic_learningrate -hnodes nodes_hiddenlayer
-steps max_episode_length -update_T update_after_T -epochs num_epochs -clip policy_clip -save save_frequency -seed seed 
```
<br>
<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of iterations(-i): `Required`
4. Gamma(-g): `0.99`
5. Learning rate actor(-lr_actor): `0.0003`
6. Learning rate critic (-lr_critic): `0.001`
7. Hidden layer Nodes(-hnodes): `64`
8. Episode Length(-steps): `50`
9. Update Timestep(-update_T): `200`
10. Epochs(-epochs): `3`
11. Policy clip (-clip): `0.2`
12. Save timestep(-save): `50000`
13. seed(-seed):`7`


<br>Example: 
---
1. Running with given hyperparameters: <br> ``` python train.py -d ../data/MQ2008_All_DIC_0,1,2 -i 50 -nf 46 -g 0.98 -lr_actor 0.005 -lr_critic 0.006 -hnodes 45 -steps 30 -update_T 100 -epochs 3 -clip 0.3 -hnodes 32 -save 10000 -seed 3```



<br>
This runs PPORank MC advantage estimate for given hyperparameters and saves all results,graphs,models in the Folder, Results.

---



