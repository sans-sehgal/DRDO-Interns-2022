### Runs PPO on MDPRank setting for selected hyperparameters on 70-30 Train Test Split

---


To run the algorithm, enter the following code:<br>
`$ python train.py -d data_directory -nf num_features -i num_iterations -g gamma -lr_actor actor_learningrate -lr_critic critic_learningrate -hnodes nodes_hiddenlayer -update_T update_after_T -steps max_episode_length -epochs num_epochs -clip policy_clip -save save_frequency -seed seed `<br>

<br>Default hyperparameters: 
---
1. Data(-d): `Required`
2. Number of features(-nf): `Required`
3. Number of iterations(-i): `Required`
4. Gamma(-g): `0.99`
5. Learning rate actor(-lr_actor): `0.0003`
6. Learning rate critic (-lr_critic): `0.001`
7. Hidden layer Nodes(-hnodes): `64`
8. Update Timestep(-update_T): `200`
9. Max Episode Length(-steps): `999999`
10. Epochs(-epochs): `3`
11. Policy clip (-clip): `0.2`
12. Save timestep(-save): `50000`
13. seed(-seed):`7`


<br>Example: 
---
1. Running on default hyperparameters for 70-30 split and 100 iterations (MQ2007): <br> `$ python train.py -d ../data/MQ2007_All_DIC_0,1,2 -i 100 -nf 46`
2. Running with given hyperparameters: <br> `$ python train.py -d ../data/MQ2007_All_DIC_0,1,2 -i 150 -nf 46 -g 0.99 -lr_actor 0.0003 -lr_critic 0.001 -update_T 1024 -steps 50 -epochs 3 -clip 0.2 -hnodes 64 -save 10000 -seed 7`



<br>
This runs PPORank MC advantage estimate for given hyperparameters and saves all results,graphs,models in the Folder, Results.

---


---
<br>PPO Algorithm:

<img src="https://user-images.githubusercontent.com/51087175/121606087-2837f780-ca6b-11eb-8560-e31ad96f05b3.png" width="700" height="350">

<br>PPORank Algorithm:
<br>
<img src="https://user-images.githubusercontent.com/51087175/125517130-94d112bb-703e-49ac-ad37-e390c10759bd.png" width="700" height="600">
<img src="https://user-images.githubusercontent.com/51087175/125517202-85f9a811-5f38-4980-9787-f594cf2c0be8.png" width="700" height="400">

---
