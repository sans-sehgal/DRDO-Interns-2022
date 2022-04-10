### Runs RLIRank on MQ2007/MQ2008 using given hyperparameters

---


To run the algorithm, enter the following code:<br>
`$ python main3.py -d data_directory -nf no_features -e no_epochs -g gamma -lr_actor actor_lr -lr_critic critic_lr -episode_length episode_length -seed seed `<br>


<br>Example: 
---
1. Running on given hyperparameters: <br> `$ python main3.py -d ./Data/MQ2008/all_0,1,2 -nf 46 -lr_actor 0.0001 -lr_critic 0.0002 -g 1 -e 50 -episode_length 30 -seed 3 `
2. Running on given hyperparameters: <br> `$ python main3.py -d ./Data/MQ2007/all_0,1,2 -nf 46 -lr_actor 0.0001 -lr_critic 0.0002 -g 1 -e 50 -episode_length 20 -seed 3 `
---

## RLIRANK Paper
[RLIRank: Learning to Rank with Reinforcement Learning for Dynamic Search](https://arxiv.org/abs/2105.10124)


