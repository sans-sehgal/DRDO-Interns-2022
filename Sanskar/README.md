### Instructions to generate dataset and then run PPO on the generated Dataset.

To begin generating the dataset, run the following command:<br>
`$ python3 generating_dicts.py`

After the code has sucessfully run, verify that a datset file has been generated, called "complete_msmarco". 


Next, run the command to start training PPO on the generated datset. The command for running PPO is: 

`$ python train.py -d ../complete_msmarco -i 150 -nf 50 -g 0.99 -lr_actor 0.0003 -lr_critic 0.001 -update_T 1024 -steps 50 -epochs 3 -clip 0.2 -hnodes 64 -save 10000 -seed 7`

Note that this command needs to be run from inside the PPORank-MCAE-BootStrap directory. 

