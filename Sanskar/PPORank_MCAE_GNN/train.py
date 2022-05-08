import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from PPO import PPO
from Environment import *
from Evaluation import *
from progress.bar import Bar
import pickle
from utils import *
import argparse

def test_train(model, data):
    """tests the model on train set and returns ndcg,avgstepreward values"""
    score = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTrain()))
    score_history = []
    for Q in data.getTrain():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        score = 0
        n_steps = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # select action with policy
            action = model.select_action_test(observation)
            action_list.append(state[action])
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            n_steps += 1
            score += reward

            # Update state
            state = state_

        score_history.append(score/n_steps)

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        bar.next()

    bar.finish()
    print(f'Avg Step Reward: {round(np.array(score_history).mean(), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}\n")

    return final_result, np.array(score_history).mean()

def test(model, data):
    """tests the model on test set and returns ndcg,avgstepreward values"""
    score = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))
    score_history = []
    for Q in data.getTest():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        score = 0
        n_steps = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)


            # select action with policy
            action = model.select_action_test(observation)
            action_list.append(state[action])
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            n_steps += 1
            score += reward

            # Update state
            state = state_

        score_history.append(score/n_steps)

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        bar.next()

    bar.finish()
    print(f'Avg Step Reward: {round(np.array(score_history).mean(), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}\n")

    return final_result, np.array(score_history).mean()

def pickle_data(data, output_file):
    """"Pickle the python objects into the output_file"""""

    if os.path.exists(output_file):
        os.remove(output_file)

    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()

def get_name(datadir):
    """Get the dataset name from the path"""
    lst=datadir.split('/')
    ds=""
    for i in lst:
        if('ohsumed' in i.lower()):
            ds='OHSUMED'
        elif('mq2008' in i.lower()):
            ds='MQ2008'
        elif('mq2007' in i.lower()):
            ds='MQ2007'
        elif('cora_embeddings' in i.lower()):
            ds = 'CORA'
    if(len(ds)==0):
        print("Wrong Dataset,Please check path")
        exit()
    else:
        return ds

################################### Training ###################################

def train(datadir,n_features,n_iterations,max_ep_len,save_model_freq,update_timestep,K_epochs,eps_clip,gamma,lr_actor,lr_critic,hidden_nodes,random_seed):

    ####### initialize environment hyperparameters ######
    env_name = "MDPRank"
    # max_ep_len = 50                    # max timesteps in one episode
    # save_model_freq = int(50000)      # save model frequency (in num timesteps)
    # n_features=46
    state_dim=n_features
    action_dim=1
    # datadir = "/home/siva/Desktop/Project_Work/MDPRank/LTR-MDPRank/Data/MQ2007_All_DIC_0,1,2"

    ################ PPO hyperparameters ################
    # n_iterations=20            # No. of iterations to run ppo 
    # update_timestep = 100      # update policy every n timesteps
    # K_epochs = 2               # update policy for K epochs
    # eps_clip = 0.2              # clip parameter for PPO
    # gamma = 0.99                # discount factor

    # lr_actor = 0.0003       # learning rate for actor network
    # lr_critic = 0.0005       # learning rate for critic network

    # hidden_nodes=64          #Hidden nodes in the hidden layer

    # random_seed = 3         # set random seed 
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    ################### checkpointing ###################

    ds=get_name(datadir)
    spath=f"./Results/{ds}/{ds}-i_{n_iterations}-lr_ac_{lr_actor},{lr_critic}-g_{gamma}-hnodes_{hidden_nodes}-max_T_{max_ep_len}-up_T_{update_timestep}-epochs_{K_epochs}-seed_{random_seed}"
    checkpoint_path = spath + "_model"

    #####################################################


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    #####################################################

    print("============================================================================================")
    
    ################# training procedure ################


    # initialize a PPO agent
    ppo_agent = PPO(state_dim,action_dim,hidden_nodes,lr_actor,lr_critic,gamma,K_epochs,eps_clip)


    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    #Monitor timestep to update and save models
    time_step = 0
    i_episode = 0
    prev_shape = 20
    #Load the data objects

    #Stores the training and testing ndcg scores and avg rewards
    results={'train':[],'test':[]}

    # #Initial Results on train and test sets
    # print("Initial Training Results:")
    # final_result,avgr=test_train(ppo_agent, data)
    # results['train'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(avgr, 4)))

    # print("\nInitial Test Results:")
    # final_result,avgr=test(ppo_agent, data)
    # results['test'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(avgr, 4)))
    i = 0

    while prev_shape < 2708:
        data = Dataset(prev_shape)

        print("\n\n---","Iteration:",i+1,"----\n")
        i+=1
        train_queries=data.getTrain()
        #train_queries=train_queries[:int(len(train_queries)/2)]
        train_queries=np.random.choice(train_queries,len(train_queries),replace=False)#Randomize the train queries
        
        #Monitors
        dcg_results={}
        avgstep_ep=[]
        bar = Bar('Training', max=len(train_queries))
        ts=time.time()
        
        for Q in train_queries:

            #Docids in the current state
            state = data.getDocQuery()[Q]
            current_ep_reward = 0

            action_list=[]
            effective_length=min(max_ep_len,len(state))

            for t in range(0, effective_length):

                #Feature matrix of the current state
                observation = [data.getFeatures()[x] for x in state]
                observation = np.array(observation, dtype=float)

                # select action with policy
                action = ppo_agent.select_action(observation)
                action_list.append(state[action])
                
                #Update state based on action and get reward
                state_, reward = update_state(t, Q, state[action], state, data.getTruth())
                
                current_ep_reward+=reward
                
                #Check if terminal state
                if(len(state_)==0):
                    done=True
                else:
                    done=False

                #saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step +=1

                # update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # save model weights
                if time_step % save_model_freq == 0:
                    ppo_agent.save(checkpoint_path)

                #If episode over then break    
                if done:
                    break

                #Update the state
                state=state_

            bar.next()

            #Track current episode performance
            avgstep_ep.append(current_ep_reward/effective_length)
            dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
            dcg_results[Q] = np.round(dcg_results[Q], 4)
        
        bar.finish()

        #Print current iteration performance
        print("Train results:")
        print(f'Avg Reward per step: {round(np.array(avgstep_ep).mean(), 4)}, time: {round(time.time() - ts)}')
        final_result = calculate(dcg_results)
        print(f"NDCG@1: {final_result[0]}\t"
              f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")
        
        #Save current iteration performance
        results['train'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(np.array(avgstep_ep).mean(), 4)))
        
        print("\nTest Results:")
        final_result,avgr=test(ppo_agent, data)
        results['test'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(avgr, 4)))
        prev_shape = data.get_prev_shape()

    #Saving results for visualization and analysis
    pickle_data(results,spath+"_results")

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    #Display and save graphs on the screen
    view_save_plot(results, spath)
    view_save_plot_ndcg(results, spath)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="relative or absolute directory of the directory where complete dictionary exists")
    ap.add_argument("-i", "--iterations", required=True,
                    help="number of iterations to run for")
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-g", "--gamma", required=False, default=0.99,
                    help="Gamma value; default = 0.99")
    ap.add_argument("-lr_actor", "--lr_actor", required=False, default=0.0003,
                    help="Learning rate of the actor; Defaults to 0.0003")
    ap.add_argument("-lr_critic", "--lr_critic", required=False, default=0.001,
                    help="Learning rate of the critic; Defaults to 0.001")
    ap.add_argument("-hnodes", "--hidden_layer_nodes", required=False, default=64,
                    help="Number of hidden nodes ")
    ap.add_argument("-steps", "--episode_length", required=False, default=50,
                    help="Number of steps to take in each episode")
    ap.add_argument("-epochs", "--epochs", required=False, default=3,
                    help="epochs to optimize surrogate")
    ap.add_argument("-clip", "--policy_clip", required=False, default=0.2,
                    help="clipping parameter")
    ap.add_argument("-save", "--save_freq", required=False, default=50000,
                    help="save frequency in steps")
    ap.add_argument("-update_T", "--update_timestep", required=False, default=200,
                    help="number of steps after which to update")
    ap.add_argument("-seed", "--seed", required=False, default=7,
                    help="Random seed to initialize")

    args = vars(ap.parse_args())

    # Initializing arguments
    data_dir = str(args["data"])
    num_features = int(args["features_no"])
    num_iterations = int(args["iterations"])
    gamma = float(args["gamma"])
    lr_actor = float(args["lr_actor"])
    lr_critic = float(args["lr_critic"])
    hidden_nodes = int(args["hidden_layer_nodes"])
    eps_clip= float(args["policy_clip"])

    # Extra arguments for PPO
    max_ep_len = int(args["episode_length"])
    K_epochs = int(args["epochs"])
    save_model_freq=int(args['save_freq'])
    update_timestep=int(args['update_timestep'])
    random_seed=int(args['seed'])

    #Start the training process
    train(data_dir,num_features,num_iterations,max_ep_len,save_model_freq,update_timestep,K_epochs,eps_clip,gamma,lr_actor,lr_critic,hidden_nodes,random_seed)
    