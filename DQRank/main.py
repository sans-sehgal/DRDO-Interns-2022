import gym
from dqn import Agent
import numpy as np
from Environment import *
from Evaluation import *
from progress.bar import Bar
import torch
import time
import os
import pickle
from Utils_DQN import *
import argparse
import scipy

def similarity_score(doc_list, q_vec):
    similarity_score = []
    for action in doc_list:
        #action = action[:46]
        try:
            cos_distance = scipy.spatial.distance.cosine(q_vec, action)
            cos_similarity = 1 - cos_distance
            similarity_score.append(cos_similarity)
        except:
            print("skipped")
            continue
    return similarity_score

def feedback(Q,action_list,data):
    for i,j in enumerate(action_list):
        print(i,'\t',j)
    
    val = input("Enter Feedback \n")
    f = [0]*len(action_list)

    for i in val.split():
        f[int(i)]=1
        data.updateRelevance(Q,action_list[int(i)],int(i)+1)        

    data.updateIDCG(Q)
    return int(val)





def pickle_data(data, output_file):
    """Pickles given data object into an outfile"""
    if os.path.exists(output_file):
        os.remove(output_file)

    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()

def get_name(datadir):
    """Gets name of dataset from the path"""
    lst=datadir.split('/')
    ds=""
    for i in lst:
        if('ohsumed' in i.lower()):
            ds='OHSUMED'
        elif('mq2008' in i.lower()):
            ds='MQ2008'
        elif('mq2007' in i.lower()):
            ds='MQ2007'
    if(len(ds)==0):
        print("Wrong Dataset,Please check path")
        exit()
    else:
        return ds

def test_train(model, data):
    """Testing model performance on train set"""

    #Initialize monitors
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTrain()))
    score_history = []

    for Q in data.getTrain():
        ep_reward = 0
        done = False
        state = data.getDocQuery()[Q]
        action_list = []
        M=len(state)

        for t in range(0,len(state)):

            #Get the current observation
            observation = [data.getFeatures()[x] for x in state]
            observation = torch.FloatTensor(np.array(observation,dtype=float))

            #Choose action based on the current observation
            action = agent.choose_action_test(observation)
            
            #Monitor actions to evaluate training
            action_list.append(state[action])

            #Update to next state,observation and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())
            observation_ = [data.getFeatures()[x] for x in state_]
            observation_ = torch.FloatTensor(np.array(observation_,dtype=float))
            
            ep_reward += reward
            
            if(len(state_)==0):
                done=True

            #Change the state
            state = state_
            
            if(done):
                break

        score_history.append(ep_reward/M)

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
    """Testing model performance on test set"""

    #Initialize monitors
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))
    score_history = []

    for Q in data.getTest():
        ep_reward = 0
        done = False
        state = data.getDocQuery()[Q]
        action_list = []
        M=len(state)

        for t in range(0,len(state)):

            #Get the current observation
            observation = [data.getFeatures()[x] for x in state]
            observation = torch.FloatTensor(np.array(observation,dtype=float))

            #Choose action based on the current observation
            action = agent.choose_action_test(observation)
            
            #Monitor actions to evaluate training
            action_list.append(state[action])

            #Update to next state,observation and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())
            observation_ = [data.getFeatures()[x] for x in state_]
            observation_ = torch.FloatTensor(np.array(observation_,dtype=float))
            
            ep_reward += reward
            
            if(len(state_)==0):
                done=True

            #Change the state
            state = state_
            
            if(done):
                break

        score_history.append(ep_reward/M)

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

def get_state(ss,rs):
    ss2=ss.copy()
    rs2=rs.copy()
    ss2.reverse()
    rs2.reverse()
    state=np.array([ss2[0],rs2[0]])
    for i in range(1,len(ss2)):
        state=np.vstack((np.array([ss2[i],rs2[i]]),state))
    #print(state.shape)
    return state

if __name__ == '__main__':

    ap=argparse.ArgumentParser()
    ap.add_argument("-d","--data",required=True,help="Relative or absolute directory of the folder where data dictionary exists.")
    ap.add_argument("-e","--epochs",required=True,help="Number of epochs")
    ap.add_argument("-nf", "--features_no", required=True,help="Number of features in the dataset")
    ap.add_argument("-lr", "--learning_rate", required=False,default=0.00005,help="learning rate for dqn")
    ap.add_argument("-g", "--gamma", required=False,default=1,help="gamma value")
    ap.add_argument("-episode_length", "--episode_length", required=False,default=50,help="Episode length to consider")
    ap.add_argument("-seed", "--seed", required=False,default=3,help="seed for initialization")
    ap.add_argument("-eps_end", "--eps_end", required=False,default=0.03,help="ending epsilon for e-greedy selection;starts at 1")
    ap.add_argument("-eps_dec", "--eps_dec", required=False,default=0.00005,help="Linear deacy for epsilon")
    ap.add_argument("-hnodes", "--hidden_layer_nodes", required=False, default=32,help="Number of hidden nodes")
    ap.add_argument("-batch_size", "--batch_size", required=False, default=128,help="Batch size")
    ap.add_argument("-max_mem_size", "--max_mem_size", required=False, default=50000,help="Size of Buffer")
    ap.add_argument("-replace_target", "--replace_target", required=False, default=2000,help="Number of steps to replace target")

    
    #HyperParameters
    args = vars(ap.parse_args())
    lr=float(args['learning_rate'])
    gamma=float(args['gamma'])
    epsilon=0.2
    eps_end=float(args['eps_end'])
    eps_dec = float(args['eps_dec'])
    batch_size=int(args['batch_size'])
    max_mem_size=int(args['max_mem_size'])
    replace_target = int(args['replace_target'])
    hnodes=int(args['hidden_layer_nodes'])
    n_features=int(args["features_no"])

    n_iterations=int(args["epochs"])
    t_steps=int(args['episode_length'])
    data_dir=str(args["data"])
    seed=int(args['seed'])

    # #HyperParameters
    # lr=0.00005
    # gamma=1
    # epsilon=1.0
    # eps_end=0.03
    # eps_dec = 0.00005
    # batch_size=128
    # max_mem_size=50000
    # replace_target = 2000
    # hnodes=32
    # n_features=46

    # n_iterations=150
    # t_steps=50
    # data_dir="/home/siva/Desktop/Project_Work/MDPRank/LTR-MDPRank/Data/MQ2008_All_DIC_0,1,2"
    # seed=3

    #Load the data object 
    data = Dataset(data_dir)

    #Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Initialize the agent
    agent = Agent(gamma=gamma, epsilon=epsilon, batch_size=batch_size, n_actions=1, eps_end=eps_end,
                  eps_dec=eps_dec, hnodes=hnodes,input_dims=[n_features],max_mem_size=max_mem_size,replace_target=replace_target,lr=lr)

    #Stores the training and testing ndcg scores and avg rewards
    results={'train':[],'test':[]}

    """
    #Initial scores on the train and test sets
    print("Initial Training Results:")
    final_result,avgr=test_train(agent, data)
    results['train'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(avgr, 4)))

    print("\nInitial Test Results:")
    final_result,avgr=test(agent, data)
    results['test'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(avgr, 4)))


    """

    #Begin training process for given no. of iterations
    for i in range(n_iterations):

        print(f"\n------Epoch {i+1}------------\n")
        
        train_queries=data.getTrain()
        #train_queries=train_queries[:int(len(train_queries)/3)]
        #Randomize the train queries
        train_queries=np.random.choice(train_queries,len(train_queries),replace=False)
        
        #Monitors
        score_history = []
        dcg_results = {}
        ts = time.time()
        bar = Bar('Training', max=len(train_queries))

        for Q in train_queries:
            ep_reward = 0
            done = False
            qvec = data.getQVEC(Q)
            state = data.getDocQuery()[Q]
            #print(state)
            docs = data.getDocQuery()[Q]
            docs_list = [data.getFeatures()[x] for x in state]
            docs_list = torch.FloatTensor(np.array(docs_list,dtype=float))
            ss=similarity_score(docs_list,qvec)
            #print(ss)
            rs=data.getRelevance(Q,docs)
            #rs=[(x+1)*10 for x in rs]
            #print(rs)
            state = get_state(ss,rs)          

            action_list = []
            effective_length=min(t_steps,len(state))
            print('NEW QUERY \n')
            for i in range(2):
                #print('STATE')
                #print(state)
                observation = torch.FloatTensor(np.array(state,dtype=float))
                action = agent.choose_action(observation)
                #print(action)
                ##feedback
                action_list=[]
                action2=action.copy()
                docs=[x for i,x in sorted(zip(action2,docs))]
                docs.reverse()
                #for i in range(len(action2)):
                #    arg=np.argmax(action2)
                #    action_list.append(docs[arg])                
                #    action2[arg]=0
                #print(action_list)
                #pos = feedback(Q,docs,data)
                pos=i+1

                reward=1/(pos)
                ep_reward+=reward
                rs=data.getRelevance(Q,docs)
                #rs=[(x+1)*10 for x in rs]
                #print(rs)
                #print(ss)
                ss=[x for i,x in sorted(zip(action2,ss))]
                ss.reverse()
                #print(ss)
                state_ = get_state(ss,rs)

                #print('STATE_')
                #print(state_)
                observation_ = torch.FloatTensor(np.array(state_,dtype=float))

                agent.store_transition(observation, action, reward, observation_, done)

                state=state_

                #agent.learn()

            '''
            for t in range(0,effective_length):
                #Get the current observation
                observation = [data.getFeatures()[x] for x in state]
                observation = torch.FloatTensor(np.array(observation,dtype=float))


                #Choose action based on the current observation
                action = agent.choose_action(observation)

                
                #Monitor actions to evaluate training
                action_list.append(state[action])

                #Update to next state,observation and get reward
                state_, reward = update_state(t, Q, state[action], state, data.getTruth())
                observation_ = [data.getFeatures()[x] for x in state_]
                observation_ = torch.FloatTensor(np.array(observation_,dtype=float))
                

                ##Feedback + reward
                ep_reward += reward
                
                if(len(state_)==0):
                    done=True

                #Store transition in the buffer
                agent.store_transition(observation, action, reward, 
                                        observation_, done)

                # #Learn
                # agent.learn()
                
                #Change the state
                state = state_
                
                if(done):
                    break
            '''
            #Update parameters based on the mini-batch
            #for i in range(5):
            agent.learn()

            score_history.append(ep_reward/effective_length)

            # Update Query DCG results:
            dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
            dcg_results[Q] = np.round(dcg_results[Q], 4)

            bar.next()

        # #Update parameters based on the mini-batch
        # agent.learn()

        bar.finish()
        #Current training results
        print(f'Avg Step Reward: {round(np.array(score_history).mean(), 4)}, time: {round(time.time() - ts)}')
        final_result = calculate(dcg_results)
        print(f"NDCG@1: {final_result[0]}\t"
              f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}\n")

        results['train'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(np.array(score_history).mean(), 4)))
        
        #Testing
        print("\nTest Results:")
        final_result,avgr=test(agent, data)
        results['test'].append((final_result[0],final_result[2],final_result[4],final_result[9],round(avgr, 4)))

    #Saving results for visualization and analysis
    ds=get_name(data_dir)
    fpath=f"./Results/{ds}/{ds}-i_{n_iterations}-lr_{lr}-g_{gamma}-replace_target_{replace_target}-singlelayer-uepisodic-hnodes_{hnodes}-length_{t_steps}-eps_start_{epsilon}-eps_end_{eps_end}-max_mem_{max_mem_size}-batchsize_{batch_size}-seed_{seed}-results"
    pickle_data(results,fpath)
    
    #Save the model
    checkpoint_path=f"./Results/{ds}/{ds}-i_{n_iterations}-lr_{lr}-g_{gamma}-replace_target_{replace_target}-singlelayer-uepisodic-hnodes_{hnodes}-length_{t_steps}-eps_start_{epsilon}-eps_end_{eps_end}-max_mem_{max_mem_size}-batchsize_{batch_size}-seed_{seed}-model"
    torch.save(agent.Q_eval.state_dict(), checkpoint_path)

    #View and save plots
    view_save_plot_avgreturn(results, fpath)
    view_save_plot_ndcg(results, fpath)