from Environment import Dataset, update_state
from Evaluation import validate_individual, calculate
from Agent import Agent
import time
from progress.bar import Bar
import numpy as np
import os
import torch
import pickle
import argparse
import random
import scipy 
import numpy as np 
from visualizer import visualize_test, visualize_train, visualize_rewards

np.random.seed(4)
random.seed(4)
torch.manual_seed(4)

# python main3.py -d ./data/MQ2008/all_0,1,2 -nf 46 -e 200
'''
Q-> current qurey
action_list -> list of documets
action_list2 ->feature vectors of the docs in action_list & query 
data -> dataset (refer to LoadData for format) 
'''

'''
Returns a list of similarity scores with the vector received from user feedback
similarity_score[] -> list of similarity scores
'''
def similarity_score(action_list2, data_vec):
    similarity_score = []
    for action in action_list2:
        # print(action)
        action = action[:46]
        try:
            cos_distance = scipy.spatial.distance.cosine(data_vec, action)
            cos_similarity = 1 - cos_distance
            similarity_score.append(cos_similarity)
        except:
        #     print("skipped")
            continue
    return similarity_score

def feedback(Q,action_list,action_list2,data,iter):
    for i,j in enumerate(action_list):
        print(i,'\t',j)
    
    val = input("Enter Feedback \n")
    f = [0]*len(action_list)
    
    positive=np.zeros(46,dtype=float)
    negative=np.zeros(46,dtype=float)
    for i in val.split():
        f[int(i)]=1
        #print(Q,action_list[int(i)])
        similarity_list = similarity_score(action_list2,action_list2[int(i)])
        for s in range(len(action_list)):
            data.updateRelevance(Q,action_list[int(s)],similarity_list[int(s)])

    print(data.QUERY_DOC_TRUTH[Q])
    #print(similarity_list)
    #print(data.MAX_DCG[Q])
    data.updateIDCG(Q)
    #print(data.MAX_DCG[Q])

    for i in f:
        if i:
            positive = positive + np.array(action_list2[i][0][:46], dtype=float)
        else:
            negative = negative + np.array(action_list2[i][0][:46], dtype=float)
    
    n = len(val.split())
    positive = positive/n
    negative = negative/(max(0,len(action_list)-n))
    #query=(1-gamma*(b-c))*query + gamma*(b*positive-c*negative)
    query = action_list2[0][0][46:]
    query = (1-(0.9**iter)*0.5)*query + (0.9**iter)*(0.75*positive-0.25*negative)
    #print(query)
    return query



# Training the model
def train(model, data, episode_length,epoch):

    # Setting pytorch to training mode
    model.actor.train()
    model.critic.train()

    # Buffer for rewards to perform average later
    episode_rewards_list = []
    epoch_avg_step_reward = 0

    dcg_results = {}
    ts = time.time()
    bar = Bar('Training', max=len(data.getTrain()))

    # For each query in the training set
    for Q in data.getTrain():        
        for iter in range(5):
            dcg_results[Q] = []

            qvec = data.getQVEC(Q)
            # Initialize state with Total number of documents
            state = data.getDocQuery()[Q]
            action_list = []
            action_list2=[]
            action_list3 = []
            episode_reward = 0
            #effective_length = min(episode_length, len(state))
            effective_length = len(state)

            for t in range(0, effective_length):

                # Converting the current state into numpy array to make into tensors later
                observation = [data.getFeatures()[x] for x in state]
                observation = np.array(observation, dtype=float)

                qvec_temp = np.array(qvec).reshape(1,46)
                
                observation = np.concatenate([observation,np.repeat(qvec_temp,observation.shape[0],axis=0)],axis=1)

                # Actor chooses an action (a document at t position)
                action = model.choose_action(observation,action_list2)
                
                # all actions stored in buffer for calculation of DCG scores later
                action_list.append(state[action])
                action_list2.append(np.concatenate([np.array(data.getFeatures()[state[action]], dtype=float).reshape(1,46),qvec_temp ],axis=1))
                #print(action_list2)

                #action_list3.append(data.getFeatures()[state[action]])
                

                # Get the next state and the reward based on the action
                state_, reward = update_state(t, Q, state[action], state, data.getTruth())


                episode_reward += reward
                observation_ = [data.getFeatures()[x] for x in state_]
                observation_ = np.array(observation_, dtype=float)

                if observation_.shape[0] != 0:
                    observation_ = np.concatenate([observation_,np.repeat(qvec_temp,observation_.shape[0],axis=0)],axis=1)

                # Update agent parameters
                model.update(observation, reward, observation_, action_list2)

                # Update state
                state = state_

            epoch_avg_step_reward += episode_reward / effective_length
            episode_rewards_list.append(episode_reward / effective_length)
            
            # if epoch==0:
            updatedQuery = feedback(Q,action_list,action_list2,data,iter)
            data.updateQVEC(Q,updatedQuery)
             
            
            # Update Query DCG results:
            dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
            dcg_results[Q] = np.round(dcg_results[Q], 4)

        bar.next()
        #print('\n')


    bar.finish()
    print(
    f'\n Average step reward: {round(epoch_avg_step_reward / len(data.getTrain()), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result, episode_rewards_list


def init_train(model, data):

    # Setting pytorch to evaluation mode (stops updating the gradients)
    model.actor.eval()
    model.critic.eval()

    # Buffer for rewards to perform average later
    ep_reward = 0
    episode_rewards_list = []
    epoch_avg_step_reward = 0

    dcg_results = {}
    ts = time.time()
    bar = Bar('Initial Training', max=len(data.getTrain()))

    # For each query in the training set
    for Q in data.getTrain():
        dcg_results[Q] = []
        
        # Initialize state with Total number of documents
        state = data.getDocQuery()[Q]
        action_list = []
        action_list2=[]
        action_list3=[]
        episode_reward = 0
        effective_length = len(state)

        for t in range(0, len(state)):
            #print(t)
            # Converting the current state into numpy array to make into tensors later
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)

            # Actor chooses an action (a document at t position)

            action = model.choose_action_test(observation,action_list2)

            # all actions stored in buffer for calculation of DCG scores later
            action_list.append(state[action])
            action_list2.append(data.getFeatures()[state[action]])

            # Get the next state and the reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            episode_reward += reward

            # Update state
            state = state_



        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)

        # Average the rewards with by the number of documents
        epoch_avg_step_reward += episode_reward / effective_length
        episode_rewards_list.append(episode_reward / effective_length)

        bar.next()

    bar.finish()
    print(
    f'\n Average step reward: {round(epoch_avg_step_reward / len(data.getTrain()), 4)}, time: {round(time.time() - ts)}')
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result, episode_rewards_list


def test(model, data):
    
    # Setting pytorch to evaluation mode (stops updating the gradients)
    model.actor.eval()
    model.critic.eval()
    ep_reward = 0
    dcg_results = {}
    ts = time.time()
    bar = Bar('Testing', max=len(data.getTest()))

    for Q in data.getTest():
        dcg_results[Q] = []
        state = data.getDocQuery()[Q]
        action_list = []
        action_list2=[]
        qvec = data.getQVEC(Q)
        Q_reward = 0
        for t in range(0, len(state)):
            observation = [data.getFeatures()[x] for x in state]
            observation = np.array(observation, dtype=float)
            qvec_temp = np.array(qvec).reshape(1,46)
            observation = np.concatenate([observation,np.repeat(qvec_temp,observation.shape[0],axis=0)],axis=1)



            # Note: Actor takes action and returns index of the state
            action = model.choose_action_test(observation,action_list2)
            action_list.append(state[action])
            #action_list2.append(data.getFeatures()[state[action]])
            action_list2.append(np.concatenate([np.array(data.getFeatures()[state[action]], dtype=float).reshape(1,46),qvec_temp ],axis=1))

            # Update to next state and get reward
            state_, reward = update_state(t, Q, state[action], state, data.getTruth())

            # Update state
            state = state_

        # Update Query DCG results:
        dcg_results[Q] = validate_individual(data.getTruth()[Q], data.getIDCG()[Q], action_list)
        dcg_results[Q] = np.round(dcg_results[Q], 4)
        bar.next()

    bar.finish()
    final_result = calculate(dcg_results)
    print(f"NDCG@1: {final_result[0]}\t"
          f"NDCG@3: {final_result[2]}\tNDCG@5: {final_result[4]}\tNDCG@10: {final_result[9]}")

    return final_result

def get_name(datadir):
    lst = datadir.split('/')
    ds = ""
    for i in lst:
        if 'ohsumed' in i.lower():
            ds = 'OHSUMED'
        elif 'mq2008' in i.lower():
            ds = 'MQ2008'
        elif 'mq2007' in i.lower():
            ds = 'MQ2007'
        elif 'mslr-web10k' in i.lower():
            ds = 'MSLR-WEB10K'
    if len(ds) == 0:
        print("Wrong Dataset,Please check path")
        exit()
    else:
        return ds

def pickle_data(data, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    output_handle = open(output_file, 'wb')
    pickle.dump(data, output_handle)
    output_handle.close()


if __name__ == '__main__':

    # Taking arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="relative or absolute directory of the directory where fold exists")
    ap.add_argument("-e", "--epoch", required=True,
                    help="number of epochs to run for")
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-g", "--gamma", required=False, default=1,
                    help="Gamma value; default = 1")
    ap.add_argument("-alpha", "--alpha", required=False, default=1e-05,
                    help="Learning rate of the actor")
    ap.add_argument("-beta", "--beta", required=False, default=2e-05,
                    help="Learning rate of the critic")
    ap.add_argument("-length", "--episode_length", required=False, default=20,
                    help="Episode length")
    args = vars(ap.parse_args())

    # Initializing arguments
    data_dir = str(args["data"])
    num_features = int(args["features_no"])
    num_epochs = int(args["epoch"])
    gamma = float(args["gamma"])
    alpha = float(args["alpha"])
    beta = float(args["beta"])
    episode_length = int(args['episode_length'])

    # Agent object initialization
    agent = Agent(actor_lr=alpha, critic_lr=beta, input_dims=num_features,
                  gamma=gamma)

    # Dataset object initialization
    dataset = f"{data_dir}"
    data_object = Dataset(dataset)

    print("\n--- Training Started ---\n")

    train_results = []
    test_results = []

    # Initial results on the current weights
    #train_results.append(init_train(agent, data_object))
    #test_results.append(test(agent, data_object))

    #feedback_results = []
    for i in range(0, num_epochs):
        print(f"\nEpoch: {i+1}\n")
        train_results.append(train(agent, data_object, episode_length,i))
        #feedback_results.append(train(agent, data_object, episode_length,i))
        test_results.append(test(agent, data_object))


    # Saving all results
    pickle_data(train_results, f"./Result/{get_name(data_dir)}/_train_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}")

    pickle_data(test_results, f"./Result/{get_name(data_dir)}/_test_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}")

    outfile = f"./Result/{get_name(data_dir)}/_train_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}"
    visualize_train(train_results, outfile)

    outfile = f"./Result/{get_name(data_dir)}/_test_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}"
    visualize_test(test_results, outfile)
    
    outfile = f"./Result/{get_name(data_dir)}/_rewards_epochs{num_epochs}_a,c{alpha},{beta}_g{gamma}_length{episode_length}"
    visualize_rewards(train_results, outfile)
