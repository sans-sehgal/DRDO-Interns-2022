# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
#
# for index in range(1, 4):
#     f = open(f"{index}_train_50_0.05_0.99_45", 'rb')
#     t = pickle.load(f)
#     print(t)
#     print()
#     f = open(f"{index}_vali_50_0.05_0.99_45", 'rb')
#     t = pickle.load(f)
#     print(t)
#     print()
#     f = open(f"{index}_test_50_0.05_0.99_45", 'rb')
#     t = pickle.load(f)
#     print(t)
#     print()
#     f = open(f"{index}_train_rewards_50_0.05_0.99_45", 'rb')
#     all_rewards = pickle.load(f)
#     print(all_rewards)
#     smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 1).mean()
#     smoothed_rewards = [elem for elem in smoothed_rewards]
#     plt.plot(all_rewards)
#     plt.plot(smoothed_rewards)
#     plt.plot()
#     plt.xlabel('Epoch')
#     plt.ylabel('Reward')
#     plt.show()

import pickle
import pandas as pd
import matplotlib.pyplot as plt

for index in range(3, 4):
    f = open(f"_test_50_5e-05_0.0001_1.0_", 'rb')
    t = pickle.load(f)
    print(t)
    NDCG_train = [x[0] for x in t]
    print()
    f = open(f"_train_50_5e-05_0.0001_1.0_", 'rb')
    t = pickle.load(f)
    # print(t)
    print()
    # smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 1).mean()
    # smoothed_rewards = [elem for elem in smoothed_rewards]
    # plt.plot(all_rewards)
    # plt.plot(smoothed_rewards)
    plt.plot(NDCG_train)
    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Reward')
    plt.show()

