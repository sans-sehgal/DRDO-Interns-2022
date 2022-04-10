import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np

def make_plot_test(results, axs):
    """Takes the results dictionary, k value as input and plots a graph of train NDCG@k,validation NDCG@k and test NDCG@k"""
    tsvals= results
    axs.plot(tsvals,  color='red', label='testing')
    axs.grid()
    axs.legend()

def visualize_test(test_results, outfile):
    fig = plt.figure(figsize=(17, 17))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    ax1.set_title("NDCG@1")
    make_plot_test([x[0] for x in test_results], axs=ax1)

    ax2.set_title("NDCG@3")
    make_plot_test([x[2] for x in test_results], axs=ax2)

    ax3.set_title("NDCG@5")
    make_plot_test([x[4] for x in test_results], axs=ax3)

    ax4.set_title("NDCG@10")
    make_plot_test([x[9] for x in test_results], axs=ax4)

    plt.savefig(outfile + ".png")
    plt.show()

def make_plot_train(results, axs):
    """Takes the results dictionary, k value as input and plots a graph of train NDCG@k,validation NDCG@k and test NDCG@k"""
    tsvals= results
    axs.plot(tsvals,  color='blue', label='training')
    axs.grid()
    axs.legend()

def visualize_train(train_results, outfile):
    fig = plt.figure(figsize=(17, 17))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    ax1.set_title("NDCG@1")
    make_plot_train([x[0][0] for x in train_results], axs=ax1)

    ax2.set_title("NDCG@3")
    make_plot_train([x[0][2] for x in train_results], axs=ax2)

    ax3.set_title("NDCG@5")
    make_plot_train([x[0][4] for x in train_results], axs=ax3)

    ax4.set_title("NDCG@10")
    make_plot_train([x[0][9] for x in train_results], axs=ax4)

    plt.savefig(outfile + ".png")
    plt.show()

def visualize_rewards(train_results, outfile):
    plt.plot([np.mean(x[1]) for x in train_results], color='purple', label='Average step Reward')
    plt.grid()
    plt.legend()
    plt.savefig(outfile + ".png")
    plt.show()

# if __name__ == '__main__':
#     f = open("./Result/MQ2007/test_100_actor_lr_1e-05_critic_lr_2e-05_g_1.0_hnodes_46_T_50_mbsize_5_epochs_3", 'rb')
#     results = pickle.load(f)
#     outfile = './Result/MQ2007/test_100_actor_lr_1e-05_critic_lr_2e-05_g_1.0_hnodes_46_T_50_mbsize_5_epochs_3'
#     # print(results[8])

#     # plt.plot([np.mean(x[1]) for x in results], color='purple', label='Reward')
#     # plt.grid()
#     # plt.legend()
#     # plt.show()
#     visualize(results, outfile)