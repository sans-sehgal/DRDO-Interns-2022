import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def view_save_plot_avgreturn(results,outfile):
    """Plots the avg step reward vs epoch and saves it in outfile"""
    lst_avgreturns=[]
    for i in results['train']:
        lst_avgreturns.append(i[-1])
    fig = plt.figure(figsize=(26, 17))
    plt.title("DQN")
    plt.ylabel("Avg Step Reward")
    plt.xlabel("Epochs")
    plt.plot(lst_avgreturns, color='blue', label='Avg Step Reward')
    plt.grid()
    plt.legend()
    plt.savefig(outfile+"_avgstepreward.png")
    plt.show()


def make_plot(results, axs, pos):
    """Takes the results dictionary, k value as input and plots a graph of NDCG@k on the given axis"""
    #Maps index to ndcg position
    indexer = {0: 1, 1: 3, 2: 5, 3: 10}
    tvals = []
    vvals = []
    tsvals= []
    for i in results['train']:
        tvals.append(i[pos])
    #for i in results['validation']:
    #    vvals.append(i[pos])
    #for i in results['test']:
    #    tsvals.append(i[pos])

    #print("Initial NDCG@",indexer[pos],":",tsvals[0]," Final NDCG@",indexer[pos],":",tsvals[-1])
    axs.plot(tvals, color='green', label='training')
    #axs.plot(tsvals,  color='red', label='testing')
    axs.grid()
    axs.legend()


def view_save_plot_ndcg(results,outfile):
    """Plots the ndcg 1,3,5,10 graphs and saves it in outfile"""
    fig = plt.figure(figsize=(26, 17))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6, figure=fig)

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])


    ax1.set_title("NDCG@1")
    make_plot(results, axs=ax1, pos=0)

    ax2.set_title("NDCG@3")
    make_plot(results, axs=ax2, pos=1)

    ax3.set_title("NDCG@5")
    make_plot(results, axs=ax3, pos=2)

    ax4.set_title("NDCG@10")
    make_plot(results, axs=ax4, pos=3)

    plt.savefig(outfile+"_ndcg.png")
    plt.show()