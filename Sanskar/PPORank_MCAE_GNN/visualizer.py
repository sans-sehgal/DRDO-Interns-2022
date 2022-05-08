import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def make_plot(results, axs, pos):
    """Takes the results dictionary, k value as input and plots a graph of train NDCG@k,validation NDCG@k and test NDCG@k"""
    indexer = {0: 1, 1: 3, 2: 5, 3: 10}
    tvals = []
    vvals = []
    tsvals= []
    for i in results['train']:
        tvals.append(i[pos])
    #for i in results['validation']:
    #    vvals.append(i[pos])
    for i in results['test']:
        tsvals.append(i[pos])
    print("Initial NDCG@",indexer[pos],":",tsvals[0]," Final NDCG@",indexer[pos],":",tsvals[-1])
    #smooth_tvals=pd.Series.rolling(pd.Series(tvals), 10).mean()
    axs.plot(tvals, color='green', label='training')
    #axs.plot(vvals, marker='o', markersize='5', color='orange', linestyle='dashed', label='validation')
    axs.plot(tsvals,  color='blue', label='testing')
    axs.grid()
    axs.legend()
    # plt.plot(tvals, markersize='7', color='green', label='training')
    # plt.plot(smooth_tvals, linewidth=3,color='blue', label='Smooth NDCG')
    # plt.plot(vvals, marker='o', markersize='5', color='orange', linestyle='dashed', label='validation')
    # plt.plot(tsvals, markersize='7', color='red', label='testing')
    # plt.legend()
    # plt.show()
    

ap=argparse.ArgumentParser()
ap.add_argument("-d","--datadir",required=True,help="Path to data file to visualize")
args=vars(ap.parse_args())
path=str(args["datadir"])


with open(path,'rb') as handle:
    results=pickle.load(handle)

fig = plt.figure(figsize=(17, 17))
grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6, figure=fig)
# ax0 = fig.add_subplot(grid[0, :])
# ax1 = fig.add_subplot(grid[1, 0])
# ax2 = fig.add_subplot(grid[1, 1])
# ax3 = fig.add_subplot(grid[2, 0])
# ax4 = fig.add_subplot(grid[2, 1])
ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, 0])
ax4 = fig.add_subplot(grid[1, 1])


#Trends for NDCG@1,3,5,10 and NDCG@K
#ax0.set_title("NDCG@K")
#make_plot(results, axs=ax0, pos=0)

ax1.set_title("NDCG@1")
make_plot(results, axs=ax1, pos=0)

ax2.set_title("NDCG@3")
make_plot(results, axs=ax2, pos=1)

ax3.set_title("NDCG@5")
make_plot(results, axs=ax3, pos=2)

ax4.set_title("NDCG@10")
make_plot(results, axs=ax4, pos=3)

plt.show()