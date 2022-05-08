import pickle 
import matplotlib.pyplot as plt
import argparse
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("-d","--datadir",required=True,help="Path to data file to visualize")
args=vars(ap.parse_args())
path=str(args["datadir"])


with open(path,'rb') as handle:
	results=pickle.load(handle)

lst_avgreturns=[]
for i in results['train']:
	lst_avgreturns.append(i[-1])

plt.title("PPO(MC AE)")
plt.ylabel("Avg Step Returns")
plt.xlabel("Iterations")
plt.plot(lst_avgreturns, markersize='5', color='blue', label='Avg Step Returns')
plt.grid()
plt.legend()
plt.show()
