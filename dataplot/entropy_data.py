# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:45:12 2024

@author: rikci
"""

import numpy as np
import matplotlib.pyplot as plt

start_N = 4
N_list = list(range(start_N, 8, 2))

avgent_perN_perdepth = []
sdoment_perN_perdepth = []

for N_qubits in N_list:
    avgent_per_depth = []
    sdoment_per_depth = []
    N_shadows = 50
    N_samples = 1000
    depths = [i for i in range(1, 11)]
    for depth in depths:
        # load array
        ent_per_sample = np.loadtxt('D:/SPYDER BACKUP/.spyder-py3/CliffordCircuits/Results/Entropy/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_entropy_per_sample.csv', delimiter=',')
        avgent_per_depth.append(np.mean(ent_per_sample))
        sdoment_per_depth.append(np.std(ent_per_sample)/np.sqrt(N_samples))
    
    avgent_perN_perdepth.append(np.array(avgent_per_depth))
    sdoment_perN_perdepth.append(np.array(sdoment_per_depth))
    
    plt.figure(dpi=600)
    plt.errorbar(depths, avgent_per_depth, yerr = sdoment_per_depth, fmt='bo')
    plt.xlabel(r"$Depth$")
    plt.ylabel("Renyi2")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()
    

start_depth = 1

avgent_perN_perdepth = np.array(avgent_perN_perdepth)
sdoment_perN_perdepth = np.array(sdoment_perN_perdepth)
    
plt.figure(dpi=600)
for i in range(len(N_list)):
    plt.errorbar(depths[start_depth-1:], avgent_perN_perdepth[i][start_depth-1:], yerr = sdoment_perN_perdepth[i][start_depth-1:], label = "N = "+str(start_N+2*i))
plt.xticks(depths[start_depth-1:])
plt.xlabel(r"t")
plt.yticks(np.arange(0, 9.5))
plt.ylabel("Renyi2")
# plt.ylim(0, 10)
plt.legend()
plt.title(r'$M = $'+str(N_shadows)+', $N_{s} = $'+str(N_samples), fontsize=16)
plt.grid(axis = 'y', linestyle = '--')
plt.show()