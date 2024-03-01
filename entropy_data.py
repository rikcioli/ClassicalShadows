# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:45:12 2024

@author: rikci
"""

import numpy as np
import matplotlib.pyplot as plt



for N_qubits in range(6, 8, 2):
    avgent_per_depth = []
    sdoment_per_depth = []
    N_shadows = 50
    N_samples = 1000
    depths = [i for i in range(1, 11)]
    for depth in depths:
        # load array
        ent_per_sample = np.loadtxt('Results/Entropy/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_entropy_per_sample.csv', delimiter=',')
        avgent_per_depth.append(np.mean(ent_per_sample))
        sdoment_per_depth.append(np.std(ent_per_sample)/np.sqrt(N_samples))
    
    plt.figure(dpi=600)
    plt.errorbar(depths, avgent_per_depth, yerr = sdoment_per_depth, fmt='bo')
    plt.xlabel(r"$Depth$")
    plt.ylabel("Renyi2")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()