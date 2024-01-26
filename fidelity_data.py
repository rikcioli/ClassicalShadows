# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:45:12 2024

@author: rikci
"""

import numpy as np
import matplotlib.pyplot as plt



for N_qubits in range(20, 22, 2):
    avgfid_per_depth = []
    sdomfid_per_depth = []
    N_shadows = 50
    N_samples = 1000
    depths = [i for i in range(1, 21)]
    for depth in depths:
        # load array
        fid_per_sample = np.loadtxt('Results/Fidelity/GHZ 20Q/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fidelity_per_sample.csv', delimiter=',')
        avgfid_per_depth.append(np.mean(fid_per_sample))
        sdomfid_per_depth.append(np.std(fid_per_sample)/np.sqrt(N_samples))
    
    plt.figure(dpi=600)
    plt.errorbar(depths, avgfid_per_depth, yerr = sdomfid_per_depth, fmt='bo')
    plt.xlabel(r"$Depth$")
    plt.ylabel("Fidelity")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()