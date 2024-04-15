# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:45:12 2024

@author: rikci
"""

import numpy as np
import matplotlib.pyplot as plt

avgfid_perN_perdepth = []
sdomfid_perN_perdepth = []

start_N = 4
N_list = list(range(start_N, 42, 2))

for N_qubits in N_list:
    avgfid_per_depth = []
    sdomfid_per_depth = []
    N_shadows = 50
    N_samples = 1000
    depths = [i for i in range(1, 19)]
    for depth in depths:
        # load array
        fid_per_sample = np.loadtxt('D:/SPYDER BACKUP/.spyder-py3/CliffordCircuits/Results/Fidelity/Random3 4-40/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fidelity_per_sample.csv', delimiter=',')
        avgfid_per_depth.append(np.mean(fid_per_sample))
        sdomfid_per_depth.append(np.std(fid_per_sample)/np.sqrt(N_samples))
    
    avgfid_perN_perdepth.append(np.array(avgfid_per_depth))
    sdomfid_perN_perdepth.append(np.array(sdomfid_per_depth))
    
    # plt.figure(dpi=600)
    # plt.errorbar(depths, avgfid_per_depth, yerr = sdomfid_per_depth, fmt='bo')
    # plt.xlabel(r"$Depth$")
    # plt.ylabel("Fidelity")
    # plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    # plt.grid(axis = 'y', linestyle = '--')
    # plt.show()
    
start_depth = 1

avgfid_perN_perdepth = np.array(avgfid_perN_perdepth)
sdomfid_perN_perdepth = np.array(sdomfid_perN_perdepth)
    
plt.figure(dpi=600)
for i in range(len(N_list)):
    plt.errorbar(depths[start_depth-1:], avgfid_perN_perdepth[i][start_depth-1:], yerr = sdomfid_perN_perdepth[i][start_depth-1:], label = "N = "+str(start_N+2*i))
plt.xticks(depths[start_depth-1:])
plt.xlabel(r"$t$")
plt.xlim(start_depth, 8)
plt.ylabel("Fidelity")
# plt.ylim(0, 10)
plt.legend(loc = 'upper right', bbox_to_anchor=(1.28, 1.1))
plt.title(r'$M = $'+str(N_shadows)+', $N_{s} = $'+str(N_samples), fontsize=16)
plt.grid(axis = 'y', linestyle = '--')
plt.show()

eps_list = [0.1*2**(-j) for j in range(5)]
tbar_pereps_perN = []

for eps in eps_list:
    tbar = []
    for i in range(len(N_list)):
        for j, avgfid in enumerate(avgfid_perN_perdepth[i]):
            err = sdomfid_perN_perdepth[i, j]
            if abs(avgfid-1) < eps+err:
                depth = j+1
                tbar.append(depth)
                break
    tbar_pereps_perN.append(tbar)


plt.figure(dpi=600)
for i in range(len(eps_list)):
    plt.plot(N_list, tbar_pereps_perN[i], '-o', label = r"$\epsilon$ = "+str(eps_list[i]))


plt.xlabel("N")
plt.ylabel(r"$\bar{t}$")
# plt.ylim(0, 7.2)
plt.legend(loc = 'lower right')
plt.title(r'$M = $'+str(N_shadows)+', $N_{s} = $'+str(N_samples), fontsize=16)
plt.grid(axis = 'y', linestyle = '--')
plt.show()
