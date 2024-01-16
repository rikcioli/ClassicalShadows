# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:07:13 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
from state import StabilizerState
import copy
import matplotlib.pyplot as plt
import tqdm
import colorama
from collections import Counter
from numpy import savetxt
import warnings

depths = [i for i in range(1, 11)]
min_qubits = 6
max_qubits = 6

avgfid_per_N_qubits = []
sdomfid_per_N_qubits = []

save_results = False
if not save_results:
    print("WARNING: save_results set to False", flush=True)

for N_qubits in range(min_qubits, max_qubits+2, 2):

    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    
    sc.randClifford([a for a in range(N_qubits)])
    sc.run()
    phi = sc.state.copy()
    
    N_shadows = 50
    N_samples = 2
    avgfid_per_depth = []
    sdomfid_per_depth = []
    
    print("Running program with "+str(min_qubits)+" to "+str(max_qubits)+" qubits", flush=True)
    print("N_shadows = "+str(N_shadows)+", N_samples =", N_samples)
    
    for depth in tqdm.tqdm(depths, desc="Evaluating circuit depths up to "+str(depths[-1])):
        
        shadows = sc.saveShadows(N_shadows*N_samples, depth)      
        random_circuits = shadows[0]
        outcomes = shadows[1]
        
        fidelity_per_sample = np.zeros(N_samples)
        
        for sample_number in range(N_samples):
            
            circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
            outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
            Uphi_states = [StabilizerCircuit(N_qubits, phi, circuits_current_sample[r]) for r in range(N_shadows)]
            [[Uphi_states[r].X(i) for i in range(N_qubits) if outcomes_current_sample[r][i]] for r in range(N_shadows)]
            [state.run() for state in Uphi_states]
            
            fidelity = ((2**N_qubits+1)/N_shadows)*sum((state.dot_zero()**2 for state in Uphi_states)) - 1 
            fidelity_per_sample[sample_number] = fidelity
        
        avg_fidelity = np.mean(fidelity_per_sample)
        sdom_fidelity = np.std(fidelity_per_sample)/np.sqrt(N_samples)
        if save_results:
            savetxt('Results/Fidelity/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_avg_fidelity.csv', avg_fidelity, delimiter=',')
            savetxt('Results/Fidelity/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_sdom_fidelity.csv', sdom_fidelity, delimiter=',')
        
        avgfid_per_depth.append(avg_fidelity)
        sdomfid_per_depth.append(sdom_fidelity)
    
    plt.figure(dpi=600)
    plt.errorbar(depths, avgfid_per_depth, yerr = sdomfid_per_depth)
    plt.xlabel(r"$Depth$")
    plt.ylabel("Fidelity")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()  
    
    avgfid_per_N_qubits.append(avgfid_per_depth)
    sdomfid_per_N_qubits.append(sdomfid_per_depth)
    
