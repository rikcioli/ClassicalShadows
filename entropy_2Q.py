# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:01:28 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
from state import StabilizerState
import copy
import matplotlib.pyplot as plt
import tqdm
from collections import Counter
from numpy import savetxt
import warnings
import itertools

N_qubits = 2

sc = StabilizerCircuit(N_qubits)

# Prepare arbitrary initial state

sc.H(0)
sc.CNOT(0, 1)
sc.run()
print(sc.state)
saved_state = sc.state


N_shadows = 50
N_bins = 1
samples_per_bin = 10
starting_N_samples = 100
save_results = False
if not save_results:
    warnings.warn("WARNING: save_results set to False")

for depth in range(1, 5):
    
    purities_per_bin = []
    
    #fidelities = []
    
    #tN = 4.15*(N_shadows/100)**2 - 2.67*(N_shadows/100) + 1.95
    tN = (N_qubits/2)*depth*3.72*(N_shadows/100)**2 - 1.88*(N_shadows/100) + 1.12
    
    estimated_time = tN*N_bins*(starting_N_samples + samples_per_bin*(N_bins-1)/2)
    
    print("\nEvaluating purity with the following parameters:\nN_qubits =", N_qubits, "\ndepth =", depth, "\nN_shadows =", N_shadows, "\nN_samples:", N_bins, "bins starting from", starting_N_samples, "samples, step size", samples_per_bin)
    print("\nEstimated total time: ", round(estimated_time), "s =", round(estimated_time/60), "min")
    
    
    #Tot_N_shadows = N_shadows*N_bins*(starting_N_samples + samples_per_bin*(N_bins-1)/2)
    #shadows = sc.saveShadows(Tot_N_shadows, depth)
    
    
    for bin_number in range(0, N_bins):
        N_samples = starting_N_samples + samples_per_bin*bin_number
        purity_samples = []
        
        for sample_number in tqdm.tqdm(range(N_samples), desc="Evaluating "+str(N_shadows)+" shadows with "+str(N_samples)+ " samples..."):
            
            # Extract classical shadows     
            shadows = sc.saveShadows(N_shadows, depth)
            random_circuits = shadows[0]
            outcomes = shadows[1]
            
            inverse_circuits = copy.deepcopy(random_circuits)
            [circuit.reverse() for circuit in inverse_circuits]
            
            # Compute global purity
            Udagb_states = [StabilizerCircuit(N_qubits, circuit = inverse_circuits[i]) for i in range(N_shadows)]
            for i in range(N_shadows):
                Udagb_states[i].state[N_qubits:2*N_qubits, 2*N_qubits] = np.array(outcomes[i])
                [gate.dagger() for gate in Udagb_states[i].circuit]
            
            [state.run() for state in Udagb_states]
                
            
            # U_Udagb_states = [StabilizerCircuit(N_qubits, Udagb_states[s].state, random_circuits[r]) for s, r in itertools.combinations(range(N_shadows), 2)]
            # outcomes_combinations = (outcomes[r] for s, r in itertools.combinations(range(N_shadows), 2))
            # [state.run() for state in U_Udagb_states]
            # main_trace = sum((state.dot_outcome(outcome)**2 for state, outcome in zip(U_Udagb_states, outcomes_combinations)))
            
            main_trace = 0
            for s, state_s in enumerate(Udagb_states):
                current_state = state_s.state.copy()
                for r in range(s+1, N_shadows):
                    state_s.circuit = random_circuits[r]
                    [state_s.X(i) for i in range(N_qubits) if outcomes[r][i]]
                    state_s.run()
                    main_trace += abs(state_s.dot_zero())**2
                    #fidelities.append(abs(state_s.dot_zero())**2)
                    state_s.state = current_state
                    
            purity = main_trace*2*((2**N_qubits+1)**2)/(N_shadows*(N_shadows-1)) - (2**N_qubits+2)
            purity_samples.append(purity)
        
        purity_samples = np.array(purity_samples)
        purities_per_bin.append(purity_samples)
        if save_results:
            savetxt('Results/Purity/'+str(N_shadows)+'-'+str(N_samples)+'_purity_samples.csv', purity_samples, delimiter=',')
    
    avg_purity = [np.mean(purities_i) for purities_i in purities_per_bin]
    median_purity = [np.median(purities_i) for purities_i in purities_per_bin]
    sdom_purity = [np.std(purities_i)/np.sqrt(len(purities_i)) for purities_i in purities_per_bin]
    x_axis = [starting_N_samples+samples_per_bin*i for i in range(N_bins)]
    
    print("Avg purity =", avg_purity)
    print("Sdom purity =", sdom_purity)
    
    plt.figure(dpi=600)
    plt.errorbar(x_axis, avg_purity, yerr = sdom_purity)
    plt.xlabel(r"$N_{samples}$")
    plt.ylabel("Purity")
    plt.title(r'$N_{shadows} = $'+str(N_shadows)+' depth = '+str(depth), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()  
    
    #fidelity_dict = dict(Counter(fidelities))
    
    if save_results:
        # save to csv file
        savetxt('Results/Purity/'+str(N_shadows)+'-'+str(starting_N_samples)+'-'+str(N_bins)+'-'+str(samples_per_bin)+'_purity_x_axis.csv', x_axis, delimiter=',')
        savetxt('Results/Purity/'+str(N_shadows)+'-'+str(starting_N_samples)+'-'+str(N_bins)+'-'+str(samples_per_bin)+'_purity_avg.csv', avg_purity, delimiter=',')
        savetxt('Results/Purity/'+str(N_shadows)+'-'+str(starting_N_samples)+'-'+str(N_bins)+'-'+str(samples_per_bin)+'_purity_median.csv', median_purity, delimiter=',')
        savetxt('Results/Purity/'+str(N_shadows)+'-'+str(starting_N_samples)+'-'+str(N_bins)+'-'+str(samples_per_bin)+'_purity_sdom.csv', sdom_purity, delimiter=',')
