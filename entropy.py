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

N_qubits = 2

sc = StabilizerCircuit(N_qubits)


# sc.H(0)
# sc.CNOT(0, 1)
# sc.run()
# print(sc.state)
# sc.circuit = []

# sc.randClifford([0,3])
# sc.randClifford([0,2])
# sc.run()
# print(sc.state)

# save_circuit = copy.deepcopy(sc.circuit)
# [gate.dagger() for gate in sc.circuit]
# sc.circuit.reverse()
# #sc.state = saved_state
# sc.run()
# print(sc.state)

# Prepare arbitrary initial state

sc.H(0)
sc.CNOT(0, 1)
sc.run()
print(sc.state)
saved_state = sc.state


depth = 1
N_samples = 10
N_bins = 5
shadows_per_bin = 20
save_results = False

purities = np.zeros([N_bins, N_samples])
B = N_bins
f = shadows_per_bin/100

estimated_time = 4.15*(f**2)*B*(B+1)*(2*B+1)/6 - 2.67*f*B*(B+1)/2 + 1.95*B
estimated_time *= N_samples
print("Evaluating purity for", N_bins, "bins \nEstimated total time: ", round(estimated_time), "s =", round(estimated_time/60), "min")

for bin_number in range(1, N_bins+1):
    N_shadows = shadows_per_bin*bin_number
    for sample_number in tqdm.tqdm(range(N_samples), desc="Evaluating "+str(N_shadows)+" shadows with "+str(N_samples)+ " samples..."):
        
        # Extract classical shadows     
        random_circuits, outcomes = sc.saveShadows(N_shadows, depth)
        
        inverse_circuits = copy.deepcopy(random_circuits)
        [circuit.reverse() for circuit in inverse_circuits]
        
        # Compute global purity
        Udagb_states = [StabilizerCircuit(N_qubits, circuit = inverse_circuits[i]) for i in range(N_shadows)]
        for i in range(N_shadows):
            Udagb_states[i].state[N_qubits:2*N_qubits, 2*N_qubits] = np.array(outcomes[i])
            [gate.dagger() for gate in Udagb_states[i].circuit]
        
        [state.run() for state in Udagb_states]
            
        main_trace = 0
        for s, state_s in enumerate(Udagb_states):
            current_state = state_s.state.copy()
            for r in range(s+1, N_shadows):
                state_s.circuit = random_circuits[r]
                [state_s.X(i) for i in range(N_qubits) if outcomes[r][i]]
                state_s.run()
                main_trace += abs(state_s.dot_zero())**2
                state_s.state = current_state
                
        purity = main_trace*2*((2**N_qubits+1)**2)/(N_shadows*(N_shadows-1)) - (2**N_qubits+2)
        purities[bin_number-1, sample_number] = purity

avg_purity = [np.mean(purities[row]) for row in range(bin_number)]
median_purity = [np.median(purities[row]) for row in range(bin_number)]
x_axis = [shadows_per_bin*i for i in range(1, N_bins+1)]

if N_samples>1:
    std_purity = [np.std(purities[row]) for row in range(bin_number)]
    plt.figure(dpi=600)
    plt.errorbar(x_axis, median_purity, yerr = std_purity)
    plt.xlabel("N of shadows")
    plt.ylabel("Purity")
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()  

from numpy import savetxt

if save_results:
    # save to csv file
    savetxt('Results/Purity/'+str(shadows_per_bin)+'-'+str(N_bins)+'-'+str(N_samples)+'_purity_x_axis.csv', x_axis, delimiter=',')
    savetxt('Results/Purity/'+str(shadows_per_bin)+'-'+str(N_bins)+'-'+str(N_samples)+'_purity_avg.csv', avg_purity, delimiter=',')
    savetxt('Results/Purity/'+str(shadows_per_bin)+'-'+str(N_bins)+'-'+str(N_samples)+'_purity_median.csv', median_purity, delimiter=',')
    if N_samples>1:
        savetxt('Results/Purity/'+str(shadows_per_bin)+'-'+str(N_bins)+'-'+str(N_samples)+'_purity_std.csv', std_purity, delimiter=',')