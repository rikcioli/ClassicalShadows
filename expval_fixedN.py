# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:44 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
import tqdm
import copy
from numpy import savetxt
from multiprocessing import Pool
import warnings


def evaluate_expval(N_qubits, depth, pauli_stab, N_shadows = 50, N_samples = 1000, save_results = True):
    
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    sc.run()
        
    shadows = sc.saveShadows(N_shadows*N_samples, depth)      
    random_circuits = shadows[0]
    outcomes = shadows[1]
        
    expval_per_sample = np.empty(N_samples)
        
    for sample_number in tqdm.tqdm(range(N_samples)):
            
        # Extract current sample's shadows
        circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
        outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
        
        # Extract Udag
        circuits_dag = copy.deepcopy(circuits_current_sample)
        [circuit.reverse() for circuit in circuits_dag]
        [gate.dagger() for circ in circuits_dag for gate in circ]
        
        # Prepare Udag|b> states
        Udagb_states = [StabilizerCircuit(N_qubits) for r in range(N_shadows)]
        [[Udagb_states[r].X(i) for i in range(N_qubits) if outcomes_current_sample[r][i]] for r in range(N_shadows)]
        for r in range(N_shadows):
            Udagb_states[r].circuit += circuits_dag[r]
            Udagb_states[r].run()
        
        # Evaluate expval of pauli_stab over Udag|b>
        expvals = [state.expval(pauli_stab) for state in Udagb_states]
        expval_per_sample[sample_number] = (2**N_qubits+1)*sum(expvals)/N_shadows
        if sum(pauli_stab) == 0:
            expval_per_sample[sample_number] -= 2**N_qubits
        
    mean_expval = np.mean(expval_per_sample)     # CAN ALSO BE REPLACED WITH MEDIAN
    sdom_expval = np.std(expval_per_sample)/np.sqrt(N_samples)
    
    if save_results:
        savetxt('Results/Expval/GHZ 4Q/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_expval_per_sample.csv', expval_per_sample, delimiter=',')
    
    return [mean_expval, sdom_expval]


if __name__ == '__main__':

    N_qubits = 4
    N_shadows = 50
    N_samples = 1000
    min_depth = 1
    max_depth = 5
    save_results = True
    if not save_results:
        warnings.warn("WARNING: save_results set to False")
    # Choose pauli = X1 x X2 x ... x XN
    pauli_stab = [1 for i in range(N_qubits)]
    pauli_stab += [0 for i in range(N_qubits)]
    
    # with Pool(4) as pool:
    #     results = pool.starmap(evaluate_expval, [(N_qubits, depth, pauli_stab, N_shadows, N_samples, save_results) for depth in range(min_depth, max_depth+1)]) 
    
    results = [evaluate_expval(N_qubits, depth, pauli_stab, N_shadows, N_samples, save_results=False) for depth in range(min_depth, max_depth+1)]
    
    
    print("\n", results)