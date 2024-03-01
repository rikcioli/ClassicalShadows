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
import math
import matplotlib.pyplot as plt


def compare_expval(N_qubits, depth, pauli_stab, N_shadows = 50, N_samples = 1000, save_results = True):
    
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    sc.run()
        
    shadows = sc.saveShadowsBH(N_shadows*N_samples, depth)      
    random_circuits = shadows[0]
    outcomes = shadows[1]
        
    expval_per_sample = np.empty(N_samples)
    
        
    # Evaluate exact eigenvalues
    N = N_qubits
    p = 1
    q = 2
    
    where_pauli = set(np.where(pauli_stab)[0]%N)
    initial_condition = np.zeros(N)
    initial_condition[len(where_pauli)-1] = 1
    
    if depth > 0:
        weights_matrix = np.zeros((N, N))
        for i in range(N):
            w = i+1
            for j in range(N):
                if j == i:
                    weights_matrix[i, i] = 1 - (2*p/(N*(N-1))) * (w*(N-w)*(q**2-1)**2/(q**4-1) + (2*(q**2 -1)/(q**4 -1))*math.comb(w, 2))
                elif j == i+1:
                    weights_matrix[i, j] = (2*p/(N*(N-1))) * (2*(q**2-1)/(q**4-1))*math.comb(w+1, 2)
                elif j == i-1:
                    weights_matrix[i, j] = (2*p/(N*(N-1))) * ((w-1)*(N-w+1)*(q**2-1)**2/(q**4-1))
        
        if depth > 1:     
            weights_matrix = np.linalg.multi_dot([weights_matrix for i in range(depth)])
        
        final_weights = np.dot(weights_matrix, initial_condition)
    else:
        final_weights = initial_condition
        
    eigenval = sum((final_weights[w]*(q+1)**(-w-1) for w in range(N)))
    expval_ps_exact = np.empty(N_samples)

    for sample_number in tqdm.tqdm(range(N_samples)):
            
        # Extract current sample's shadows
        circuits_current_sample = random_circuits[N_shadows*sample_number : N_shadows*(sample_number+1)]
        outcomes_current_sample = outcomes[N_shadows*sample_number : N_shadows*(sample_number+1)]
        
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
        
        # Evaluate expval with exact eigenvalues
        expval_ps_exact[sample_number] = (sum(expvals)/N_shadows)/eigenval
        
        
    mean_expval = np.mean(expval_per_sample)     # CAN ALSO BE REPLACED WITH MEDIAN
    sdom_expval = np.std(expval_per_sample)/np.sqrt(N_samples)
    
    mean_exact = np.mean(expval_ps_exact)
    sdom_exact = np.std(expval_ps_exact)/np.sqrt(N_samples)
    
    if save_results:
        savetxt('Results/Compare_expval/GHZ 4Q/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_expval_per_sample.csv', expval_per_sample, delimiter=',')
    
    return [mean_expval, sdom_expval, mean_exact, sdom_exact]



if __name__ == '__main__':

    N_qubits = 4
    N_shadows = 50
    N_samples = 1000
    min_depth = 11
    max_depth = 20
    save_results = True
    if not save_results:
        warnings.warn("WARNING: save_results set to False")
    # Choose pauli = X1 x X2 x ... x XN
    pauli_stab = [1 for i in range(N_qubits)]
    pauli_stab += [0 for i in range(N_qubits)]
    # pauli_stab = [1, 0, 0, 0, 0, 0, 0, 0]
    
    
    with Pool() as pool:
        results = pool.starmap(compare_expval, [(N_qubits, depth, pauli_stab, N_shadows, N_samples, save_results) for depth in range(min_depth, max_depth+1)]) 
    
    # results = [compare_expval(N_qubits, depth, pauli_stab, N_shadows, N_samples, save_results=False) for depth in range(min_depth, max_depth+1)]
    
    res_array = np.array(results)
    depths = [i for i in range(min_depth, max_depth+1)]
    
    plt.figure(dpi=600)
    plt.errorbar(depths, res_array[:, 0], yerr = res_array[:, 1], fmt='bo')
    plt.errorbar(depths, res_array[:, 2], yerr = res_array[:, 3], fmt='ro')
    plt.xlabel(r"$Depth$")
    plt.ylabel("Expval")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()
    
    print("\n", results)