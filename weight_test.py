# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:13:14 2024

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


def eval_pi(N_qubits, depth, pauli_stab, N_shadows = 50, N_samples = 1000, save_results = True):
    
    N = N_qubits
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    sc.run()
        
    # Extract shadows with given depth
    shadows = sc.saveShadows(N_shadows*N_samples, depth)      
    random_circuits = shadows[0]
    outcomes = shadows[1]
        
    # Prepare to store expval per sample
    pi_A_per_sample = np.empty((N_samples, N_qubits+1))

    for sample_number in tqdm.tqdm(range(N_samples)):
        
        # Extract current sample's shadows
        circuits_current_sample = random_circuits[N_shadows*sample_number : N_shadows*(sample_number+1)]
        outcomes_current_sample = outcomes[N_shadows*sample_number : N_shadows*(sample_number+1)]
        
        # Create Nsh circuits to evolve pauli stab according to U
        scs = [StabilizerCircuit(N_qubits, circuit = U) for U in circuits_current_sample]
        pauli_t = []
        for circ in scs:
            circ.state[N_qubits, 0:2*N_qubits] = pauli_stab
            circ.run()
            pauli_t.append(circ.state[N_qubits, 0:2*N_qubits])
            
        pauli_t_weights = [len(set(np.where(pauli)[0]%N)) for pauli in pauli_t]
        
        pi_A = np.zeros(N_qubits+1)
        for w in range(N+1):
            pi_w_r = [1 if weight == w else 0 for weight in pauli_t_weights]
            pi_w = sum(pi_w_r)/N_shadows
            pi_A[w] = pi_w
        
        pi_A_per_sample[sample_number] = pi_A

    mean_pi_A = np.mean(pi_A_per_sample, 0)
    std_pi_A = np.std(pi_A_per_sample, 0)/np.sqrt(N_samples)
    
    # Evaluate exact weights
    
    p = 1
    q = 2 
    where_pauli = set(np.where(pauli_stab)[0]%N)
    initial_condition = np.zeros(N+1)
    initial_condition[len(where_pauli)] = 1
    
    if depth > 0:
        brown_mat = np.empty((N+1, N+1))
        c_nmr = np.empty((N+1, N+1, N//2+1))
        d_nm = np.empty((N+1, N+1))

        for n in range(N+1):
            for m in range(N+1):
                for r in range(n//2+1):  
                    c_nmr[n, m, r] = sum((math.comb(n-2*r, j)*math.comb(r, m-n+r-j)*2**(j-m+2*(n-r)) if m-n+r-j>=0 else 0 for j in range(n-2*r+1)))
                d_nm[n, m] = ((math.factorial(m)*math.factorial(N-m))/(math.factorial(n)*math.factorial(N-n)))*sum((c_nmr[n, m, r]*math.comb(N//2, r)*math.comb(N//2-r, N//2-n+r)*(q**4-1)**(r-n) if N//2-n+r>=0 else 0 for r in range(n//2+1)))
                brown_mat[n, m] = d_nm[n, m]*((q**2-1)**n)/math.comb(N, n)
        
        if depth > 1:     
            brown_mat = np.linalg.multi_dot([brown_mat for i in range(depth)])  
        final_weights = np.dot(brown_mat, initial_condition)
    else:
        final_weights = initial_condition
    
    return [mean_pi_A, std_pi_A, final_weights]


if __name__ == '__main__':

    N_qubits = 4
    N_shadows = 50
    N_samples = 100
    min_depth = 1
    max_depth = 1
    save_results = False
    if not save_results:
        warnings.warn("WARNING: save_results set to False")
    # Choose pauli = X1 x X2 x ... x XN
    pauli_stab = [1 for i in range(N_qubits)]
    pauli_stab += [0 for i in range(N_qubits)]
    # pauli_stab = [1, 0, 0, 0, 0, 0, 0, 0]
    pauli_stab = np.array(pauli_stab)
    
    # with Pool() as pool:
        # results = pool.starmap(compare_expval, [(N_qubits, depth, pauli_stab, N_shadows, N_samples, save_results) for depth in range(min_depth, max_depth+1)]) 
    
    results = [eval_pi(N_qubits, depth, pauli_stab, N_shadows, N_samples, save_results=False) for depth in range(min_depth, max_depth+1)]
    
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