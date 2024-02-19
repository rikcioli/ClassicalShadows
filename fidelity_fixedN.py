# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:07:13 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
import matplotlib.pyplot as plt
import tqdm
from numpy import savetxt
from multiprocessing import Pool
import multiprocessing, logging

from timeit import default_timer as timer


def fidelity(N_qubits, depth, N_shadows = 50, N_samples = 10000, save_results = True):
    
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    sc.run()
    phi = sc.state.copy()
      
    print("Running program with "+str(N_qubits)+" qubits, depth "+str(depth), flush=True)
    print("N_shadows = "+str(N_shadows)+", N_samples =", N_samples)
        
    shadows = sc.saveShadows(N_shadows*N_samples, depth)      
    random_circuits = shadows[0]
    outcomes = shadows[1]
        
    fid_per_sample = np.empty(N_samples)
        
    for sample_number in tqdm.tqdm(range(N_samples)):
            
        circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
        outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
        Uphi_states = [StabilizerCircuit(N_qubits, phi, circuits_current_sample[r]) for r in range(N_shadows)]
        [[Uphi_states[r].X(i) for i in range(N_qubits) if outcomes_current_sample[r][i]] for r in range(N_shadows)]
        [state.run() for state in Uphi_states]
        
        fid_per_sample[sample_number] = (2**N_qubits+1)*sum((state.dot_zero())**2 for state in Uphi_states)/N_shadows - 1
        
    fid = np.mean(fid_per_sample)     # CAN ALSO BE REPLACED BY MEDIAN
    err = np.sqrt(np.var(fid_per_sample)/N_samples)
    
    if save_results:
        savetxt('Results/Fidelity/GHZ 4Q 10000S/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fidelity_per_sample.csv', fid_per_sample, delimiter=',')
        #savetxt('Results/Fidelity/4Q 50Sh/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fid_persample_pershadow.csv', np.array(fid_per_sample_per_shadow), delimiter=',')
    
    return [fid, err]


if __name__ == '__main__':

    N_qubits = 4
    min_depth = 4
    max_depth = 20
    
    # save_results = True
    # if not save_results:
    #     print("WARNING: save_results set to False", flush=True)
    
    with Pool() as pool:
        results = pool.starmap(fidelity, [(N_qubits, depth) for depth in range(min_depth, max_depth+1)]) 
    
    # results = [fidelity(N_qubits, depth, N_samples=1000, save_results=False) for depth in range(min_depth, max_depth+1)]
    
    # results = fidelity(N_qubits, depth=16, N_shadows=50, N_samples=3000, save_results=False)
    
    print("\n", results)
    
    
