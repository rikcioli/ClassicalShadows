# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:07:13 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
# import matplotlib.pyplot as plt
import tqdm
from numpy import savetxt
from multiprocessing import Pool
import itertools
import os

from timeit import default_timer as timer


def fidelity(N_qubits, depth, params, N_shadows = 50, N_samples = 1000, save_results = True):
    
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    
    # CNOT state
    # sc.H(0)
    # [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    
    # RandCliff with given params
    [sc.randClifford([2*i, 2*i+1], params) for i in range(sc.N//2)]
    [sc.randClifford([2*i+1, (2*i+2)%sc.N], params) for i in range(sc.N//2)]
    [sc.randClifford([2*i, 2*i+1], params) for i in range(sc.N//2)]
    
    sc.run()
    phi = sc.state.copy()
    print(phi)
      
    print("Running program with "+str(N_qubits)+" qubits, depth "+str(depth))
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
        savetxt('Results/Fidelity/test/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fidelity_per_sample.csv', fid_per_sample, delimiter=',')
        #savetxt('Results/Fidelity/4Q 50Sh/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fid_persample_pershadow.csv', np.array(fid_per_sample_per_shadow), delimiter=',')
    
    return [fid, err]


if __name__ == '__main__':
    
    number_of_cores = 2
    
    sc = StabilizerCircuit(2)
    sc.randClifford([0,1])
    cliff_params = (675, 3, 3, False, False)
    print(cliff_params)

    qubits = range(14, 16, 2)
    depths = range(1, 3)
    pairs = list(itertools.product(qubits, depths))

    # Brownian cost
    costs = [(pair[0]**(3/2))*pair[1]//2 for pair in pairs]
    # BH cost
    # costs = [pair[0] + pair[1] for pair in pairs]
    task_by_cost = sorted(list(zip(costs, pairs)))
    task_by_cost.reverse()
    
    chunksize, extra = divmod(len(task_by_cost), 4 * number_of_cores)
    if extra:
        chunksize += 1
    
    num_groups = len(task_by_cost)//chunksize + bool(len(task_by_cost)%chunksize)
    groups = [[] for _ in range(num_groups)]
    total_execution_times = [0]*num_groups
    max_capacities = [chunksize for _ in range(num_groups-1)]
    max_capacities += [len(task_by_cost)%chunksize]

    for task in task_by_cost:
        min_index = total_execution_times.index(min(total_execution_times))
        groups[min_index].append(task)
        total_execution_times[min_index] += task[0]
        if len(groups[min_index]) >= max_capacities[min_index]:
            total_execution_times[min_index] *= 10000 
    
    task_list = []
    for group in groups:
        task_list += group
    task_list = [(task[1][0], task[1][1], cliff_params) for task in task_list]
    
    
    with Pool(number_of_cores) as pool:
        results = pool.starmap_async(fidelity, task_list) 
        for value in results.get():
            print(value)
    
    # results = [fidelity(14, depth, cliff_params, N_samples=1000, save_results=False) for depth in range(1,2)]
        
    # results = fidelity(6, depth=4, N_shadows=50, N_samples=100, save_results=False)
        
    # print("\n", results)
    
    
