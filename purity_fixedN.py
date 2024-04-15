# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:01:28 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
import copy
# from collections import Counter
from numpy import savetxt
import itertools
from multiprocessing import Pool
import time
import os


def purity(N_qubits, depth, params, N_shadows = 50, N_samples = 1000, save_results = True):
    
    start_time = time.time()
    print("Running program with "+str(N_qubits)+" qubits, depth "+str(depth)+", at time "+str(start_time), flush=True)

    sc = StabilizerCircuit(N_qubits)

    # # Prepare arbitrary initial state
    #[sc.randClifford([2*i, 2*i+1], params) for i in range(sc.N//2)]
    #[sc.randClifford([2*i+1, (2*i+2)%sc.N], params) for i in range(sc.N//2)]
    #[sc.randClifford([2*i, 2*i+1], params) for i in range(sc.N//2)]

    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    
    sc.run()
    
    shadows = sc.saveShadows(N_shadows*N_samples, depth)      
    random_circuits = shadows[0]
    outcomes = shadows[1]
        
    purity_per_sample = np.empty(N_samples)
        
    for sample_number in range(N_samples):
            
        circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
        outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
        
        inverse_circuits = copy.deepcopy(circuits_current_sample)
        [circuit.reverse() for circuit in inverse_circuits]
            
        # Compute global purity
        Udagb_states = [StabilizerCircuit(N_qubits, circuit = inverse_circuits[i]) for i in range(N_shadows)]
        for i in range(N_shadows):
            Udagb_states[i].state[N_qubits:2*N_qubits, 2*N_qubits] = np.array(outcomes_current_sample[i])
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
                [state_s.X(i) for i in range(N_qubits) if outcomes_current_sample[r][i]]
                state_s.run()
                main_trace += abs(state_s.dot_zero())**2
                state_s.state = current_state
                    
        purity = main_trace*2*((2**N_qubits+1)**2)/(N_shadows*(N_shadows-1)) - (2**N_qubits+2)
        purity_per_sample[sample_number] = purity
        
    avg_purity = np.mean(purity_per_sample)     # CAN ALSO BE REPLACED BY MEDIAN
    err = np.sqrt(np.var(purity_per_sample)/N_samples)
        
    if save_results:
        savetxt('Results/Purity/GHZ All Qubits/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'-'+str(N_samples)+'_purity_per_sample.csv', purity_per_sample, delimiter=',')
    
    return [avg_purity, err]
    

if __name__ == '__main__':

    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    # number_of_cores = 4
    print("# CPUs:", number_of_cores)

    # cliff_params = (675, 3, 3, False, False)
    cliff_params = "GHZ"
    print(cliff_params)

    qubits = range(32, 42, 2)
    depths = range(1, 21)
    pairs = list(itertools.product(qubits, depths))

    # Brownian cost
    costs = [(pair[0]**(3/2))*pair[1]//2 for pair in pairs]
    # BH cost
    # costs = [pair[0] + pair[1] for pair in pairs]
    task_by_cost = sorted(list(zip(costs, pairs)))
    task_by_cost.reverse()
    
    chunksize, extra = divmod(len(task_by_cost), number_of_cores)
    if extra:
        chunksize += 1

    num_groups = len(task_by_cost)//chunksize + bool(len(task_by_cost)%chunksize)
    print("Chunksize:", chunksize, "\nNumber of groups:", num_groups)
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
    print(task_list)


    with Pool(number_of_cores) as pool:
        results = pool.starmap_async(purity, task_list) 
        for value in results.get():
            print(value)    
