# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:01:28 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
import copy
import matplotlib.pyplot as plt
import tqdm
from collections import Counter
from numpy import savetxt
import itertools
from multiprocessing import Pool



def purity(N_qubits, depth, N_shadows = 50, N_samples = 1000, save_results = True):
    
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    sc.run()

      
    print("Running program with "+str(N_qubits)+" qubits, depth "+str(depth), flush=True)
    print("N_shadows = "+str(N_shadows)+", N_samples =", N_samples)
        
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

    qubits = range(4, 18, 2)
    depths = range(1, 19)
    pairs = list(itertools.product(qubits, depths))
    # Brownian cost
    costs = [(pair[0]**(3/2))*pair[1]//2 for pair in pairs]
    # BH cost
    # costs = [pair[0] + pair[1] for pair in pairs]
    pairs_by_cost = sorted(list(zip(costs, pairs)))
    
    num_groups = 16
    groups = [[] for _ in range(num_groups)]
    total_execution_times = [0]*num_groups

    for task in pairs_by_cost:
        min_index = total_execution_times.index(min(total_execution_times))
        groups[min_index].append(task)
        total_execution_times[min_index] += task[0]
    
    task_list = []
    for group in groups:
        task_list += group
    task_list = [task[1] for task in task_list]

    # save_results = True
    # if not save_results:
    #     print("WARNING: save_results set to False", flush=True)
        
    with Pool(4) as pool:
        results = pool.starmap(purity, task_list) 
        
    # results = [fidelity(N_qubits, depth, N_samples=1000, save_results=False) for depth in range(min_depth, max_depth+1)]
        
    # results = fidelity(14, depth=18, N_shadows=50, N_samples=100, save_results=False)
        
    print("\n", results)
