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
import warnings
import itertools
from multiprocessing import Pool


# hN = N_qubits//2

# # Prepare quadruple copy of the state
# replica = StabilizerCircuit(4*N_qubits)
# for i in range(4):
#     replica.H(2*hN*i)
#     [replica.CNOT(l, l+1) for l in range(2*hN*i, 2*hN*(i+1) - 1)]

# replica.run()
# print(replica.state)
# replica.reset_circuit()



# # project onto Bell pairs
# [replica.CNOT(i, i+2*hN) for i in range(1*hN, 2*hN)]
# [replica.CNOT(i, i+2*hN) for i in range(2*hN, 3*hN)]
# [replica.CNOT(i, i+2*hN) for i in range(5*hN, 6*hN)]
# [replica.CNOT(i, i+6*hN) for i in range(1*hN)]
# [replica.H(i) for i in range(3*hN)]
# [replica.H(i) for i in range(5*hN, 6*hN)]

# replica.run()
# print(replica.state)
# renyi2 = (2**N_qubits)*replica.dot_zero()

# print(renyi2)

def renyi2(N_qubits, depth, N_shadows = 50, N_samples = 1000, save_results = True):
    
    hN = N_qubits//2
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    sc.H(0)
    [sc.CNOT(i, i+1) for i in range(N_qubits-1)]
    sc.run()
    
    # Prepare bell pairs 
    replica = StabilizerCircuit(4*N_qubits)
    [replica.H(i) for i in range(3*hN)]
    [replica.H(i) for i in range(5*hN, 6*hN)]
    [replica.CNOT(i, i+2*hN) for i in range(1*hN, 2*hN)]
    [replica.CNOT(i, i+2*hN) for i in range(2*hN, 3*hN)]
    [replica.CNOT(i, i+2*hN) for i in range(5*hN, 6*hN)]
    [replica.CNOT(i, i+6*hN) for i in range(1*hN)]

    replica.run()
    bell_start = replica.state
    
    # Extract classical shadows     
    shadows = sc.saveShadows(N_shadows*N_samples, depth)      
    random_circuits = shadows[0]
    outcomes = shadows[1]
    
    renyi_per_sample = np.empty(N_samples)  
    
    # Evaluate subsystem purity for each sample
    for sample_number in tqdm.tqdm(range(N_samples), desc="Evaluating "+str(N_shadows)+" shadows with "+str(N_samples)+ " samples..."):
        
        circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
        outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
        
        # create conjugate circuits on qubits 2 and 3
        circuits_conj = copy.deepcopy(circuits_current_sample)
        for circ in circuits_conj:
            for gate in circ:
                gate.qubits = [i+2*hN for i in gate.qubits]
                gate.conj()
    
        
        replica_states = [StabilizerCircuit(4*N_qubits, state = bell_start, circuit = circuits_current_sample[s]+circuits_conj[s]) for s in range(N_shadows)]
        [replica_states[s].X(i) for s in range(N_shadows) for i in range(N_qubits) if outcomes_current_sample[s][i]]
        [replica_states[s].X(i+2*hN) for s in range(N_shadows) for i in range(N_qubits) if outcomes_current_sample[s][i]]
        [replica.run() for replica in replica_states]
        
        # U_Udagb_states = [StabilizerCircuit(N_qubits, Udagb_states[s].state, random_circuits[r]) for s, r in itertools.combinations(range(N_shadows), 2)]
        # outcomes_combinations = (outcomes[r] for s, r in itertools.combinations(range(N_shadows), 2))
        # [state.run() for state in U_Udagb_states]
        # main_trace = sum((state.dot_outcome(outcome)**2 for state, outcome in zip(U_Udagb_states, outcomes_combinations)))
                
        renyi = 0
        for s, state_s in enumerate(replica_states):
            V_states = [StabilizerCircuit(4*N_qubits, state = state_s.state, circuit = circuits_current_sample[r]+circuits_conj[r]) for r in range(s+1, N_shadows)]
            for r, state_r in enumerate(V_states):
                for gate in state_r.circuit:
                    gate.qubits = [i+4*hN for i in gate.qubits]
                [state_r.X(i+4*hN) for i in range(N_qubits) if outcomes_current_sample[s+r+1][i]]
                [state_r.X(i+6*hN) for i in range(N_qubits) if outcomes_current_sample[s+r+1][i]]
                
            [final_state.run() for final_state in V_states]
            renyi += sum(((2**N_qubits)*final_state.dot_zero() for final_state in V_states))
            
        renyi = renyi*2*((2**N_qubits+1)**2)/(N_shadows*(N_shadows-1)) - (2**(hN + 1))*(2**N_qubits+1) + 2**(N_qubits+hN)
        renyi_per_sample[sample_number] = renyi
            
        
    renyi_per_sample = np.array(renyi_per_sample)
    if save_results:
        savetxt('Results/Entropy/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_entropy_per_sample.csv', renyi_per_sample, delimiter=',')
    
    avg_renyi = np.mean(renyi_per_sample)
    sdom_renyi = np.std(renyi_per_sample)/np.sqrt(N_samples)
    
    return [avg_renyi, sdom_renyi]


if __name__ == '__main__':
    
    N_qubits = 6
    N_shadows = 50
    N_samples = 1000
    min_depth = 7
    max_depth = 14
    save_results = True
    if not save_results:
        warnings.warn("WARNING: save_results set to False")
    
    with Pool(4) as pool:
        results = pool.starmap(renyi2, [(N_qubits, depth, N_shadows, N_samples, save_results) for depth in range(min_depth, max_depth+1)]) 
    
    print("\n", results)