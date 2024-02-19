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


# # Prepare quadruple copy of the state
# replica = StabilizerCircuit(4*N_qubits)
# for i in range(4):
#     replica.H(2*i)
#     replica.CNOT(2*i, 2*i+1)

# replica.run()
# print(replica.state)
# replica.reset_circuit()

# # project onto Bell pairs
# replica.CNOT(2, 4)
# replica.CNOT(1, 3)
# replica.CNOT(5, 7)
# replica.CNOT(0, 6)
# replica.H(0)
# replica.H(1)
# replica.H(2)
# replica.H(5)


# replica.run()
# print(replica.state)
# renyi2 = 4*replica.dot_zero()

# print(renyi2)

N_shadows = 50
N_samples = 100
depth = 1

save_results = True
if not save_results:
    warnings.warn("WARNING: save_results set to False")


# Extract classical shadows     
shadows = sc.saveShadowsGlobal(N_shadows*N_samples)      
random_circuits = shadows[0]
outcomes = shadows[1]

renyi_per_sample = np.empty(N_samples)

# Prepare quadruple copy of the state
replica = StabilizerCircuit(4*N_qubits)
replica.H(0)
replica.H(1)
replica.H(2)
replica.H(5)
replica.CNOT(2, 4)
replica.CNOT(1, 3)
replica.CNOT(5, 7)
replica.CNOT(0, 6)
replica.run()

bell_start = replica.state

        
for sample_number in tqdm.tqdm(range(N_samples), desc="Evaluating "+str(N_shadows)+" shadows with "+str(N_samples)+ " samples..."):
    
    circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
    outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
    
    # create conjugate circuits on qubits 2 and 3
    circuits_conj = copy.deepcopy(circuits_current_sample)
    for circ in circuits_conj:
        for gate in circ:
            gate.qubits = [i+2 for i in gate.qubits]
            gate.conj()

    
    replica_states = [StabilizerCircuit(4*N_qubits, state = bell_start, circuit = circuits_current_sample[s]+circuits_conj[s]) for s in range(N_shadows)]
    [replica_states[s].X(i) for s in range(N_shadows) for i in range(N_qubits) if outcomes_current_sample[s][i]]
    [replica_states[s].X(i+2) for s in range(N_shadows) for i in range(N_qubits) if outcomes_current_sample[s][i]]
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
                gate.qubits = [i+4 for i in gate.qubits]
            [state_r.X(i+4) for i in range(N_qubits) if outcomes_current_sample[s+r+1][i]]
            [state_r.X(i+6) for i in range(N_qubits) if outcomes_current_sample[s+r+1][i]]
            
        [final_state.run() for final_state in V_states]
        renyi += sum((4*final_state.dot_zero() for final_state in V_states))
        
    renyi = renyi*2*((2**N_qubits+1)**2)/(N_shadows*(N_shadows-1)) - (2**(N_qubits//2 + 1))*(2**N_qubits+1) + 2**(3*N_qubits//2)
    renyi_per_sample[sample_number] = renyi
        
    
renyi_per_sample = np.array(renyi_per_sample)
if save_results:
    savetxt('Results/Entropy/'+str(N_shadows)+'-'+str(N_samples)+'_entropy_per_sample.csv', renyi_per_sample, delimiter=',')

avg_renyi = np.mean(renyi_per_sample)
sdom_renyi = np.std(renyi_per_sample)/np.sqrt(N_samples)

print(avg_renyi, sdom_renyi)