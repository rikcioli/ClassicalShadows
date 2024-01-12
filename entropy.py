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

N_qubits = 2

sc = StabilizerCircuit(N_qubits)

saved_state = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 1, 0, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1, 1, 1, 0, 1],
                        [0, 0, 1, 1, 1, 0, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 1, 1, 1, 1]])

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

# Extract classical shadows
N_shadows = 100
depth = 1

random_circuits, outcomes = sc.saveShadows(N_shadows, depth)

inverse_circuits = copy.deepcopy(random_circuits)
[circuit.reverse() for circuit in inverse_circuits]

# Compute global purity
Udagb_states = [StabilizerCircuit(N_qubits, circuit = inverse_circuits[i]) for i in range(N_shadows)]
for i in range(N_shadows):
    Udagb_states[i].state[N_qubits:2*N_qubits, 2*N_qubits] = np.array(outcomes[i])
    [gate.dagger() for gate in Udagb_states[i].circuit]

[state.run() for state in Udagb_states]
    
all_circuits = []
for s in range(N_shadows):
    for r in range(s+1, N_shadows):
        new_state = StabilizerCircuit(N_qubits, Udagb_states[s].state, random_circuits[r])
        [new_state.X(i) for i in range(N_qubits) if outcomes[r][i]]
        all_circuits.append(new_state)

[circ.run() for circ in all_circuits]
scalar_prods_sq = [circuit.dot_zero()**2 for circuit in all_circuits]
main_trace = sum(scalar_prods_sq)

main_trace = main_trace*2*((2**N_qubits+1)**2)/(N_shadows*(N_shadows-1)) - (2**N_qubits+2)
print(main_trace)