# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from circuit import StabilizerCircuit

# insert EVEN number of qubits    
N_qubits = 4

N_shadows = 100
depth = 4
N_shots = 1

sc = StabilizerCircuit(N_qubits)
sc.H(0)
sc.CNOT(0, 1)
sc.CNOT(1, 2)
sc.CNOT(2, 3)
state_prep_len = len(sc.circuit)
temp_circ = sc.circuit.copy()

random_circuits = []
outcomes = []


for i in range(N_shadows):
    sc._circuit = temp_circ.copy()
    sc.randEvolution(depth)
    random_cliff = [gate for gate in sc.circuit[state_prep_len:]]
    sc.measure([i for i in range(N_qubits)])
    counts = sc.run(N_shots)
    
    random_circuits.append(random_cliff)
    outcomes.append(list(counts.keys())[0])
    
    


print(sc.stabilizer_table)
print("Valid state:", sc._is_valid_state())
print("Scalar prod with |0>:", sc.dot_zero())



print("Total counts are:", counts)
sc.plot_histogram(counts)