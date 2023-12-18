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

N_shots = 1000

sc = StabilizerCircuit(N_qubits)
sc.H(0)
sc.CNOT(0, 1)
sc.randEvolution(20)
sc.measure([i for i in range(N_qubits)])

counts = sc.run(N_shots)





print(sc.stabilizer_table)
print("Valid state:", sc._is_valid_state())
print("Scalar prod with |0>:", sc.dot_zero())



print("Total counts are:", counts)
sc.plot_histogram(counts)