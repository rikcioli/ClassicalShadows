# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:33:22 2024

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

N_qubits = 4

sc = StabilizerCircuit(N_qubits)

sc.H(0)
[sc.CNOT(i, i+1) for i in range(N_qubits-1)]
sc.run()
print(sc.state)
sc.circuit = []

# sc.randClifford([0,1])
# sc.randClifford([2,3])
# sc.randClifford([0,2])
# sc.randClifford([1,3])
# sc.circuit[0].params = [381, 2, 0, False, True]
# sc.circuit[1].params = [585, 3, 2, False, False]
# sc.circuit[2].params = [306, 2, 2, False, False]
# sc.circuit[3].params = [45, 3, 1, False, False]
sc.randClifford([0,1,2,3])
[gate.conj() for gate in sc.circuit]
sc.run()
print(sc.state)

save_circuit = copy.deepcopy(sc.circuit)
[gate.dagger() for gate in sc.circuit]
sc.circuit.reverse()
#sc.state = saved_state
sc.run()
print(sc.state)