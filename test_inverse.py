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

N_qubits = 2

sc = StabilizerCircuit(N_qubits)


sc.H(0)
sc.CNOT(0, 1)
sc.run()
print(sc.state)
sc.circuit = []

sc.randClifford([1,0])
sc.randClifford([0,1])
sc.run()
print(sc.state)

save_circuit = copy.deepcopy(sc.circuit)
[gate.dagger() for gate in sc.circuit]
sc.circuit.reverse()
#sc.state = saved_state
sc.run()
print(sc.state)