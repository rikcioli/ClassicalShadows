# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:04:35 2024

@author: rikci
"""

from circuit import StabilizerCircuit
from instruction import RandomClifford
import numpy as np
import tqdm

n = 2

s = 2**(2*n)-1
prod = 1
for j in range(1, n+1):
    prod *= 4**j-1
dim_symplectic = 2**(n*n)*prod
dim_clifford = dim_symplectic*4**n

cliffords = [RandomClifford([0,1], params = (i//16, (i%16)//4, (i%16)%4, False, False)) for i in range(dim_clifford)]

states = [StabilizerCircuit(N_qubits = n, circuit = [cliffords[i]]) for i in range(dim_clifford)]
[state.run() for state in states]

Sn = []
for i in tqdm.tqdm(range(dim_clifford)):
    state = states[i].state
    if True in [np.array_equal(state, state_prev) for state_prev in Sn]:
        print("Trovati due uguali a i =", i)
        break
    else:
        Sn.append(state)
        
        