# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:43:45 2024

@author: rikci
"""

import numpy as np
from math import sqrt
import itertools
import functools as ft

N = 1

H = np.array([[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]])
Id = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [Id, X, -1*X, Y, -1*Y, Z, -1*Z]
paulis_repeat = [paulis for _ in range(N)]

paulis_N = [ft.reduce(np.kron, tuplet) for tuplet in itertools.product(*paulis_repeat)]

CH = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1/sqrt(2),1/sqrt(2)],
              [0,0,1/sqrt(2),-1/sqrt(2)]],
              dtype=complex)

Upi4 = np.array([[1,0,0,0,0,0,0,0], 
              [0,1,0,0,0,0,0,0], 
              [0,0, 1/sqrt(2), 0,0, -1j/sqrt(2), 0,0],
              [0,0,0, 1/sqrt(2), -1j/sqrt(2), 0,0,0],
              [0,0,0, -1j/sqrt(2), 1/sqrt(2), 0,0,0],
              [0,0, -1j/sqrt(2), 0,0, 1/sqrt(2), 0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1]], 
              dtype=complex)

Upi2 = np.array([[1,0,0,0,0,0,0,0], 
              [0,1,0,0,0,0,0,0], 
              [0,0,0,0,0,-1j,0,0],
              [0,0,0,0,-1j,0,0,0],
              [0,0,0,-1j,0,0,0,0],
              [0,0,-1j,0,0,0,0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1]], 
              dtype=complex)

Sum = np.array([[0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0]],
               dtype = complex)

Veo4 = np.zeros((16,16), dtype=complex)
for i in range(4):
    Veo4[i,i] = 1
for i in range(12, 16):
    Veo4[i,i] = 1
Veo4[11, 4] = -1j
Veo4[8, 5] = -1j
Veo4[9, 6] = -1j
Veo4[10, 7] = -1j
Veo4[5, 8] = -1j
Veo4[6, 9] = -1j
Veo4[7, 10] = -1j
Veo4[4, 11] = -1j
    

U = np.array([[1/sqrt(2), -1j/sqrt(2)], [-1j/sqrt(2), 1/sqrt(2)]])

conjs = [np.round(np.linalg.multi_dot([U.T.conjugate(), pauli, U])) for pauli in paulis_N]

is_clifford = [True in [np.array_equal(conj, pauli) for pauli in paulis_N] for conj in conjs]

if False in is_clifford:
    print("Matrix is NOT clifford")
else:
    print("Matrix is clifford")