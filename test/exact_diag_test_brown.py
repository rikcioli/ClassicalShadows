# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:45:00 2024

@author: rikci
"""

import numpy as np
from scipy.linalg import expm
from numpy.linalg import matrix_power
import functools as ft
from math import factorial as fact
import math
import itertools as it
from sympy.utilities.iterables import multiset_permutations

N = 2
d = 2
p = 1
depth = 1

Ip = np.array([d, 0])
Im = np.array([1, np.sqrt(d**2-1)])

def kron(A, B):
    
    dimA = len(A)
    dimB = len(B)
    
    AxB = np.empty([dimA*dimB, dimA*dimB])
    for i in range(dimA*dimB):
        for j in range(dimA*dimB):
            AxB[i,j] = A[i//dimB, j//dimB]*B[i%dimB, j%dimB]
    return AxB

def Ujk(j, k):
    if j == k:
        raise ValueError("j must be different from k")

    IpIp_list = [np.eye(2) for _ in range(N)]
    IpIp_list[j] = np.outer(Ip, Ip)
    IpIp_list[k] = np.outer(Ip, Ip)
    IpIp = ft.reduce(np.kron, IpIp_list)
    
    ImIm_list = [np.eye(2) for _ in range(N)]
    ImIm_list[j] = np.outer(Im, Im)
    ImIm_list[k] = np.outer(Im, Im)
    ImIm = ft.reduce(np.kron, ImIm_list)
    
    IpIm_list = [np.eye(2) for _ in range(N)]
    IpIm_list[j] = np.outer(Ip, Im)
    IpIm_list[k] = np.outer(Ip, Im)
    IpIm = ft.reduce(np.kron, IpIm_list)
    
    ImIp_list = [np.eye(2) for _ in range(N)]
    ImIp_list[j] = np.outer(Im, Ip)
    ImIp_list[k] = np.outer(Im, Ip)
    ImIp = ft.reduce(np.kron, ImIp_list)
    
    Ujk = (IpIp + ImIm - (IpIm + ImIp)/d**2)/(d**4-1)
    
    return Ujk

def Wn(n):
    Wn_list = [d*Im - Ip for _ in range(n)] + [Ip for _ in range(N-n)]
    Wn = ft.reduce(np.kron, Wn_list)
    Wn = Wn/(fact(n)*fact(N-n))
    return Wn

X = np.array([[0, 1],
       [1, 0]])
Id = np.eye(2)
Z = np.array([[1,0], [0,-1]])

sites = range(N)
permutations = list(it.permutations(sites))

L = sum((Ujk(perm[0], perm[1]) for perm in permutations))/fact(N)

# L = sum((np.linalg.multi_dot([Ujk(perm[i], perm[i+1]) for i in range(N//2)]) for perm in permutations))/fact(N)
Mt = lambda t: matrix_power(L, t)

Wnt = lambda n, t: np.dot(Wn(n), Mt(t))

sigma_x_sigma = (d*Im - Ip)/(d**2-1)
# insert sigma as if it were a stabilizer, but with an OR between X and Z stabs
# s.t. location of 1 indicates there's a pauli there (doesn't matter which one)
pauli_stab = [1, 1]
# construct all permutations of pauli stab in a smart way
perm_pauli_stab = list(multiset_permutations(pauli_stab))
perm_pauli_op = []
for given_perm in perm_pauli_stab:
    perm_pauli_op.append([sigma_x_sigma if value else Ip for value in given_perm])

Oa_x_Oa_perm_inv = sum(ft.reduce(np.kron, sigma_list) for sigma_list in perm_pauli_op)
Oa_x_Oa_t = lambda t: np.dot(Mt(t), Oa_x_Oa_perm_inv)

# mult be multiplied by k!(N-k)! where k is the number of 1s in pauli_stab
k = len(np.where(pauli_stab))
pi_nt = lambda n, t: (fact(N)/len(perm_pauli_stab))*np.dot(Wn(n), Oa_x_Oa_t(t))*d**(-2*N)

pi_vec = [pi_nt(n, depth) for n in range(0, N+1)]
print("ED pi vec =", pi_vec)


# Compare with differential equation
where_pauli = set(np.where(pauli_stab)[0]%N)
initial_condition = np.zeros(N+1)
initial_condition[len(where_pauli)] = 1

q = d
if depth > 0:
    weight_mat2 = np.zeros((N+1, N+1))
    for i in range(N+1):
        w = i
        for j in range(N+1):
            if i == j:
                weight_mat2[i, j] = 1 - (2*p/(N*(N-1)))*((q**2-1)/(q**4-1))*((q**2-1)*w*(N-w) + w*(w-1))
            elif j == i-1:
                weight_mat2[i, j] = (2*p/(N*(N-1)))*((q**2-1)**2/(q**4-1))*(w-1)*(N-w+1)
            elif j == i+1:
                weight_mat2[i, j] = (2*p/(N*(N-1)))*((q**2-1)/(q**4-1))*(w+1)*w
    
    if depth > 1:     
        weight_mat2 = np.linalg.multi_dot([weight_mat2 for i in range(depth)])  
    final_weights = np.dot(weight_mat2, initial_condition)
else:
    final_weights = initial_condition

print("Diff eq pi vec =", final_weights)
