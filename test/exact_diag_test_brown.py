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

N = 4
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

def sum_Wn(n):
    region = [1 for _ in range(n)] + [0 for _ in range(N-n)]
    diff_perm = list(set(it.permutations(region)))
    Wn_tosum = []
    for perm in diff_perm:
        Wn_list = [d*Im - Ip if value else Ip for value in perm]
        Wn = ft.reduce(np.kron, Wn_list)
        Wn_tosum.append(Wn)

    return sum(Wn_tosum)

def multinomial(lst):
    # Given a list of elements a1, a2, ..., an computes (a1+...+an)!/(a1!a2!...an!)
    if any(np.array(lst) < 0):
        return 0
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res


sites = range(N)
permutations = list(it.permutations(sites))

if N == 2:
    L = sum((Ujk(perm[0], perm[1]) for perm in permutations))/fact(N)
else:
    L = sum((np.linalg.multi_dot([Ujk(perm[2*i], perm[2*i+1]) for i in range(N//2)]) for perm in permutations))/fact(N)

Mt = lambda t: matrix_power(L, t)

# Wnt = lambda n, t: np.dot(sum_Wn(n), Mt(t))

sigma_x_sigma = (d*Im - Ip)/(d**2-1)
# insert sigma as if it were a stabilizer, but with an OR between X and Z stabs
# s.t. location of 1 indicates there's a pauli there (doesn't matter which one)
pauli_stab = [0, 1, 1, 1]

Oa_x_Oa_list = [sigma_x_sigma if value else Ip for value in pauli_stab]
Oa_x_Oa = ft.reduce(np.kron, Oa_x_Oa_list)
Oa_x_Oa_t = lambda t: np.dot(Mt(t), Oa_x_Oa)

pi_nt = lambda n, t: np.dot(sum_Wn(n), Oa_x_Oa_t(t))*d**(-2*N)

pi_vec = [pi_nt(n, depth) for n in range(0, N+1)]
print("ED pi vec =", pi_vec)


# Compare with differential equation
where_pauli = set(np.where(pauli_stab)[0]%N)
initial_condition = np.zeros(N+1)
initial_condition[len(where_pauli)] = 1

q = d
if depth > 0:
    brown_mat = np.empty((N+1, N+1))
    alpha_nm = np.empty((N+1, N+1))
    
    for n in range(N+1):
        for m in range(N+1): 
            alpha_nm[n, m] = (d**(n-m)/math.comb(N, m))*sum((d**(2*n-4*r)*(d**4-1)**(r-n)*multinomial([r, n-2*r, N//2-n+r])*math.comb(n-r, m-n+r) if m-n+r>=0 else 0 for r in range(n//2+1)))
            brown_mat[n, m] = alpha_nm[n, m]*(d**2-1)**n

    if depth > 1:     
        brown_mat = np.linalg.multi_dot([brown_mat for i in range(depth)])  
    final_weights = np.dot(brown_mat, initial_condition)
else:
    final_weights = initial_condition

print("Diff eq pi vec =", final_weights)