# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:15:26 2023

@author: rikci
"""

import numpy as np
from numba import njit

@njit
def inner(v, w):
    N = len(v)
    vw = [v[i] & w[i+(-1)**i] for i in range(N)]
    return sum(vw)%2

@njit
def direct_sum(A, B):
    dim = len(A) + len(B)
    dirsum = np.zeros((dim, dim), dtype = np.int64)
    dirsum[0:len(A), 0:len(A)] = A
    dirsum[len(A):dim, len(A):dim] = B
    return dirsum

@njit
def metric(n):
    M = np.array([[0,1], [1,0]])
    N = M.copy()
    for i in range(n-1):
        N = direct_sum(N, M)
    return N

@njit
def transvection(h, v):
    return v ^ inner(v,h)&h

@njit
def mult_transv(h_list, v):
    # apply multiple transvections in order, from right to left
    input_state = v.copy()
    for i in range(len(h_list)-1, -1, -1):
        input_state = transvection(h_list[i], input_state)
    return input_state

@njit
def find_transvection(e, v):
    n = len(e)//2
    if np.array_equal(e, v):
         return [np.zeros(2*n, dtype=np.int64)]
    # find h or h1,h2 such that Zh(e) = v
    if inner(e, v) == 1:
        h = (e+v)%2
        return [h]
    else:
        z = np.zeros(2*n, dtype=np.int64)
        # look for j where both (e_2j, e_2j+1) and (v_2j, v_2j+1) are different from (0, 0)
        for j in range(n):
            if (e[2*j] + e[2*j+1] != 0) and (v[2*j] + v[2*j+1] != 0):   #found the pair
                z[2*j] = (e[2*j] + v[2*j])%2 
                z[2*j+1] = (e[2*j+1] + v[2*j+1])%2 
                if z[2*j]+z[2*j+1] == 0:    # they were the same so they added to 00
                    z[2*j+1] = 1
                    if e[2*j] != e[2*j+1]:
                        z[2*j] = 1
                return [(e+z)%2, (v+z)%2]
            
        # otherwise look for (v, v) = (0, 0) and (e, e) different from (0, 0)
        for j in range(n):
            if (e[2*j] + e[2*j+1] != 0) and (v[2*j] + v[2*j+1] == 0):   #found the pair
                if e[2*j] == e[2*j+1]:
                    z[2*j+1] = 1
                else:
                    z[2*j+1] = e[2*j]
                    z[2*j] = e[2*j+1]
                break
        # finally look for (v, v) = (0, 0) and (e, e) different from (0, 0)
        for k in range(n):
            if (e[2*k] + e[2*k+1] == 0) and (v[2*k] + v[2*k+1] != 0):
                if v[2*k] == v[2*k+1]:
                    z[2*k+1] = 1
                else:
                    z[2*k+1] = v[2*k]
                    z[2*k] = v[2*k+1]
                break
        return [(e+z)%2, (v+z)%2]
        
    
@njit
def symplectic_n3(n, i):
    # Returns i-th symplectic matrix (Sp(2n)), 0 \leq i \leq 2^{n^2} \prod_{j=1}^n (4^j -1) 
    # step 1
    s = 2**(2*n)-1          
    k = i%s + 1             
    # step 2
    f1 = np.array([(k//(2**j))%2 for j in range(2*n)], dtype=np.int64)        
    # step 3
    e1 = np.zeros(2*n, dtype = np.int64)
    e1[0] = 1
    T = find_transvection(e1, f1)     
    # step 4     
    bits = [((i//s)//(2**j))%2 for j in range(2*n)]
    # step 5
    e_prime = e1.copy()
    for j in range(2, 2*n):
        e_prime[j] = bits[j-1]
    h0 = mult_transv(T, e_prime)
    # step 6
    T_prime = []
    if bits[0] == 1:
        T_prime.append(h0)
    else:
        T_prime.append(f1)
        T_prime.append(h0)
    e2 = np.zeros(2*n, dtype = np.int64)
    e2[1] = 1
    f2 = mult_transv(T_prime + T, e2)
    # step 7
    if n==1:
        f1f2 = np.zeros((2, 2*n), dtype=np.int64)
        f1f2[0] = f1
        f1f2[1] = f2
        return f1f2.T
    else:
        prev_g = symplectic_n3(n-1, (i//s)//2**(2*n-1))
        enlarged_g_transp = direct_sum(np.eye(2, dtype=np.int64), prev_g).T
        gi = np.zeros((2*n, 2*n), dtype=np.int16)
        for row_index in range(2*n):
            gi[row_index] = mult_transv(T_prime + T, enlarged_g_transp[row_index])
        
        return gi.T
        
@njit
def symplecticInverse(n, g):
    # step 1
    v = g[:, 0]
    w = g[:, 1]
    # step 2
    e1 = np.zeros(2*n, dtype = np.int64)
    e1[0] = 1
    T = find_transvection(v, e1)
    # step 3
    Tw = mult_transv(T, w)
    b = Tw[0]
    bl = [Tw[i] for i in range(2, 2*n)]
    h0 = np.zeros(2*n, dtype = np.int64)
    h0[0] = 1
    h0[2: 2*n] = np.array(bl)
    # step 4
    zv = sum([v[i]*2**i for i in range(2*n)]) - 1
    bits = [b] + bl
    zw = sum([bits[i]*2**i for i in range(2*n-1)])
    cvw = zw*(2**(2*n)-1) + zv
    # step 5
    if n == 1:
        return cvw
    else:
        T_final = [h0] + T
        if b == 0:
            T_final = [e1] + T_final
        g_prime = np.array([mult_transv(T_final, row) for row in g.T]).T
        g_next = g_prime[2:, 2:]
        return symplecticInverse(n-1, g_next)*(2**(2*n)-1)*(2**(2*n-1)) + cvw
        
    
    
# n = 2
# i = 0

# s = 2**(2*n)-1
# prod = 1
# for j in range(1, n+1):
#     prod *= 4**j-1
# dim_symplectic = 2**(n*n)*prod

# Sn = []
# saved_index = []
# for i in range(dim_symplectic):
#     g = symplectic_n3(n, i)
#     if True in [np.array_equal(g, gprev) for gprev in Sn]:
#         print("Trovati due uguali a i =", i)
#         break
#     else:
#         Sn.append(g)
#         symp_ind = symplecticInverse(n, g)
#         if symp_ind in saved_index:
#             print("Trovati due indici uguali a i =", i)
#             break
#         saved_index.append(symp_ind)
        
# symplecticInverse(n, g)

# [np.linalg.multi_dot([symp.T, np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]]), symp])%2 for symp in Sn]