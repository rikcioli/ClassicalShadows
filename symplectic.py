# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:15:26 2023

@author: rikci
"""

import numpy as np


def inner(v, w):
    n = len(v)//2
    v = v.copy()
    w = w.copy()
    for i in range(n):
        even = w[2*i].copy()
        w[2*i] = w[2*i+1].copy()
        w[2*i+1] = even.copy()
    return np.dot(v, w.T)%2

def direct_sum(A, B):
    dim = len(A) + len(B)
    dirsum = np.zeros([dim, dim], dtype = int)
    dirsum[0:len(A), 0:len(A)] = A
    dirsum[len(A):dim, len(A):dim] = B
    return dirsum
    

def transvection(h, v):
    return (v + inner(v,h)*h)%2

def mult_transv(h_list, v):
    # apply multiple transvections in order, from right to left
    input_state = v.copy()
    for h in reversed(h_list):
        input_state = transvection(h, input_state)
    return input_state

def find_transvection(e, v):
    n = len(e)//2
    if np.array_equal(e, v):
         return [np.zeros(2*n, dtype=int)]
    # find h or h1,h2 such that Zh(e) = v
    if inner(e, v) == 1:
        h = (e+v)%2
        return [h]
    else:
        z = np.zeros(2*n, dtype=int)
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
        
    

def symplectic_n3(n, i):
    # Returns i-th symplectic matrix (Sp(2n)), 0 \leq i \leq 2^{n^2} \prod_{j=1}^n (4^j -1) 
    # step 1
    s = 2**(2*n)-1          
    k = i%s + 1             
    # step 2
    f1 = np.array([(k//(2**j))%2 for j in range(2*n)])        
    # step 3
    e1 = np.zeros(2*n, dtype = int)
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
    e2 = np.zeros(2*n, dtype = int)
    e2[1] = 1
    f2 = mult_transv(T_prime + T, e2)
    # step 7
    if n==1:
        return np.array([f1, f2]).T
    else:
        prev_g = symplectic_n3(n-1, (i//s)//2**(2*n-1))
        enlarged_g_transp = direct_sum(np.eye(2, dtype=int), prev_g).T
        gi = np.array([mult_transv(T_prime + T, row) for row in enlarged_g_transp]).T
        return gi
        
    
    
# n = 2
# i = 0

# s = 2**(2*n)-1
# prod = 1
# for j in range(1, n+1):
#     prod *= 4**j-1
# dim_symplectic = 2**(n*n)*prod

# Sn = []
# for i in range(dim_symplectic):
#     g = symplectic_n3(n, i)
#     if True in [np.array_equal(g, gprev) for gprev in Sn]:
#         print("Trovati due uguali a i =", i)
#         break
#     else:
#         Sn.append(g)

# [np.linalg.multi_dot([symp.T, np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]]), symp])%2 for symp in Sn]