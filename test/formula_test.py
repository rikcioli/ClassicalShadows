# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:55:35 2024

@author: rikci
"""

import numpy as np
import math

N = 4
p = 1
q = 2

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

weights_matrix = np.zeros((N, N))
for i in range(N):
    w = i+1
    for j in range(N):
        if j == i:
            weights_matrix[i, i] = 1 - (2*p/(N*(N-1))) * (w*(N-w)*(q**2-1)**2/(q**4-1) + (2*(q**2-1)/(q**4-1))*math.comb(w, 2))
        elif j == i+1:
            weights_matrix[i, j] = (2*p/(N*(N-1))) * (2*(q**2-1)/(q**4-1))*math.comb(w+1, 2)
        elif j == i-1:
            weights_matrix[i, j] = (2*p/(N*(N-1))) * ((w-1)*(N-w+1)*(q**2-1)**2/(q**4-1))
            
print(weights_matrix)

# # initial_condition = np.zeros(N)
# # initial_condition[-1] = 1

# # for depth in range(2, 20):
# #     weights_matrix_evolved = np.linalg.multi_dot([weights_matrix for i in range(depth)])
# #     final_weights = np.dot(weights_matrix_evolved, initial_condition)
# #     eigenval = sum((final_weights[w]*(q+1)**(-w-1) for w in range(N)))
# #     print("\nDepth:", depth)
# #     print(weights_matrix_evolved)
# #     print(eigenval**(-1))


# # WEIGHTS FOR BLACK HOLE EVOLUTION

# weight_mat2 = np.zeros((N, N))
# for i in range(N):
#     w = i+1
#     for j in range(N):
#         if i == j:
#             weight_mat2[i, j] = 1 - (2*p/(N*(N-1)))*((q**2-1)/(q**4-1))*((q**2-1)*w*(N-w) + w*(w-1))
#         elif j == i-1:
#             weight_mat2[i, j] = (2*p/(N*(N-1)))*((q**2-1)**2/(q**4-1))*(w-1)*(N-w+1)
#         elif j == i+1:
#             weight_mat2[i, j] = (2*p/(N*(N-1)))*((q**2-1)/(q**4-1))*(w+1)*w

# print(weight_mat2)

d = q

brown_mat = np.empty((N+1, N+1))
alpha_nm = np.empty((N+1, N+1))

for n in range(N+1):
    for m in range(N+1): 
        alpha_nm[n, m] = (d**(n-m)/math.comb(N, m))*sum((d**(2*n-4*r)*(d**4-1)**(r-n)*multinomial([r, n-2*r, N//2-n+r])*math.comb(n-r, m-n+r) if m-n+r>=0 else 0 for r in range(n//2+1)))
        brown_mat[n, m] = alpha_nm[n, m]*(d**2-1)**n
print(brown_mat)


