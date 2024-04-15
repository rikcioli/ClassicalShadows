# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:45:12 2024

@author: rikci
"""

import numpy as np
import matplotlib.pyplot as plt
import math

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

avgval_perN_perdepth = []
sdomval_perN_perdepth = []

avgexact_perN_perdepth = []
sdomexact_perN_perdepth = []

start_N = 4
N_list = list(range(start_N, 18, 2))

for N_qubits in N_list:
    
    avgval_per_depth = []
    sdomval_per_depth = []

    avgexact_per_depth = []
    sdomexact_per_depth = []
    
    N_shadows = 50
    N_samples = 1000
    depths = [i for i in range(1, 19)]
    
    
    # GHZ stab
    pauli_stab = [1 for i in range(N_qubits)]
    pauli_stab += [0 for i in range(N_qubits)]
    
    # Evaluate exact eigenvalues
    N = N_qubits
    p = 1
    d = 2
    
    where_pauli = set(np.where(pauli_stab)[0]%N)
    initial_condition = np.zeros(N+1)
    initial_condition[len(where_pauli)] = 1
    
    for depth in depths:
        
        # eval eigenvalue
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
            
        eigenval = sum((final_weights[w]*(d+1)**(-w) for w in range(1, N+1)))
        
        # load array
        mean_over_sh_per_sample = np.loadtxt('D:/SPYDER BACKUP/.spyder-py3/CliffordCircuits/Results/Compare_expval/Test GHZ/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_mean_over_sh_per_sample.csv', delimiter=',')
        
        # Evaluate expval from data with inversion formula
        expval_per_sample = (2**N_qubits+1)*mean_over_sh_per_sample
        if sum(pauli_stab) == 0:
            expval_per_sample -= 2**N_qubits
        
        # Evaluate expval from data with exact eigenvalues
        expval_ps_exact = mean_over_sh_per_sample/eigenval
        
        # Save both in current depth array
        avgval_per_depth.append(np.mean(expval_per_sample))
        sdomval_per_depth.append(np.std(expval_per_sample)/np.sqrt(N_samples))
        avgexact_per_depth.append(np.mean(expval_ps_exact))
        sdomexact_per_depth.append(np.std(expval_ps_exact)/np.sqrt(N_samples))
        
    avgval_perN_perdepth.append(np.array(avgval_per_depth))
    sdomval_perN_perdepth.append(np.array(sdomval_per_depth))
    avgexact_perN_perdepth.append(np.array(avgexact_per_depth))
    sdomexact_perN_perdepth.append(np.array(sdomexact_per_depth))


    plt.figure(dpi=600)
    plt.errorbar(depths, avgval_per_depth, yerr = sdomval_per_depth, fmt='bo')
    plt.errorbar(depths, avgexact_per_depth, yerr = sdomexact_per_depth, fmt='ro')
    plt.xlabel(r"t")
    plt.xticks(np.arange(1, 19))
    plt.xlim(0.5, 18.5)
    plt.ylabel("$<\sigma_x^{\otimes N}>$")
    plt.title(r'$N = $'+str(N_qubits)+', $M = $'+str(N_shadows)+', $N_{s} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()
    
# Plot tbar as a function of N
eps_list = [0.1*2**(-j) for j in range(5)]
tbar_pereps_perN = []

avgval_perN_perdepth = np.array(avgval_perN_perdepth)
avgexact_perN_perdepth = np.array(avgexact_perN_perdepth)

for eps in eps_list:
    tbar = []
    for i in range(len(N_list)):
        for j, expval in enumerate(avgval_perN_perdepth[i]):
            expexact = avgexact_perN_perdepth[i, j]
            if abs(expval-expexact) < eps:
                depth = j+1
                tbar.append(depth)
                break
    tbar_pereps_perN.append(tbar)


plt.figure(dpi=600)
for i in range(len(eps_list)):
    plt.plot(N_list, tbar_pereps_perN[i], '-o', label = "$\epsilon$ = "+str(eps_list[i]))

plt.xlabel("N")
plt.ylabel(r"$\bar{t}$")
plt.legend(loc = 'upper right', bbox_to_anchor=(1.05, 1))
plt.title(r'$M = $'+str(N_shadows)+', $N_{s} = $'+str(N_samples), fontsize=16)
plt.grid(axis = 'y', linestyle = '--')
plt.show()
