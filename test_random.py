# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:48:50 2024

@author: rikci
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from circuit import StabilizerCircuit

n = 2
prod = 1
N_shadows = 50
N_samples = 1000
N = N_shadows*N_samples

# for j in range(1, n+1):
#     prod *= 4**j-1
# dim_symplectic = prod*2**(n*n)
# random_array = np.array([random.randint(0, dim_symplectic-1) for i in range(N)])

circs = [StabilizerCircuit(2) for i in range(N)]
[sc.randEvolution(1) for sc in circs]
indices = [gate.params[0] for sc in circs for gate in sc.circuit]
        

#[sc.run() for sc in circs]

plt.figure(dpi=600)
plt.errorbar(list(range(N)), indices, fmt='bo')
plt.ylabel(r"<U Z_2 U^{\dag}>")
plt.xlabel("Sample number")
plt.grid(axis = 'y', linestyle = '--')
#plt.xticks(rotation=70)  
plt.show() 

counts = dict(Counter(indices))
sorted_counts = dict(sorted(counts.items()))
keylist = list(sorted_counts.keys())
keycounts = list(sorted_counts.values())
yerr = np.sqrt(keycounts)/N
keycounts = np.array(keycounts)/N
plt.figure(dpi=600)
plt.bar(keylist, keycounts, yerr = yerr, width=0.005)
plt.ylabel("Probabilities")
plt.grid(axis = 'y', linestyle = '--')
plt.xticks(rotation=70)  
#plt.savefig('D:/Fisica/TESI/Final results/filename.png')
plt.show()  