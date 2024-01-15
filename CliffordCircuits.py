# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from circuit import StabilizerCircuit


# insert EVEN number of qubits    
N_qubits = 2

N_shadows = 10000
depth = 1
N_shots = 1

sc = StabilizerCircuit(N_qubits)
#sc.H(0)
#sc.CNOT(0, 1)

state_prep_len = len(sc.circuit)
temp_circ = sc.circuit.copy()

random_circuits = []
outcomes = []

"""
for i in range(N_shadows):
    sc.circuit = temp_circ.copy()
    sc.randEvolution(depth)
    random_cliff = [gate for gate in sc.circuit[state_prep_len:]]
    sc.measure([i for i in range(N_qubits)])
    counts = sc.run(N_shots)
    
    random_circuits.append(random_cliff)
    outcomes.append(list(counts.keys())[0])
"""


paulis = [np.array([i,j,k,l]) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

x_axis = [int(i) for i in np.linspace(1, 10000, 50)]
y_axis = []
             
for N_shadows in x_axis:
    exp_values = []
    for i in range(N_shadows):
        sc.randEvolution(depth)
        sc.run(N_shots)
        
        expval = np.array([sc.expval(pauli) for pauli in paulis])
        exp_values.append(expval)
        
        sc.reset_state()
        sc.reset_circuit()
    
    exp_values = np.array(exp_values)
    avg_expvalues = np.mean(exp_values, 0)
    y_axis.append(avg_expvalues[1])
    std_expvalues = np.std(exp_values, 0)

plt.figure(dpi=600)
plt.plot(x_axis, y_axis, 'bo')
plt.ylabel("\mathbb{E} \langle\psi|U X_2 U^{\dag}|\psi\rangle")
plt.xlabel("N_shadows")
plt.grid(axis = 'y', linestyle = '--')
plt.xticks(rotation=70)  
#plt.savefig('D:/Fisica/TESI/Final results/filename.png')
plt.show()  

# print("Total counts are:", counts)
# sc.plot_histogram(counts)



# # save numpy array as csv file
# from numpy import savetxt

# # save to csv file
# savetxt('x_axis.csv', x_axis, delimiter=',')
# savetxt('y_axis.csv', y_axis, delimiter=',')

# from numpy import loadtxt
# # load array
# xaxis = loadtxt('x_axis.csv', delimiter=',')
# data = loadtxt('y_axis.csv', delimiter=',')
# # print the array
# print(data)