# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from circuit import StabilizerCircuit
from instruction import RandomClifford
import tqdm
# from bootstrap_confidence import bootstrap_dist



paulis = [np.array([i,j,k,l]) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

def test(N_qubits = 2, N_shadows = 50, N_samples = 100, depth = 1):
    circs = [StabilizerCircuit(N_qubits) for i in range(N_shadows*N_samples)]
    [sc.randEvolution(depth) for sc in circs]
    [sc.run() for sc in circs]
    exp_values = np.array([sc.expval(paulis[1]) for sc in circs])
    var_tot = np.var(exp_values)/len(exp_values)
    
    expval_per_sample = [np.mean(exp_values[sample_num*N_shadows : (sample_num+1)*N_shadows], 0) for sample_num in range(N_samples)]  
    
    expval = np.mean(expval_per_sample)
    medval = np.median(expval_per_sample)
    err = np.std(expval_per_sample)
    
    plt.figure(dpi=600)
    plt.errorbar(list(range(N_samples)), expval_per_sample, fmt='bo')
    plt.ylabel(r"<U Z_2 U^{\dag}>")
    plt.xlabel("Sample number")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    #plt.xticks(rotation=70)  
    plt.show() 
    
    counts = dict(Counter(expval_per_sample))
    sorted_counts = dict(sorted(counts.items()))
    keylist = list(sorted_counts.keys())
    keycounts = list(sorted_counts.values())
    yerr = np.sqrt(keycounts)/N_samples
    keycounts = np.array(keycounts)/N_samples
    plt.figure(dpi=600)
    plt.bar(keylist, keycounts, yerr = yerr, width=0.005)
    plt.ylabel("Probabilities")
    plt.grid(axis = 'y', linestyle = '--')
    plt.xticks(rotation=70)  
    plt.show()  
    
    return np.array([expval, medval, err])

def confidence_interval(N_qubits = 2, N_shadows = 50, N_samples = 100, depth = 1, repetitions = 100):
    true_res = 0
    
    results = np.array([test(N_qubits, N_shadows, N_samples, depth) for i in range(repetitions)])
    differences = abs(results[:, 0] - true_res)
    fell_inside = differences < results[:, 1]
    fraction_inside = fell_inside.sum()/repetitions
    
    partial_avgs = [fell_inside[20*i : 20*(i+1)].sum()/(repetitions//5) for i in range(5)]
    err = np.std(partial_avgs)
    
    return [fraction_inside, err]

# results = test(N_shadows = 50, N_samples=1000)
# print(results)

# insert EVEN number of qubits    
N_qubits = 2
depth = 1

N_samples_list = [1000 for i in range(1, 11)]
N_shadows = 50
y_axis = []
y_err = []
data_vs_x = []
             
for N_samples in tqdm.tqdm(N_samples_list):
    exp_values = []
    for i in range(N_shadows*N_samples):
        sc = StabilizerCircuit(N_qubits)
        sc.randClifford([0,1])
        sc.run()
        exp_values.append(sc.expval(paulis[1]))
    
    expval_per_sample = [np.mean(exp_values[sample_num*N_shadows : (sample_num+1)*N_shadows], 0) for sample_num in range(N_samples)]  
    data_vs_x.append(expval_per_sample)
    
    y_axis.append(np.mean(expval_per_sample))
    y_err.append(np.std(expval_per_sample))
    

plt.figure(dpi=600)
plt.errorbar([i for i in range(1, 11)], y_axis, y_err, fmt='bo')
plt.ylabel(r"<U Z_2 U^{\dag}>")
plt.xlabel("N_samples")
plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows), fontsize=16)
plt.grid(axis = 'y', linestyle = '--')
#plt.xticks(rotation=70)  
plt.show()  

# max_deviation_pos = np.argmax(abs(np.array(y_axis)))

# plt.figure(dpi=600)
# plt.errorbar(list(range(100*(max_deviation_pos+1))), data_vs_x[max_deviation_pos], fmt='bo')
# plt.ylabel(r"<U Z_2 U^{\dag}>")
# plt.xlabel("N_samples")
# plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows), fontsize=16)
# plt.grid(axis = 'y', linestyle = '--')
# #plt.xticks(rotation=70)  
# plt.show() 

# print(max_deviation_pos)
# print(np.mean(data_vs_x[max_deviation_pos]), np.std(data_vs_x[max_deviation_pos])/np.sqrt(len(data_vs_x[max_deviation_pos])))


# fraction = confidence_interval(N_samples = 100)
# print(fraction)