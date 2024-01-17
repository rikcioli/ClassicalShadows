# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:07:13 2024

@author: rikci
"""

import numpy as np
from circuit import StabilizerCircuit
from state import StabilizerState
import copy
import matplotlib.pyplot as plt
import tqdm
from collections import Counter
from numpy import savetxt
import warnings

from timeit import default_timer as timer



max_depth = 10
depths = [i for i in range(1, max_depth+1)]
min_qubits = 2
max_qubits = 12

avgfid_per_N_qubits = []
sdomfid_per_N_qubits = []

save_results = True
if not save_results:
    print("WARNING: save_results set to False", flush=True)
    
    
timetable = []

for N_qubits in range(min_qubits, max_qubits+2, 2):
    
    start = timer()
    sc = StabilizerCircuit(N_qubits)

    # Prepare arbitrary initial state
    
    sc.randClifford([a for a in range(N_qubits)])
    sc.run()
    phi = sc.state.copy()
    
    N_shadows = 50
    N_samples = 1000
    avgfid_per_depth = []
    sdomfid_per_depth = []
    
    est_time = N_samples*(0.043*N_qubits**2 - 0.22*N_qubits + 1.24)*max_depth*(max_depth+1)/10
    
    print("Running program with "+str(N_qubits)+" qubits", flush=True)
    print("N_shadows = "+str(N_shadows)+", N_samples =", N_samples)
    print("Estimated time = ")
    
    for depth in tqdm.tqdm(depths, desc="Evaluating circuit depths up to "+str(depths[-1])):
        
        shadows = sc.saveShadows(N_shadows*N_samples, depth)      
        random_circuits = shadows[0]
        outcomes = shadows[1]
        
        fidelity_per_sample = np.zeros(N_samples)
        
        for sample_number in range(N_samples):
            
            circuits_current_sample = random_circuits[N_shadows*sample_number:N_shadows*(sample_number+1)]
            outcomes_current_sample = outcomes[N_shadows*sample_number:N_shadows*(sample_number+1)]
            Uphi_states = [StabilizerCircuit(N_qubits, phi, circuits_current_sample[r]) for r in range(N_shadows)]
            [[Uphi_states[r].X(i) for i in range(N_qubits) if outcomes_current_sample[r][i]] for r in range(N_shadows)]
            [state.run() for state in Uphi_states]
            
            fidelity = ((2**N_qubits+1)/N_shadows)*sum((state.dot_zero()**2 for state in Uphi_states)) - 1 
            fidelity_per_sample[sample_number] = fidelity
        
        avg_fidelity = np.mean(fidelity_per_sample)
        sdom_fidelity = np.std(fidelity_per_sample)/np.sqrt(N_samples)
        if save_results:
            savetxt('Results/Fidelity/'+str(N_qubits)+'Q-'+str(depth)+'D-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_fidelity_per_sample.csv', fidelity_per_sample, delimiter=',')
        
        avgfid_per_depth.append(avg_fidelity)
        sdomfid_per_depth.append(sdom_fidelity)
    
    if save_results:
        savetxt('Results/Fidelity/'+str(N_qubits)+'Q-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_avg_fidelity_per_depth.csv', avgfid_per_depth, delimiter=',')
        savetxt('Results/Fidelity/'+str(N_qubits)+'Q-'+str(N_shadows)+'Sh-'+str(N_samples)+'S_sdom_fidelity_per_depth.csv', sdomfid_per_depth, delimiter=',')
    
    end = timer()
    print("Time taken for", N_qubits+':'+str(end-start))
    timetable.append(end-start)
    
    
    plt.figure(dpi=600)
    plt.errorbar(depths, avgfid_per_depth, yerr = sdomfid_per_depth)
    plt.xlabel(r"$Depth$")
    plt.ylabel("Fidelity")
    plt.title(r'$N_{qubits} = $'+str(N_qubits)+', $N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
    plt.grid(axis = 'y', linestyle = '--')
    plt.show()  
    
    avgfid_per_N_qubits.append(avgfid_per_depth)
    sdomfid_per_N_qubits.append(sdomfid_per_depth)


"""
x_axis = np.array([0]+list(range(min_qubits, max_qubits+2, 2)))
timetable = [0] + timetable

import scipy.optimize as spopt

def pol2(x, a, b, c):
    return a*x**2 + b*x + c

def pol3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def pol4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

popt2, pcov2 = spopt.curve_fit(pol2, x_axis, timetable)
print(popt2)
print(np.diag(pcov2))

popt3, pcov3 = spopt.curve_fit(pol3, x_axis, timetable)
print(popt3)
print(np.diag(pcov3))

popt4, pcov4 = spopt.curve_fit(pol4, x_axis, timetable)
print(popt4)
print(np.diag(pcov4))


plt.figure(dpi=600)
plt.plot(x_axis, timetable, 'bo')
plt.xlabel(r"$N_{qubits}$")
plt.ylabel("Time (s)")
plt.title(r'$N_{shadows} = $'+str(N_shadows)+', $N_{samples} = $'+str(N_samples), fontsize=16)
plt.grid(axis = 'y', linestyle = '--')
plt.plot(x_axis, pol2(x_axis, *popt2), 'r-',
          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt2))
plt.plot(x_axis, pol3(x_axis, *popt3), 'g-',
          label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt3))
plt.plot(x_axis, pol4(x_axis, *popt4), 'b-',
          label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(popt4))
plt.legend()
plt.show()  
"""