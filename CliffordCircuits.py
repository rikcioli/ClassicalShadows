# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

class StabilizerState():
    
    def __init__(self, N_qubits):
        self.N = N_qubits
        self.state = np.eye(2*self.N, 2*self.N+1, 0, dtype=int)
        
    @property
    def check_matrix(self) -> np.ndarray:
        """Returns the Clifford state of the system"""
        return self.state
    
    def _sum_rows(self, row_h, row_i):
        # given two rows h and i of 2*N+1 elements, returns h+i with correct rh
        xi = row_i[0:self.N]
        zi = row_i[self.N:2*self.N]
        xh = row_h[0:self.N]
        zh = row_h[self.N:2*self.N]
        
        # evaluate sum of g terms
        vec_g = (xi & zi)*(zh-xh) + (xi & (zi^1))*(2*xh-1)*zh + ((xi^1) & zi)*(1-2*zh)*xh  #calculate new rh based on rh, ri and sumg = \sum_j g_j
        sum_g = sum(vec_g)
        
        # evaluate rh and row_h+row_i and update
        rh = ((2*row_h[2*self.N] + 2*row_i[2*self.N] + sum_g)%4)/2   #calculate new rh based on rh, ri and sumg = \sum_j g_j
        row_h = row_h^row_i
        row_h[2*self.N] = rh
        return row_h

    def _rowsum(self, h, i):
        # given target h and input i, UPDATES IN PLACE h = h+i 
        self._sum_rows(self.state[h], self.state[i])    #update h row as sum of i and h rows
    
    
    def set_state(self, state):
        ### TO DO: check that this is a valid stabilizer state
        self.state = state.copy()
        
    def reset_state(self):
        self.state = np.eye(2*self.N, 2*self.N+1, 0, dtype=int)

    def CNOT(self, a, b):
        vec_r = self.state[:, 2*self.N]      # CAREFUL: these are just POINTERS to the columns of state
        xcol_a = self.state[:, a]
        xcol_b = self.state[:, b]
        zcol_a = self.state[:, self.N+a]
        zcol_b = self.state[:, self.N+b]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_b & (xcol_b ^ zcol_a ^ 1))    # WARN: value of vec_r is changed after this line (since same pointer)
        self.state[:, b] = xcol_b^xcol_a     # WARN: value of xcol_b = state[:, b] changed after this line
        self.state[:, self.N+a] = zcol_a^zcol_b      # WARN: value of zcol_a = state[:, N+a] changed after this line
    
    
    def H(self, a):
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a].copy()    # Necessary copy, pointers are not enough
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_a)    # WARN: value of vec_r is changed after this line
        self.state[:, a] = zcol_a
        self.state[:, self.N+a] = xcol_a
    
    
    def S(self, a):
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a]
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_a)
        self.state[:, self.N+a] = xcol_a^zcol_a
        
    
    def measure(self, a):
        xa_positions = np.where(self.state[:, a] == 1)[0]     #save position of all stab and destab that do not commute with Z_a
        p = xa_positions[-1]
        if p >= self.N:      #if any stab does not commute with measure, state needs to be updated
            [self._rowsum(i, p) for i in xa_positions if (i!=p and i!=p-self.N)]      #update all stabilizers that anticommute with meas
            self.state[p-self.N] = self.state[p].copy()          #set (p-n)th row equal to p, moving the non commuting stab to destabs
            self.state[p] = np.zeros([2*self.N+1])
            self.state[p, self.N+a] = 1
            rp = random.randint(0,1)
            self.state[p, 2*self.N] = rp
            return rp
        else:
            extra_row = np.zeros([2*self.N+1], dtype=int)
            for i in xa_positions:
                extra_row = self._sum_rows(extra_row, self.state[i+self.N])
            r_extra = extra_row[2*self.N]
            return r_extra
        
    def measure_mult(self, qubits_list):
        return [self.measure(a) for a in qubits_list]
    
    def measure_mult_shots(self, qubits_list, n_shots):
        current_state = self.state.copy()
        outcomes = []
        for i in range(n_shots):
            measurements = [self.measure(a) for a in qubits_list]
            outcome_str = ''.join([str(item) for item in measurements])
            outcomes.append(outcome_str)     
            self.state = current_state.copy()       #n^2 complexity
        counts = dict(Counter(outcomes))
        return counts 
    
    def plot_histogram(self, counts):
        sorted_counts = dict(sorted(counts.items()))
        keylist = list(sorted_counts.keys())
        keycounts = list(sorted_counts.values())
        n_shots = sum(keycounts)
        yerr = np.sqrt(keycounts)/n_shots
        keycounts = np.array(keycounts)/n_shots
        plt.figure(dpi=600)
        plt.bar(keylist, keycounts, width = 0.5, yerr = yerr)
        plt.ylabel("Probabilities")
        plt.grid(axis = 'y', linestyle = '--')
        plt.xticks(rotation=70)  
        #plt.savefig('D:/Fisica/TESI/Final results/filename.png')
        plt.show()  
    
                   
N_qubits = 1000

N_shots = 1000

clifford_circ = StabilizerState(N_qubits)
clifford_circ.H(0)
clifford_circ.CNOT(0, 250)
clifford_circ.H(0)
clifford_circ.S(0)
clifford_circ.S(0)
clifford_circ.H(0)
counts = clifford_circ.measure_mult_shots([0,250], N_shots)


#counts = clifford_circ.run(my_circuit, [0,1], N_shots)

print("Total counts are:", counts)
clifford_circ.plot_histogram(counts)