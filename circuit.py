# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:00:55 2023

@author: rikci
"""

import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import symplectic as sp
from state import StabilizerState
from instruction import Instruction, RandomClifford


class StabilizerCircuit(StabilizerState):
    
    def __init__(self, N_qubits):
        super().__init__(N_qubits)
        self._circuit = []
    
    @property
    def circuit(self):
        return self._circuit
        
    def set_state(self, state):
        ### TO DO: check that this is a valid stabilizer state
        self.state = state.copy()
        
    def reset_state(self):
        self.state = np.eye(2*self.N, 2*self.N+1, 0, dtype=int)
        
    def _CNOT(self, pair):
        a, b = pair[0], pair[1]
        vec_r = self.state[:, 2*self.N]      # CAREFUL: these are just POINTERS to the columns of state
        xcol_a = self.state[:, a]
        xcol_b = self.state[:, b]
        zcol_a = self.state[:, self.N+a]
        zcol_b = self.state[:, self.N+b]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_b & (xcol_b ^ zcol_a ^ 1))    # WARN: value of vec_r is changed after this line (since same pointer)
        self.state[:, b] = xcol_b^xcol_a     # WARN: value of xcol_b = state[:, b] changed after this line
        self.state[:, self.N+a] = zcol_a^zcol_b      # WARN: value of zcol_a = state[:, N+a] changed after this line
    
    
    def _H(self, a):
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a].copy()    # Necessary copy, pointers are not enough
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_a)    # WARN: value of vec_r is changed after this line
        self.state[:, a] = zcol_a
        self.state[:, self.N+a] = xcol_a
    
    
    def _S(self, a):
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a]
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_a)
        self.state[:, self.N+a] = xcol_a^zcol_a
        
    def _U(self, qubits_list, matrices, vectors):
        """Apply most generic clifford unitary as specified by matrices alpha, 
        beta, gamma, delta and vectors r, s

        """
        
        n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
        if n > self.N:
            raise RuntimeError("Number of given qubits exceeds N")
            
        alpha = matrices[0]
        beta = matrices[1]
        gamma = matrices[2]
        delta = matrices[3]
        r = vectors[0]
        s = vectors[1]
        
        x = np.array([self.state[:, i] for i in qubits_list])
        z = np.array([self.state[:, i+self.N] for i in qubits_list])
        
        # update minus sign
        beta_gammaT = np.dot(beta, gamma.T)
        beta_alphaT = np.dot(beta, alpha.T)
        delta_gammaT = np.dot(delta, gamma.T)
        delta_alphaT = np.dot(delta, alpha.T)
        
        minus_exponent = self.state[:, 2*self.N].copy()
        proj = np.ones(n, dtype=int)
        
        # we introduce two variables: i (which initial qubit) and order_i 
        # order_i means the position of qubit i in the subspace on which the clifford is acting
        # e.g. we act con qubit i = 3 and i = 7, then i = 3 is the first (order_i = 0)
        # and i = 7 is the second (order_i = 1)
        # this is useful as we need to map the alpha, beta (etc) n by n matrices to the actual qubits
        
        for order_i, i in enumerate(qubits_list):
            
            minus_exponent ^= (r[order_i]&x[order_i] ^ s[order_i]&z[order_i])
            
            B1 = np.array([np.dot(beta_gammaT[order_i], proj & z[:, l]) %2 for l in range(2*self.N)])
            proj[order_i] = 0
            B2 = np.array([np.dot(beta_alphaT[order_i], proj & x[:, l])%2 for l in range(2*self.N)])
            minus_exponent ^= x[order_i]&(B1^B2)
            
            D1 = np.array([np.dot(delta_gammaT[order_i], proj & z[:, l])%2 for l in range(2*self.N)])
            D2 = np.array([np.dot(delta_alphaT[order_i], proj & x[:, l])%2 for l in range(2*self.N)])            
            minus_exponent ^= z[order_i]&(D1^D2)
        
        self.state[:, 2*self.N] = minus_exponent
            
        
        # update state
        for order_q, q in enumerate(qubits_list):
            
            x_exponent = np.array([(np.dot(x[:, l], alpha[:, order_q]) + np.dot(z[:, l], gamma[:, order_q]))%2 for l in range(2*self.N)])
            z_exponent = np.array([(np.dot(x[:, l], beta[:, order_q]) + np.dot(z[:, l], delta[:, order_q]))%2 for l in range(2*self.N)])
            self.state[:, q] = x_exponent
            self.state[:, q+self.N] = z_exponent
        
        
        
    def _randClifford(self, qubits_list, params):
        """Applies clifford on qubits in qubits_list
        which clifford is given as three numbers in the params tuple:
            index_sp (int): specifies which symplectic matrix in Sp(2n)
            r (int): specifies minus sign of the transformed single qubit X gates
            s (int): specifies minus sign of the transformed single qubit S gates
        (thus r and s specifies which pauli we choose for each qubit)
        
        """
        n = len(qubits_list)
        index_sp = params[0]
        r = np.array([(params[1]>>j) % 2 for j in range(n)])
        s = np.array([(params[2]>>j) % 2 for j in range(n)])
        gi = sp.symplectic_n3(n, index_sp)
        
        alpha = np.zeros([n, n], dtype = int)
        beta = np.zeros([n, n], dtype = int)
        gamma = np.zeros([n, n], dtype = int)
        delta = np.zeros([n, n], dtype = int)
        
        for i in range(n):
            for j in range(n):
                alpha[i, j] = gi[2*j, 2*i]
                beta[i, j] = gi[2*j+1, 2*i]
                gamma[i, j] = gi[2*j, 2*i+1]
                delta[i, j] = gi[2*j+1, 2*i+1]
        
        self._U(qubits_list, [alpha, beta, gamma, delta], [r, s])
        
    
    def _measure(self, a):
        xa_positions = np.where(self.state[:, a] == 1)[0]     #save position of all stab and destab that do not commute with Z_a
        p = xa_positions[-1]
        if p >= self.N:      #if any stab does not commute with measure, state needs to be updated
            [self._rowsum(i, p) for i in xa_positions if (i!=p and i!=p-self.N)]      #update all stab and destab that anticommute with meas
            self.state[p-self.N] = self.state[p].copy()          #set (p-n)th row equal to p, moving the non commuting stab to destabs
            self.state[p] = np.zeros([2*self.N+1])          #set p-th row to all zeros except 1 in position N+a (the current measurement stabilizer)
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
        
                
    def CNOT(self, a, b):
        cnot = Instruction('CNOT', [(a, b)])
        self.circuit.append(cnot)
    
    def H(self, a):
        h = Instruction('H', [a])
        self.circuit.append(h)
        
    def S(self, a):
        s = Instruction('S', [a])
        self.circuit.append(s)
        
    def measure(self, qubits_list):
        meas = Instruction('measure', qubits_list)
        self.circuit.append(meas)
        
    def randClifford(self, qubits_list):
        cliff = RandomClifford('random_clifford', qubits_list)
        self.circuit.append(cliff)
        
    def randEvolution(self, depth):
        qubits = [i for i in range(self.N)]
        for layer in range(depth):
            random.shuffle(qubits)
            pairs = [qubits[2*i:2*(i+1)] for i in range(len(qubits)//2)]
            for pair in pairs:
                self.randClifford(pair)
    
    def run(self, shots):
        initial_state = self.state.copy()
        outcomes = []
        is_measured = False
        for i in range(shots):
            self.state = initial_state.copy()       #n^2 complexity
            for gate in self.circuit:
                if gate.name == 'measure':
                    is_measured = True
                    measurements = [self._measure(a) for a in gate.qubits]
                    outcome_str = ''.join([str(item) for item in measurements])
                    outcomes.append(outcome_str)
                    counts = dict(Counter(outcomes))
                    
                elif gate.name == 'random_clifford':
                    self._randClifford(gate.qubits, gate.params)
                    
                else:
                    func_to_call = getattr(StabilizerCircuit, '_'+gate.name)
                    [func_to_call(self, target) for target in gate.qubits]
            
        if is_measured == True:  
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
        
        