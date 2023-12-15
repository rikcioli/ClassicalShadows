# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import Symplectic as sp

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
        
    def _gauss_elim(self, M):

        m, n = M.shape
    
        i=0
        j=0
    
        while i < m and j < n:
            # find value and index of largest element in remainder of column j
            k = np.argmax(M[i:, j]) +i
    
            # swap rows
            #M[[k, i]] = M[[i, k]] this doesn't work with numba
            temp = np.copy(M[k])
            M[k] = M[i]
            M[i] = temp
    
            aijn = M[i, j:]
    
            col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
    
            col[i] = 0 #avoid xoring pivot row with itself
    
            flip = np.outer(col, aijn)
    
            M[:, j:] = M[:, j:] ^ flip
    
            i += 1
            j +=1
    
        return M
    
    def _fullrank_X(self):
        # Applies hadamards to make X matrix have full rank
        gauss_X = self._gauss_elim(self.state[self.N:2*self.N, 0:self.N].copy())
        qubits = [i for i in range(self.N) if gauss_X[i,i] == 0]
        [self.H(a) for a in qubits]
        
        return qubits
    
    def _is_valid_state(self, state):
        det = np.linalg.det(state[:, 0:2*self.N])
        if det == 0:
            return False
        else:
            return True
    
    
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
        
    def randClifford(self, qubits_list):
        # Applies a random n-qubit clifford on qubits in qubits_list
        n = len(qubits_list)
        if n > self.N:
            raise RuntimeError("Number of given qubits exceeds N")
        s = 2**(2*n)-1
        prod = 1
        for j in range(1, n+1):
            prod *= 4**j-1
        dim_symplectic = 2**(n*n)*prod
        index_symplectic = random.randint(0, dim_symplectic - 1)
        
        r = np.array([random.randint(0, 1) for j in range(n)])
        s = np.array([random.randint(0, 1) for j in range(n)])
        gi = sp.symplectic_n3(n, index_symplectic)
        
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
        
    
    def measure(self, a):
        xa_positions = np.where(self.state[:, a] == 1)[0]     #save position of all stab and destab that do not commute with Z_a
        p = xa_positions[-1]
        if p >= self.N:      #if any stab does not commute with measure, state needs to be updated
            [self._rowsum(i, p) for i in xa_positions if (i!=p and i!=p-self.N)]      #update all stabilizers that anticommute with meas
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
    
    def randEvolution(self, depth):
        qubits = [i for i in range(self.N)]
        for layer in range(depth):
            random.shuffle(qubits)
            pairs = [qubits[2*i:2*(i+1)] for i in range(2)]
            for pair in pairs:
                self.randClifford(pair)
    
                   
N_qubits = 4

N_shots = 1

clifford_circ = StabilizerState(N_qubits)
clifford_circ.randEvolution(20)
print(clifford_circ.check_matrix)
print(clifford_circ._is_valid_state(clifford_circ.state))

# counts = clifford_circ.measure_mult_shots([0, 2], N_shots)



# print("Total counts are:", counts)
# clifford_circ.plot_histogram(counts)