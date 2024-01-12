# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np

class StabilizerState():
    
    def __init__(self, N_qubits):
        self.N = N_qubits
        self.state = np.eye(2*self.N, 2*self.N+1, 0, dtype=int)
        
    @property
    def stabilizer_table(self) -> np.ndarray:
        """Returns the stabilizer table of the system"""
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
        self.state[h] = self._sum_rows(self.state[h], self.state[i])    #update h row as sum of i and h rows
        
    def _gauss_elim(self, M):

        M = M.copy()
        m, n = M.shape
    
        i=0
        j=0
    
        while i < m and j < n:
            # find value and index of largest element in remainder of column j
            k = np.argmax(M[i:, j]) +i
    
            # swap rows
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

    
    def _is_valid_state(self, state = None):
        if state is None:
            state = self.state
            
        det = np.linalg.det(state[:, 0:2*self.N])
        if det == 0:
            return False
        else:
            return True
        
    
    def dot_zero(self, state = None):
        # returns overlap with |0> state <0|\psi>
        if state is None:
            state = self.state
        state = self._gauss_elim(state[self.N:])
        X_matrix = state[:, :self.N]
        Z_matrix = state[:, self.N:2*self.N]
        vec_r = state[:, 2*self.N]
        
        s = 0
        for i in range(self.N):
            if 1 in X_matrix[i, i:]:
                s += 1
            elif (vec_r[i] != 0):
                    return 0
        return 2**(-s/2)
    
                   
