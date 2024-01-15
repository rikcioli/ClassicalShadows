# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np


class StabilizerState():
    
    def __init__(self, N_qubits, state = None):
        self._N = N_qubits
        if state is None:
            self._state = np.eye(2*N_qubits, 2*N_qubits+1, 0, dtype=int)
        else:
            self._state = state.copy()
        
    @property
    def N(self) -> int:
        return self._N
    
    @N.setter
    def N(self, value) -> int:
        self._N = value
    
    @property
    def state(self) -> np.ndarray:
        """Returns the stabilizer table of the system"""
        return self._state
    
    @state.setter
    def state(self, input_state) -> np.ndarray:
        """Set the stabilizer table of the system"""          
        self._state = input_state.copy()
        
    
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
    
    def _fullrank_X(self):
        # Applies hadamards to make X matrix have full rank
        gauss_X = self._gauss_elim(self.state[self.N:2*self.N, 0:self.N].copy())
        qubits = [i for i in range(self.N) if gauss_X[i,i] == 0]
        [self.H(a) for a in qubits]
        
        return qubits

    
    def _is_valid_state(self, state = None):
        if state is None:
            state = self.state
        if len(state) != 2*self.N:
            return False
        
        det = np.linalg.det(state[:, 0:2*self.N])
        if det == 0:
            return False
        else:
            return True
        
    
    def dot_zero(self, state = None):
        # returns overlap with |0> state <0|\psi>
        if state is None:
            state = self.state
        state = gauss_elim(state[self.N:])
        X_matrix = state[:, :self.N]
        vec_r = state[:, 2*self.N]
        
        s = 0
        for l in range(self.N):
            if 1 in X_matrix[l, l:]:
                s += 1
            elif (vec_r[l] != 0):
                    return 0
        return 2**(-s/2)
    
    def skew_product(self, state1, state2):
        return np.dot(state1[:self.N], state2[self.N:2*self.N])%2 ^ np.dot(state1[self.N:2*self.N], state2[:self.N])%2
    
    def expval(self, stabilizer, state = None):
        """
        Evaluate expectation value of given stabilizer on given state.

        Args:
            stabilizer (numpy array): stabilizer specified by an array of
            2*N_qubits elements (as in the check matrix)
        """
        if state is None:
            state = self.state
        if sum(stabilizer) == 0:    #if stab is identity expval is 1
            return 1
        # else evaluate skew product with the stabilizers of the state
        skew_with_stabs = [self.skew_product(stabilizer, state[i][:2*self.N]) for i in range(self.N, 2*self.N)]
        if 1 in skew_with_stabs:    # if it anticommutes with one of them expval is 0
            return 0
        else:       # else expval is +1 or -1, sign must be determined as in measure
            skew_with_destabs = [self.skew_product(stabilizer, state[i][:2*self.N]) for i in range(self.N)]
            positions = [i for i in range(self.N) if skew_with_destabs[i] == 1]
            extra_row = np.zeros([2*self.N+1], dtype=int)
            for i in positions:
                extra_row = self._sum_rows(extra_row, self.state[i+self.N])
            r_extra = extra_row[2*self.N]
            return (-1)**r_extra
            
        
def gauss_elim(M):

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
                   
