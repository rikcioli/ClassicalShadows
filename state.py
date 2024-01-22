# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:13:00 2023

@author: rikci
"""

import numpy as np
from numba import njit


class StabilizerState():
    
    def __init__(self, N_qubits, state = None):
        self._N = N_qubits
        if state is None:
            self._state = np.eye(2*N_qubits, 2*N_qubits+1, 0, dtype=np.int64)
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
        
    @staticmethod
    @njit(cache=True)
    def _sum_rows(row_h, row_i):
        # given two rows h and i of 2*N+1 elements, returns h+i with correct rh
        N = (len(row_h)-1)//2
        xi = row_i[0:N]
        zi = row_i[N:2*N]
        xh = row_h[0:N]
        zh = row_h[N:2*N]
        
        # evaluate sum of g terms
        vec_g = (xi & zi)*(zh-xh) + (xi & (zi^1))*(2*xh-1)*zh + ((xi^1) & zi)*(1-2*zh)*xh  #calculate new rh based on rh, ri and sumg = \sum_j g_j
        sum_g = sum(vec_g)
        
        # evaluate rh and row_h+row_i and update
        rh = ((2*row_h[2*N] + 2*row_i[2*N] + sum_g)%4)/2   #calculate new rh based on rh, ri and sumg = \sum_j g_j
        row_h = row_h^row_i
        row_h[2*N] = rh
        return row_h
    
    def _rowsum(self, h, i):
        # given target h and input i, UPDATES IN PLACE h = h+i         
        self.state[h] = self._sum_rows(self.state[h], self.state[i])    #update h row as sum of i and h rows
        

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
    
    def gauss_elim_stab(self, state_stabs):
        """Given lower part of state table matrix returns gaussian eliminated version 
        using rowsum
        """
        M = state_stabs.copy()
        m, n = M.shape
        i, j = 0, 0
        
        while i < m and j < n:
            # find position of first 1 in the remainder of column j
            pos = np.argmax(M[i:, j]) + i
            
            if M[pos, j] == 0:
                j += 1
            else:
                # swap rows
                if pos > i:
                    temp = M[i].copy()
                    M[i] = M[pos]
                    M[pos] = temp
                
                # rowsum over all rows that have a 1 below col j
                for k in range(i+1, m):
                    if M[k, j] == 1:
                        M[k] = self._sum_rows(M[k], M[i])
                i += 1
                j += 1
        
        return M
        
    def dot_zero(self, state = None):
        # returns overlap with |0> state |<0|\psi>|
        if state is None:
            state = self.state
        state = self.gauss_elim_stab(state[self.N:])
        X_matrix = state[:, :self.N]
        vec_r = state[:, 2*self.N]
        
        s = 0
        for l in range(self.N):
            if 1 in X_matrix[l, l:]:
                s += 1
            elif (vec_r[l] != 0):
                #print(state, "\n")
                return 0
        return 2**(-s/2)
    
    def skew_product(self, stab1, stab2):
        return np.dot(stab1[:self.N], stab2[self.N:2*self.N])%2 ^ np.dot(stab1[self.N:2*self.N], stab2[:self.N])%2
    
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
            
        

                   

                
        
        