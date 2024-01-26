# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:16:27 2023

@author: rikci
"""

import random
import symplectic as sp


class Instruction():
    
    def __init__(self, name, qubits_list):
        """Create a new instruction       

        Args:
            name (str): instruction name
            qubits (list): qubits involved in the instruction
        """
        self._name = name
        self._qubits = qubits_list
        
    @property
    def name(self):
        return self._name
    
    @property
    def qubits(self):
        return self._qubits
    
    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}"
            f", qubits={self.qubits!r}"
            ")"
        )


class RandomClifford(Instruction):
    
    rng = random.Random()
    
    def __init__(self, name, qubits_list, params = None):
        super().__init__(name, qubits_list)
        
        n = len(qubits_list)
        self._num_qubits = n
        
        if params is None:       
            prod = 1
            for j in range(1, n+1):
                prod *= 4**j-1
            dim_symplectic = prod*2**(n*n)
            index_symplectic = self.rng.randint(0, dim_symplectic-1)
            r_int = self.rng.randint(0, 2**n - 1)
            s_int = self.rng.randint(0, 2**n - 1)
            inverse = False
            params = (index_symplectic, r_int, s_int, inverse)
        
        self._params = params   
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, param_list):
        self._params = param_list.copy()
    
    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"qubits={self.qubits!r}"
            f", params={self.params!r}"
            ")"
        )
    
    def index_to_matrix(self):
        return sp.symplectic_n3(self.num_qubits, self.params[0])
    
    def dagger(self):
        self.params = (self.params[0], self.params[1], self.params[2], 1^self.params[3])
        
    