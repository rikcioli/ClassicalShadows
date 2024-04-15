# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:00:55 2023

@author: rikci
"""

import copy
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import symplectic as sp
from state import StabilizerState
from instruction import Instruction, RandomClifford
from numba import njit, types, typed


class StabilizerCircuit(StabilizerState):
    
    rng = random.Random()
    
    def __init__(self, N_qubits, state = None, circuit = None):
        super().__init__(N_qubits, state)
        if circuit is None:
            self._circuit = []
        else:
            self._circuit = copy.deepcopy(circuit)
    
    @property
    def circuit(self):
        return copy.deepcopy(self._circuit)
    
    @circuit.setter
    def circuit(self, circuit):
        self._circuit = copy.deepcopy(circuit)
        
    def reset_state(self):
        self._state = np.eye(2*self.N, 2*self.N+1, 0, dtype=np.int64)
    
    def reset_circuit(self):
        self._circuit = []
    
    def set_circuit(self, stabilizer_circuit):
        self._circuit = copy.deepcopy(stabilizer_circuit)
    
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
        if not isinstance(a, int):
            a = a[0]
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a].copy()    # Necessary copy, pointers are not enough
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_a)    # WARN: value of vec_r is changed after this line
        self.state[:, a] = zcol_a
        self.state[:, self.N+a] = xcol_a
    
    def _S(self, a):
        if not isinstance(a, int):
            a = a[0]
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a]
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & zcol_a)
        self.state[:, self.N+a] = xcol_a^zcol_a
        
    def _Sdg(self, a):
        if not isinstance(a, int):
            a = a[0]
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a]
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^(xcol_a & (1^zcol_a))
        self.state[:, self.N+a] = xcol_a^zcol_a
        
    def _X(self, a):
        if not isinstance(a, int):
            a = a[0]
        vec_r = self.state[:, 2*self.N]
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^zcol_a
    
    def _Z(self, a):
        if not isinstance(a, int):
            a = a[0]
        vec_r = self.state[:, 2*self.N]
        xcol_a = self.state[:, a]
        
        self.state[:, 2*self.N] = vec_r^xcol_a
        
        
    def _SWAP(self, pair):
        a, b = pair[0], pair[1]
        anew, bnew = self.state[:, b].copy(), self.state[:, a].copy()
        self.state[:, a], self.state[:, b] = anew, bnew
        
        Nanew, Nbnew = self.state[:, self.N+b].copy(), self.state[:, self.N+a].copy()
        self.state[:, self.N+a], self.state[:, self.N+b] = Nanew, Nbnew
        
        
    @staticmethod
    @njit(cache=True)
    def _Unew_static(state, qubits_list, matrices, vectors, conj):
        
        n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
        N = len(state)//2   # total number of qubits
        if n > N:
            raise RuntimeError("Number of given qubits exceeds N")
        
        alpha = matrices[0]
        beta = matrices[1]
        gamma = matrices[2]
        delta = matrices[3]
        
        r = vectors[0]
        s = vectors[1]
        
        # reduce and REMAP state
        reduced_state = np.empty((2*N, 2*n+1), dtype = np.int64)
        for order_i, i in enumerate(qubits_list):
            reduced_state[:, order_i] = state[:, i]
            reduced_state[:, order_i + n] = state[:, i + N]
        reduced_state[:, 2*n] = state[:, 2*N]
        
        
        # create matrix with ones positions
        where1_X = [[i for i in range(n) if reduced_state[l, i]] for l in range(2*N)]
        where1_Z = [[i-n for i in range(n, 2*n) if reduced_state[l, i]] for l in range(2*N)]
        
        # create matrix with i positions
        wherei = [[i for i in range(n) if reduced_state[l, i]&reduced_state[l, i+n]] for l in range(2*N)]
        
        # update state
        for l in range(2*N):
            minus_sign = 0
            i_exp = len(wherei[l])
            if conj: i_exp += 2*(i_exp%2)
            for q in range(n):
                # update state
                Xq_exp_list_alpha = [alpha[i,q] for i in where1_X[l]]
                Xq_exp_list_gamma = [gamma[i,q] for i in where1_Z[l]]
                state[l, qubits_list[q]] = sum(Xq_exp_list_alpha + Xq_exp_list_gamma)%2
                
                Zq_exp_list_beta = [beta[i,q] for i in where1_X[l]]
                Zq_exp_list_delta = [delta[i,q] for i in where1_Z[l]]
                state[l, qubits_list[q] + N] = sum(Zq_exp_list_beta + Zq_exp_list_delta)%2
                
                # update i exponent
                alpha_and_beta = np.array(Xq_exp_list_alpha, dtype=np.int64) & np.array(Zq_exp_list_beta, dtype=np.int64)
                gamma_and_delta = np.array(Xq_exp_list_gamma, dtype=np.int64) & np.array(Zq_exp_list_delta, dtype=np.int64)
                i_exp += sum(alpha_and_beta) + sum(gamma_and_delta)
                
                if len(where1_X[l])>0:
                    for betapos in where1_X[l]:
                        alpha_to_sum = typed.List.empty_list(types.int64)
                        [alpha_to_sum.append(alpha[pos, q]) for pos in where1_X[l] if pos > betapos]                    
                        gamma_to_sum = typed.List.empty_list(types.int64)
                        if len(where1_Z[l])>0:
                            [gamma_to_sum.append(gamma[pos, q]) for pos in where1_Z[l] if pos >= betapos]
                            
                        minus_sign ^= beta[betapos, q] & (sum(alpha_to_sum) + sum(gamma_to_sum))%2
                        
                if len(where1_Z[l])>0:
                    for deltapos in where1_Z[l]:
                        gamma_to_sum = typed.List.empty_list(types.int64)
                        [gamma_to_sum.append(gamma[pos, q]) for pos in where1_Z[l] if pos > deltapos]
                        alpha_to_sum = typed.List.empty_list(types.int64)
                        if len(where1_X[l])>0:
                            [alpha_to_sum.append(alpha[pos, q]) for pos in where1_X[l] if pos > deltapos]

                        minus_sign ^= delta[deltapos, q] & (sum(alpha_to_sum) + sum(gamma_to_sum))%2    
                        
            # create rs_list to sum        
            r_list = [r[i] for i in where1_X[l]]
            s_list = [s[i] for i in where1_Z[l]]
            # update minus sign without considering i factors
            minus_sign ^= (sum(r_list) + sum(s_list))%2
            
            # reabsorb i in all the stabilizers that have XZ
            X_after = np.array([state[l, qubit] for qubit in qubits_list], dtype=np.int64)
            Z_after = np.array([state[l, qubit+N] for qubit in qubits_list], dtype=np.int64)
            if conj: i_exp += 2*(i_exp%2)
            i_exp -= sum(X_after & Z_after)
            
            if i_exp%4 == 2:
                minus_sign ^= 1
            
            state[l, 2*N] ^= minus_sign
    

    def _Unew(self, qubits_list, matrices, vectors, conj):
        
        # OLD NON STATIC METHOD
        n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
        if n > self.N:
            raise RuntimeError("Number of given qubits exceeds N")
        
        alpha = matrices[0]
        beta = matrices[1]
        gamma = matrices[2]
        delta = matrices[3]
        
        r = vectors[0]
        s = vectors[1]
        
        # conversion_dict = dict(enumerate(qubits_list))
        
        # reduce and REMAP state
        reduced_state = np.empty((2*self.N, 2*n+1), dtype = np.int64)
        for order_i, i in enumerate(qubits_list):
            reduced_state[:, order_i] = self.state[:, i]
            reduced_state[:, order_i + n] = self.state[:, i + self.N]
        reduced_state[:, 2*n] = self.state[:, 2*self.N]
        
        
        # create matrix with ones positions
        where1_X = [[i for i in range(n) if reduced_state[l, i]] for l in range(2*self.N)]
        where1_Z = [[i-n for i in range(n, 2*n) if reduced_state[l, i]] for l in range(2*self.N)]
        
        # create matrix with i positions
        wherei = [[i for i in range(n) if reduced_state[l, i]&reduced_state[l, i+n]] for l in range(2*self.N)]
        
        # update state
        for l in range(2*self.N):
            minus_sign = 0
            i_exp = len(wherei[l])
            if conj: i_exp += 2*(i_exp%2)
            for q in range(n):
                # update state
                Xq_exp_list_alpha = [alpha[i,q] for i in where1_X[l]]
                Xq_exp_list_gamma = [gamma[i,q] for i in where1_Z[l]]
                self.state[l, qubits_list[q]] = sum(Xq_exp_list_alpha + Xq_exp_list_gamma)%2
                
                Zq_exp_list_beta = [beta[i,q] for i in where1_X[l]]
                Zq_exp_list_delta = [delta[i,q] for i in where1_Z[l]]
                self.state[l, qubits_list[q] + self.N] = sum(Zq_exp_list_beta + Zq_exp_list_delta)%2
                
                # update i exponent
                alpha_and_beta = np.array(Xq_exp_list_alpha, dtype=int) & np.array(Zq_exp_list_beta, dtype=int)
                gamma_and_delta = np.array(Xq_exp_list_gamma, dtype=int) & np.array(Zq_exp_list_delta, dtype=int)
                i_exp += sum(alpha_and_beta) + sum(gamma_and_delta)
                
                if len(where1_X[l])>0:
                    for betapos in where1_X[l]:
                        alpha_to_sum = [alpha[pos,q] for pos in where1_X[l] if pos > betapos]
                        gamma_to_sum = []
                        if len(where1_Z[l])>0:
                            gamma_to_sum = [gamma[pos, q] for pos in where1_Z[l] if pos >= betapos]

                        minus_sign ^= beta[betapos, q] & sum(alpha_to_sum + gamma_to_sum)%2
                if len(where1_Z[l])>0:
                    for deltapos in where1_Z[l]:
                        gamma_to_sum = [gamma[pos, q] for pos in where1_Z[l] if pos > deltapos]
                        alpha_to_sum = []
                        if len(where1_X[l])>0:
                            alpha_to_sum = [alpha[pos, q] for pos in where1_X[l] if pos > deltapos]

                        minus_sign ^= delta[deltapos, q] & sum(alpha_to_sum + gamma_to_sum)%2    
            # create rs_list to sum        
            rs_list = [r[i] for i in where1_X[l]]
            rs_list += [s[i] for i in where1_Z[l]]
            # update minus sign without considering i factors
            minus_sign ^= sum(rs_list)%2
            
            # reabsorb i in all the stabilizers that have XZ
            X_after = np.array([self.state[l, qubit] for qubit in qubits_list], dtype=int)
            Z_after = np.array([self.state[l, qubit+self.N] for qubit in qubits_list], dtype=int)
            if conj: i_exp += 2*(i_exp%2)
            i_exp -= sum(X_after & Z_after)
            
            if i_exp%4 == 2:
                minus_sign ^= 1
            elif i_exp%4 == 1 or i_exp%4 == 3:
                raise RuntimeError("i factor is raised to the power of 1 or 3")
            
            self.state[l, 2*self.N] ^= minus_sign
    

    def _UdgNew(self, qubits_list, matrices, vectors, conj):
        
        n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
        if n > self.N:
            raise RuntimeError("Number of given qubits exceeds N")
            
        current_state = self.state.copy()
        self.reset_state()
        
        # first apply Udg with null r and s
        vectors_dagger = np.array([np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)], dtype=np.int64)
        matrices_dagger = self._findDagger(matrices)
        self._Unew_static(self.state, qubits_list, matrices_dagger, vectors_dagger, conj)
        
        # then apply U and save sign changes
        self._Unew_static(self.state, qubits_list, matrices, vectors, conj)
        minus_signs = self.state[:, 2*self.N]
        
        for order_i, i in enumerate(qubits_list):
            vectors_dagger[0][order_i] = minus_signs[i]
            vectors_dagger[1][order_i] = minus_signs[i+self.N]
        
        self.state = current_state
        self._Unew_static(self.state, qubits_list, matrices_dagger, vectors_dagger, conj)
        
            
    def _findDagger(self, matrices):
        # Need to invert the unitary
        # Find new matrices
        n = len(matrices[0])
        
        alpha = matrices[0]
        beta = matrices[1]
        gamma = matrices[2]
        delta = matrices[3]
        
        M = np.zeros([2*n, 2*n], dtype=np.int64)
        M[0:n, 0:n] = alpha
        M[0:n, n:2*n] = beta
        M[n:2*n, 0:n] = gamma
        M[n:2*n, n:2*n] = delta
        M_inv = modMatInv(M, 2)
        matrices_dagger = np.array([M_inv[0:n, 0:n], M_inv[0:n, n:2*n], M_inv[n:2*n, 0:n], M_inv[n:2*n, n:2*n]], dtype=np.int64)
      
        return matrices_dagger
    
 
    def _randClifford(self, qubits_list, params):
        """Applies clifford on qubits in qubits_list
        which clifford is given as three numbers in the params tuple:
            index_sp (int): specifies which symplectic matrix in Sp(2n)
            r (int): specifies minus sign of the transformed single qubit X gates
            s (int): specifies minus sign of the transformed single qubit S gates
        (thus r and s specifies which pauli we choose for each qubit)
        
        """
        n = len(qubits_list)
        qubits_list = np.array(qubits_list, dtype=np.int64)
        index_sp = params[0]
        r = np.array([(params[1]>>j) % 2 for j in range(n)])
        s = np.array([(params[2]>>j) % 2 for j in range(n)])
        gi = sp.symplectic_n3(n, index_sp).T        #transpose is needed since symplectic_n3 returns gi_T
        
        alpha = gi[0:2*n-1:2, 0:2*n-1:2].T
        beta = gi[1:2*n:2, 0:2*n-1:2].T
        gamma = gi[0:2*n-1:2, 1:2*n:2].T
        delta = gi[1:2*n:2, 1:2*n:2].T
        
        matrices = np.array([alpha, beta, gamma, delta], dtype=np.int64)
        vectors = np.array([r, s], dtype = np.int64)
        dagger = params[3]
        conj = params[4]
        if dagger:
            self._UdgNew(qubits_list, matrices, vectors, conj)
        else:
            self._Unew_static(self.state, qubits_list, matrices, vectors, conj)
            
        
    def _measure(self, a):
        if not isinstance(a, int):
            a = a[0]
        xa_positions = np.where(self.state[:, a] == 1)[0]     #save position of all stab and destab that do not commute with Z_a
        p = xa_positions[-1]
        if p >= self.N:      #if any stab does not commute with measure, state needs to be updated
            [self._rowsum(i, p) for i in xa_positions if (i!=p and i!=p-self.N)]      #update all stab and destab that anticommute with meas
            self.state[p-self.N] = self.state[p].copy()          #set (p-n)th row equal to p, moving the non commuting stab to destabs
            self.state[p] = np.zeros([2*self.N+1])          #set p-th row to all zeros except 1 in position N+a (the current measurement stabilizer)
            self.state[p, self.N+a] = 1
            rp = self.rng.randint(0,1)
            self.state[p, 2*self.N] = rp
            return rp
        else:
            extra_row = np.zeros([2*self.N+1], dtype=int)
            for i in xa_positions:
                extra_row = self._sum_rows(extra_row, self.state[i+self.N])
            r_extra = extra_row[2*self.N]
            return r_extra
        
                
    def CNOT(self, a, b):
        if a >= self.N or b>= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        cnot = Instruction('CNOT', [a, b])
        self._circuit.append(cnot)
    
    def H(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        h = Instruction('H', [a])
        self._circuit.append(h)
        
    def S(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        s = Instruction('S', [a])
        self._circuit.append(s)
    
    def Sdg(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        sdg = Instruction('Sdg', [a])
        self._circuit.append(sdg)
        
    def X(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        x = Instruction('X', [a])
        self._circuit.append(x)
    
    def Z(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        z = Instruction('Z', [a])
        self._circuit.append(z)
        
    def SWAP(self, a, b):
        if a >= self.N or b>= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        swap = Instruction('SWAP', [a, b])
        self._circuit.append(swap)
    
        
    def measure(self, qubits_list):
        if max(qubits_list) >= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        meas = Instruction('measure', qubits_list)
        self._circuit.append(meas)
        
    def randClifford(self, qubits_list, params = None):
        if isinstance(qubits_list, int):
            qubits_list = [qubits_list]
        if max(qubits_list) >= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        cliff = RandomClifford(qubits_list, params = params)
        self._circuit.append(cliff)
        
        
    def randEvolution(self, depth):
        if self.N % 2 != 0:
            raise RuntimeError("Cannot apply Brownian evolution with an odd number of qubits")
        qubits = [i for i in range(self.N)]
        for layer in range(depth):
            self.rng.shuffle(qubits)
            pairs = [qubits[2*i:2*(i+1)] for i in range(len(qubits)//2)]
            for pair in pairs:
                self.randClifford(pair)
                
    def randEvolutionBH(self, N_gates):
        qubits = [i for i in range(self.N)]
        for layer in range(N_gates):
            self.rng.shuffle(qubits)
            pair = [qubits[0], qubits[1]]
            self.randClifford(pair)
        
    
    def run(self, shots = 1):
        initial_state = self.state.copy()
        outcomes = []
        is_measured = False
        for i in range(shots):
            self.state = initial_state.copy()       #n^2 complexity
            for gate in self._circuit:
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
                    func_to_call(self, gate.qubits)
            
        if is_measured == True:  
            return counts
        
    def dot_outcome(self, outcome):
        # bit flip all qubits for which outcome is 1
        [self._X(i) for i in range(self.N) if outcome[i]]
        
        # perform dot_zero
        state = self.gauss_elim_stab(self.state[self.N:])
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
    
          
    def saveShadows(self, N_shadows, depth):
        """
        Extract classical shadows of the current state (make sure it has 
                                                        already run)
        """
        if self.N % 2 != 0:
            raise RuntimeError("Cannot use this classical shadows protocol with an odd number of qubits")
        saved_circ = self.circuit
        saved_state = self.state
        
        random_circuits = []
        outcomes = []
        
        for i in range(N_shadows):
            self.reset_circuit()
            self.randEvolution(depth)
            # save random unitary
            random_circuits.append(self._circuit)
            
            # run random evolution
            self.state = saved_state
            for gate in self._circuit:
                self._randClifford(gate.qubits, gate.params)
            
            # measure and save outcome
            measurements = [self._measure(a) for a in range(self.N)]
            outcomes.append(measurements)
        
        self.state = saved_state
        self._circuit = saved_circ
        
        return (random_circuits, outcomes)
        
    def saveShadowsGlobal(self, N_shadows):
        
        saved_circ = self.circuit
        saved_state = self.state.copy()
        
        random_circuits = []
        outcomes = []
        
        for i in range(N_shadows):
            self.reset_circuit()
            self.randClifford([a for a in range(self.N)])
            # save random unitary
            random_circuits.append(self._circuit)
            
            # run random evolution
            self.state = saved_state
            for gate in self._circuit:
                self._randClifford(gate.qubits, gate.params)
            
            # measure and save outcome
            measurements = [self._measure(a) for a in range(self.N)]
            outcomes.append(measurements)
        
        self.state = saved_state
        self._circuit = saved_circ
        
        return [random_circuits, outcomes]
    
    def saveShadowsMixed(self, N_shadows, depth):
        if self.N % 2 != 0:
            raise RuntimeError("Cannot use this classical shadows protocol with an odd number of qubits")
        
        saved_circ = self.circuit
        saved_state = self.state.copy()
        
        random_circuits = []
        outcomes = []
        
        for i in range(N_shadows):
            self.reset_circuit()          
            # layer of single-site measurements
            [self.randClifford(a) for a in range(self.N)]
            # random evolution
            self.randEvolution(depth)
            # save random unitary
            random_circuits.append(self._circuit)
            
            # run random evolution
            self.state = saved_state
            for gate in self._circuit:
                self._randClifford(gate.qubits, gate.params)
            
            # measure and save outcome
            measurements = [self._measure(a) for a in range(self.N)]
            outcomes.append(measurements)
        
        self.state = saved_state
        self._circuit = saved_circ
        
        return [random_circuits, outcomes]
    
    def saveShadowsBH(self, N_shadows, N_gates):
        
        saved_circ = self.circuit
        saved_state = self.state.copy()
        
        random_circuits = []
        outcomes = []
        
        for i in range(N_shadows):
            self.reset_circuit()          
            # layer of single-site measurements
            [self.randClifford(a) for a in range(self.N)]
            # random evolution with single gates
            self.randEvolutionBH(N_gates)
            # save random unitary
            random_circuits.append(self._circuit)
            
            # run random evolution
            self.state = saved_state
            for gate in self._circuit:
                self._randClifford(gate.qubits, gate.params)
            
            # measure and save outcome
            measurements = [self._measure(a) for a in range(self.N)]
            outcomes.append(measurements)
        
        self.state = saved_state
        self._circuit = saved_circ
        
        return [random_circuits, outcomes]
        
    
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
        
        
from numpy import matrix
from numpy import linalg

def modMatInv(A,p):       # Finds the inverse of matrix A mod p
  n=len(A)
  A=matrix(A.copy())
  adj=np.zeros(shape=(n,n), dtype=int)
  for i in range(0,n):
    for j in range(0,n):
      adj[i][j]=((-1)**(i+j)*int(round(linalg.det(minor(A,j,i)))))%p
  return (modInv(int(round(linalg.det(A))),p)*adj)%p


def modInv(a,p):          # Finds the inverse of a mod p, if it exists
  for i in range(1,p):
    if (i*a)%p==1:
      return i
  raise ValueError(str(a)+" has no inverse mod "+str(p))


def minor(A,i,j):    # Return matrix A with the ith row and jth column deleted
  A=np.array(A)
  minor=np.zeros(shape=(len(A)-1,len(A)-1), dtype=int)
  p=0
  for s in range(0,len(minor)):
    if p==i:
      p=p+1
    q=0
    for t in range(0,len(minor)):
      if q==j:
        q=q+1
      minor[s][t]=A[p][q]
      q=q+1
    p=p+1
  return minor
    

