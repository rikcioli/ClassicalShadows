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


class StabilizerCircuit(StabilizerState):
    
    def __init__(self, N_qubits, state = None, circuit = []):
        super().__init__(N_qubits, state)
        self._circuit = circuit
    
    @property
    def circuit(self):
        return self._circuit
    
    @circuit.setter
    def circuit(self, stabilizer_circuit):
        self._circuit = copy.deepcopy(stabilizer_circuit)
        
    def reset_state(self):
        self._state = np.eye(2*self.N, 2*self.N+1, 0, dtype=int)
    
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
        
    def _X(self, a):
        vec_r = self.state[:, 2*self.N]
        zcol_a = self.state[:, self.N+a]
        
        self.state[:, 2*self.N] = vec_r^zcol_a
    

    def _Unew(self, qubits_list, matrices, vectors):
        
        n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
        if n > self.N:
            raise RuntimeError("Number of given qubits exceeds N")
        
        alpha = matrices[0]
        beta = matrices[1]
        gamma = matrices[2]
        delta = matrices[3]
        
        r = vectors[0]
        s = vectors[1]
        
        conversion_dict = dict(enumerate(qubits_list))
        
        # reduce state if n < N
        if n < self.N:
            reduced_state = np.zeros([2*self.N, 2*n+1], dtype = int)
            for order_i, i in enumerate(qubits_list):
                reduced_state[:, order_i] ^= self.state[:, i]
                reduced_state[:, order_i + n] ^= self.state[:, i + self.N]
            reduced_state[:, 2*n] ^= self.state[:, 2*self.N]
        else:
            reduced_state = self.state.copy()
        
        # create matrix with ones positions
        where1_X = [[i for i in range(n) if reduced_state[l, i]] for l in range(2*self.N)]
        where1_Z = [[i-n for i in range(n, 2*n) if reduced_state[l, i]] for l in range(2*self.N)]
        
        # create matrix with i positions
        wherei = [[i for i in range(n) if reduced_state[l, i]&reduced_state[l, i+n]] for l in range(2*self.N)]
        
        # update state
        for l in range(2*self.N):
            minus_sign = 0
            i_exp = len(wherei[l])
            for q in range(n):
                # update state
                Xq_exp_list_alpha = [alpha[i,q] for i in where1_X[l]]
                Xq_exp_list_gamma = [gamma[i,q] for i in where1_Z[l]]
                self.state[l, conversion_dict[q]] = sum(Xq_exp_list_alpha + Xq_exp_list_gamma)%2
                
                Zq_exp_list_beta = [beta[i,q] for i in where1_X[l]]
                Zq_exp_list_delta = [delta[i,q] for i in where1_Z[l]]
                self.state[l, conversion_dict[q] + self.N] = sum(Zq_exp_list_beta + Zq_exp_list_delta)%2
                
                # update i exponent
                alpha_and_beta = np.array(Xq_exp_list_alpha, dtype=int) & np.array(Zq_exp_list_beta, dtype=int)
                gamma_and_delta = np.array(Xq_exp_list_gamma, dtype=int) & np.array(Zq_exp_list_delta, dtype=int)
                i_exp += sum(alpha_and_beta) + sum(gamma_and_delta)
                
                if len(where1_X[l])>0:
                    for betapos in where1_X[l]:
                        alpha_to_sum = [alpha[pos,q] for pos in where1_X[l] if pos > betapos]
                        if len(where1_Z[l])>0:
                            gamma_to_sum = [gamma[pos, q] for pos in where1_Z[l] if pos >= betapos]
                        else:
                            gamma_to_sum = []
                        minus_sign ^= beta[betapos, q] & sum(alpha_to_sum + gamma_to_sum)%2
                if len(where1_Z[l])>0:
                    for deltapos in where1_Z[l]:
                        gamma_to_sum = [gamma[pos, q] for pos in where1_Z[l] if pos > deltapos]
                        if len(where1_X[l])>0:
                            alpha_to_sum = [alpha[pos, q] for pos in where1_X[l] if pos > deltapos]
                        else:
                            alpha_to_sum = []
                        minus_sign ^= delta[deltapos, q] & sum(alpha_to_sum + gamma_to_sum)%2    
            # create rs_list to sum        
            rs_list = [r[i] for i in where1_X[l]]
            rs_list += [s[i] for i in where1_Z[l]]
            # update minus sign without considering i factors
            minus_sign ^= sum(rs_list)%2
            
            # reabsorb i in all the stabilizers that have XZ
            X_after = np.array([self.state[l, qubit] for qubit in qubits_list], dtype=int)
            Z_after = np.array([self.state[l, qubit+self.N] for qubit in qubits_list], dtype=int)
            i_exp -= sum(X_after & Z_after)
            
            if i_exp%4 == 2:
                minus_sign ^= 1
            elif i_exp%4 == 1 or i_exp%4 == 3:
                raise RuntimeError("i factor is raised to the power of 1 or 3")
            
            self.state[l, 2*self.N] ^= minus_sign
        
        
    # def _U(self, qubits_list, matrices, vectors):
    #     """Apply most generic clifford unitary as specified by matrices alpha, 
    #     beta, gamma, delta and vectors r, s.

    #     """
    #     n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
    #     if n > self.N:
    #         raise RuntimeError("Number of given qubits exceeds N")
        
    #     alpha = matrices[0]
    #     beta = matrices[1]
    #     gamma = matrices[2]
    #     delta = matrices[3]
        
    #     r = vectors[0]
    #     s = vectors[1]
        
    #     x = np.array([self.state[:, i] for i in qubits_list])
    #     z = np.array([self.state[:, i+self.N] for i in qubits_list])
        
    #     sign_change = np.zeros(2*self.N, dtype = int)
    #     proj = np.ones(n, dtype=int)
        
    #     # update minus sign
    #     beta_gammaT = np.dot(beta, gamma.T)%2
    #     beta_alphaT = np.dot(beta, alpha.T)%2
    #     delta_gammaT = np.dot(delta, gamma.T)%2
    #     delta_alphaT = np.dot(delta, alpha.T)%2
        
    #     # we introduce two variables: i (which initial qubit) and order_i 
    #     # order_i means the position of qubit i in the subspace on which the clifford is acting
    #     # e.g. we act con qubit i = 3 and i = 7, then i = 3 is the first (order_i = 0)
    #     # and i = 7 is the second (order_i = 1)
    #     # this is useful as we need to map the alpha, beta (etc) n by n matrices to the actual qubits
        
    #     for order_i, i in enumerate(qubits_list):
            
    #         sign_change ^= (r[order_i]&x[order_i] ^ s[order_i]&z[order_i])
            
    #         B1 = np.array([np.dot(beta_gammaT[order_i], proj & z[:, l]) %2 for l in range(2*self.N)])
    #         proj[order_i] = 0
    #         B2 = np.array([np.dot(beta_alphaT[order_i], proj & x[:, l])%2 for l in range(2*self.N)])
    #         sign_change ^= x[order_i]&(B1^B2)
            
    #         D1 = np.array([np.dot(delta_gammaT[order_i], proj & z[:, l])%2 for l in range(2*self.N)])
    #         D2 = np.array([np.dot(delta_alphaT[order_i], proj & x[:, l])%2 for l in range(2*self.N)])            
    #         sign_change ^= z[order_i]&(D1^D2)
        
    #     self.state[:, 2*self.N] ^= sign_change
        
    #     # update state
    #     for order_q, q in enumerate(qubits_list):
            
    #         x_exponent = np.array([(np.dot(x[:, l], alpha[:, order_q]) + np.dot(z[:, l], gamma[:, order_q]))%2 for l in range(2*self.N)])
    #         z_exponent = np.array([(np.dot(x[:, l], beta[:, order_q]) + np.dot(z[:, l], delta[:, order_q]))%2 for l in range(2*self.N)])
    #         self.state[:, q] = x_exponent
    #         self.state[:, q+self.N] = z_exponent
    

    def _UdgNew(self, qubits_list, matrices, vectors):
        
        n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
        if n > self.N:
            raise RuntimeError("Number of given qubits exceeds N")
            
        current_state = self.state.copy()
        self.reset_state()
        
        # first apply Udg with null r and s
        vectors_dagger = [np.zeros(n, dtype=int), np.zeros(n, dtype=int)]
        matrices_dagger = self._findDagger(matrices)
        self._Unew(qubits_list, matrices_dagger, vectors_dagger)
        
        # then apply U and save sign changes
        self._Unew(qubits_list, matrices, vectors)
        minus_signs = self.state[:, 2*self.N]
        
        for order_i, i in enumerate(qubits_list):
            vectors_dagger[0][order_i] = minus_signs[i]
            vectors_dagger[1][order_i] = minus_signs[i+self.N]
        
        self.state = current_state
        self._Unew(qubits_list, matrices_dagger, vectors_dagger)
        
    
    # def _Udg(self, qubits_list, matrices, vectors, matrices_dagger):
    #     """Apply inverse of the clifford unitary specified by matrices alpha, 
    #     beta, gamma, delta and vectors r, s.
    #     """
        
    #     n = len(qubits_list)  # dim of the subspace on which U is acting non trivially
    #     if n > self.N:
    #         raise RuntimeError("Number of given qubits exceeds N")
            
    #     # save current minus sign vector
    #     minus_signs = self.state[:, 2*self.N].copy()
        
    #     # first apply Udg with null r and s
    #     vectors_dagger = [np.array([0, 0]), np.array([0, 0])]
    #     self._U(qubits_list, matrices_dagger, vectors_dagger)
        
    #     # now apply U, it should result in an identity matrix up to sign
    #     self._U(qubits_list, matrices, vectors)
        
    #     # compare two sign vectors
    #     new_signs = self.state[:, 2*self.N].copy()
    #     diff_vec = (new_signs - minus_signs)%2
        
    #     # if there is a difference, we need to change vectors_dagger
    #     if 1 in diff_vec:
    #         # construct a matrix in the following way:
    #         # first n columns are the entries of the X matrix of the state 
    #         # for the qubits in qubits_list
    #         # second n columns are the entries of the Z matrix of the state 
    #         # for the qubits in qubits_list
    #         # last column is diff_vec
    #         """
    #         reduced_state_diff_vec = np.zeros([2*self.N, 2*n+1], dtype = int)
    #         for order_i, i in enumerate(qubits_list):
    #             reduced_state_diff_vec[:, order_i] ^= self.state[:, i]
    #             reduced_state_diff_vec[:, order_i + n] ^= self.state[:, i + self.N]
    #         reduced_state_diff_vec[:, 2*n] = diff_vec
            
    #         # find correct value of r and s by inverting submatrix of x and z
    #         max_rank_submatrix = extractMaxRank(reduced_state_diff_vec)
    #         inverse_submatrix = modMatInv(max_rank_submatrix[:, :2*n], 2)
    #         rs_vec = np.dot(inverse_submatrix, max_rank_submatrix[:, 2*n])%2
    #         # update r and s
    #         vectors_dagger = [rs_vec[:n], rs_vec[n:2*n]]
    #         # restore old values of the signs
    #         self.state[:, 2*self.N] = minus_signs.copy()
    #         """
            
    #         # inverse_state = modMatInv(self.state[:, :2*self.N], 2)
    #         # rs_vec = np.dot(inverse_state, diff_vec)%2
    #         # r_vec = np.array([rs_vec[qubit] for qubit in qubits_list])
    #         # s_vec = np.array([rs_vec[qubit + self.N] for qubit in qubits_list])
    #         # vectors_dagger = [r_vec, s_vec]
    #         # self.state[:, 2*self.N] = minus_signs.copy()
        
    #     self._U(qubits_list, matrices_dagger, vectors_dagger)    
            
    def _findDagger(self, matrices):
        # Need to invert the unitary
        # Find new matrices
        n = len(matrices[0])
        
        alpha = matrices[0]
        beta = matrices[1]
        gamma = matrices[2]
        delta = matrices[3]
        
        M = np.zeros([2*n, 2*n], dtype=int)
        M[0:n, 0:n] = alpha
        M[0:n, n:2*n] = beta
        M[n:2*n, 0:n] = gamma
        M[n:2*n, n:2*n] = delta
        M_inv = modMatInv(M, 2)
        matrices_dagger = [M_inv[0:n, 0:n], M_inv[0:n, n:2*n], M_inv[n:2*n, 0:n], M_inv[n:2*n, n:2*n]]
      
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
        index_sp = params[0]
        r = np.array([(params[1]>>j) % 2 for j in range(n)])
        s = np.array([(params[2]>>j) % 2 for j in range(n)])
        gi = sp.symplectic_n3(n, index_sp)
        
        alpha = gi[0:2*n-1:2, 0:2*n-1:2].T
        beta = gi[1:2*n:2, 0:2*n-1:2].T
        gamma = gi[0:2*n-1:2, 1:2*n:2].T
        delta = gi[1:2*n:2, 1:2*n:2].T
        
        matrices = [alpha, beta, gamma, delta]
        vectors = [r, s]
        
        dagger = params[3]
        # if dagger:
        #     metric = sp.metric(n)
        #     gi_inverse = np.linalg.multi_dot([metric, gi.T, metric])%2
        #     matrices_dagger = [gi_inverse[0:3:2, 0:3:2].T, gi_inverse[1:4:2, 0:3:2].T, gi_inverse[0:3:2, 1:4:2].T, gi_inverse[1:4:2, 1:4:2].T]
        #     self._UdgNew(qubits_list, matrices, vectors, matrices_dagger)
        if dagger:
            self._UdgNew(qubits_list, matrices, vectors)
        else:
            self._Unew(qubits_list, matrices, vectors)
            
        
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
        if a >= self.N or b>= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        cnot = Instruction('CNOT', [(a, b)])
        self.circuit.append(cnot)
    
    def H(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        h = Instruction('H', [a])
        self.circuit.append(h)
        
    def S(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        s = Instruction('S', [a])
        self.circuit.append(s)
        
    def X(self, a):
        if a >= self.N:
            raise RuntimeError("target qubit exceeds N_qubits")
        x = Instruction('X', [a])
        self.circuit.append(x)
        
    def measure(self, qubits_list):
        if max(qubits_list) >= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        meas = Instruction('measure', qubits_list)
        self.circuit.append(meas)
        
    def randClifford(self, qubits_list):
        if max(qubits_list) >= self.N:
            raise RuntimeError("target qubits exceed N_qubits")
        cliff = RandomClifford('random_clifford', qubits_list)
        self.circuit.append(cliff)
        
    def randEvolution(self, depth):
        qubits = [i for i in range(self.N)]
        for layer in range(depth):
            random.shuffle(qubits)
            pairs = [qubits[2*i:2*(i+1)] for i in range(len(qubits)//2)]
            for pair in pairs:
                self.randClifford(pair)
    
    def run(self, shots = 1):
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
        
          
    def saveShadows(self, N_shadows, depth):
        """
        Extract classical shadows of the current state (make sure it has 
                                                        already run)
        """
        saved_circ = self.circuit.copy()
        saved_state = self.state.copy()
        
        random_circuits = []
        outcomes = []
        
        for i in range(N_shadows):
            self.reset_circuit()
            self.randEvolution(depth)
            # save random unitary
            random_circuits.append(self.circuit)
            
            # run random evolution
            self.state = saved_state
            for gate in self.circuit:
                self._randClifford(gate.qubits, gate.params)
            
            # measure and save outcome
            measurements = [self._measure(a) for a in range(self.N)]
            outcomes.append(measurements)
        
        self.state = saved_state
        self.circuit = saved_circ
        
        return [random_circuits, outcomes]
        
    def saveShadowsGlobal(self, N_shadows):
        
        saved_circ = self.circuit.copy()
        saved_state = self.state.copy()
        
        random_circuits = []
        outcomes = []
        
        for i in range(N_shadows):
            self.reset_circuit()
            self.randClifford([a for a in range(self.N)])
            # save random unitary
            random_circuits.append(self.circuit)
            
            # run random evolution
            self.state = saved_state
            for gate in self.circuit:
                self._randClifford(gate.qubits, gate.params)
            
            # measure and save outcome
            measurements = [self._measure(a) for a in range(self.N)]
            outcomes.append(measurements)
        
        self.state = saved_state
        self.circuit = saved_circ
        
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
    

