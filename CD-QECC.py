#!/usr/bin/env python
# coding: utf-8

# In[17]:


from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize 
import random
from qiskit.visualization import plot_histogram


QC_SIM = Aer.get_backend('qasm_simulator')

class qecc_agent:

    def __init__(self, id, k, n, perr=[0.2, 0, 0], vqc_trials=20, inp_trials=3, err_trials=5, sim_trials=1000):
        self.agent_id = id
        self.k = k  # no. of qubits for info
        self.n = n  # no. of qubits for encoding info
        self.s = self.n - self.k  # no. of qubits for syndrome ancilla
        self.perr = perr  # probability of error for each type of error
        self.vqc_trials = vqc_trials  # no. of times the same ansatz is evaluated for different parameters
        self.inp_trials = inp_trials  # no. of times the same ansatz+parameters is evaluated different input states
        self.err_trials = err_trials  # no. of times the same ansatz+parameters+input is evaluated with different errors
        self.sim_trials = sim_trials  # no. of times the same ansatz+parameters+input+errors is evaluated by statevector simulator to get measurement statistics

        self.qec_ansatz = QuantumCircuit(self.n + self.s, self.k)
        self.ansatz_init()

        self.fitness = 0

    def gen_inp_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        ax = np.random.rand() * 2 * np.pi
        qc.rx(ax, range(self.k))
        ay = np.random.rand() * 2 * np.pi
        qc.ry(ay, range(self.k))
        qc.barrier()
        return qc

    def gen_enc_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        for i in range(self.s):
            qc.cx(self.k - 1, self.k + i)
        qc.barrier()
        return qc

    def gen_err_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        for i in range(self.n):
            px = np.random.rand()
            py = np.random.rand()
            pz = np.random.rand()
            if px < self.perr[0]:
                qc.x(i)
            if py < self.perr[1]:
                qc.y(i)
            if pz < self.perr[2]:
                qc.z(i)
        qc.barrier()
        return qc

    def ansatz_init(self):
        params = []
        for i in range(self.k + self.n):
            params.append(Parameter(f'p{i}'))
            self.qec_ansatz.rx(params[i], i)

        
        num_cnot_gates = random.randint(0, self.k)  
        for _ in range(num_cnot_gates):
            control_qubit = random.randint(0, self.n + self.s - 1)
            target_qubit = random.randint(0, self.n + self.s - 1)
            self.qec_ansatz.cx(control_qubit, target_qubit)
        return

    def ansatz_mutate(self, mut_prob):
        
        return

    def gen_qec_circ(self, ansatz_params):
        qc_params = self.qec_ansatz.parameters
        qec_circ = copy.deepcopy(self.qec_ansatz)
        
        for i in range(len(qc_params)):
            qec_circ = qec_circ.assign_parameters({qc_params[i]: ansatz_params[i]})
        
        return qec_circ


    def gen_dec_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        for i in range(0, self.n - self.k):
            qc.cx(self.k - 1, self.k + i)
        qc.barrier()
        return qc

    def run_param(self, ansatz_params):
        qec_circ = self.gen_qec_circ(ansatz_params)
        enc_circ = self.gen_enc_circ()
        dec_circ = self.gen_dec_circ()

        penalties = []

        for ti in range(self.inp_trials):
            qc = self.gen_inp_circ()
            qc_noerr = copy.deepcopy(qc)
            qc_noerr.measure(range(self.k), range(self.k))

            res_noerr = execute(qc_noerr, QC_SIM, shots=self.sim_trials).result()
            m_noerr = res_noerr.get_counts(qc_noerr)

            qc_enc = qc.compose(enc_circ)

            for ei in range(self.err_trials):
                qc_enc_ei = copy.deepcopy(qc_enc)
                err_circ = self.gen_err_circ()
                qc_enc_err = qc_enc_ei.compose(err_circ)
                qc_qec = qc_enc_err.compose(qec_circ)
                qc_dec = qc_qec.compose(dec_circ)
                qc_dec.measure(range(self.k), range(self.k))
                #print(err_circ.draw())
                res_qec = execute(qc_dec, QC_SIM, shots=self.sim_trials).result()
                m_qec = res_qec.get_counts(qc_dec)
                #print(m_noerr, m_qec)
    
                penalty = sum(
                    abs(m_noerr.get(key, 0) - m_qec.get(key, 0)) / self.sim_trials for key in set(m_noerr) | set(m_qec))
                penalties.append(penalty)

        param_penalty = sum(penalties) / (self.inp_trials * self.err_trials)
        return param_penalty

    def eval_agent(self):
        no_params = len(self.qec_ansatz.parameters)
        init_params = np.random.uniform(0, 2 * np.pi, no_params)
        max_iter = self.vqc_trials
        res = minimize(self.run_param, init_params, method='nelder-mead', options={'xatol': 1e-8, 'disp': False, 'maxiter': max_iter})
        print("\tAgent ID", self.agent_id, "Best score:", res.fun, "Best params:", res.x)
        self.fitness = res.fun
        return

no_gen = 3      # Number of generations
pop_sz = 2      # Population size
pop = {}        # Population of agents as objects of the class qecc_agent
agt_id = 0      # Agent ID
max_fit = 0.018 # Maximum fitness (penalty) score for an agent to be selected for next generation
mut_prob = 0.1  # Probability of mutation of an agent's ansatz

for gi in range(no_gen):
    print("Generation", gi)
    # Create new individuals in population
    for _ in range(len(pop),pop_sz):
        pop[agt_id] = qecc_agent(agt_id, 1, 3)
        agt_id += 1
        # print(pop[i].qec_ansatz.draw())
    # Evaluate
    for ai in pop.keys():  
        pop[ai].eval_agent()
    # Select                    # TBD: Change to Elitist Selection for pop_sz/2 agents
    pop_nxt_gen = {}
    for ai in pop.keys(): 
        if pop[ai].fitness < max_fit:   
            pop_nxt_gen[ai] = pop[ai]
    # Mutate
    for ai in pop_nxt_gen.keys(): 
        pop_nxt_gen[ai].ansatz_mutate(mut_prob)
    # Update population
    pop = pop_nxt_gen
   
# Print best individual over all generations
   


# In[ ]:




