"""
Concept Discovery of Quantum Error Correction Codes
"""

from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
import numpy as np
import copy
from scipy.optimize import minimize 
QC_SIM = Aer.get_backend('qasm_simulator')

class qecc_agent:

    def __init__(self, id, k, n, perr = [0.2,0,0], vqc_trials = 20, inp_trials = 3, err_trials = 5, sim_trials = 1000):
        self.agent_id = id
        self.k = k                      # no. of qubits for info
        self.n = n                      # no. of qubits for encoding info
        self.s = self.n-self.k          # no. of qubits for syndrome ancilla
        self.perr = perr                # probability of error for each type of error
                                        # no. of ansatz evaluated is same as population size x no. of generations
        self.vqc_trials = vqc_trials    # no. of times the same ansatz is evaluated for different parameters
        self.inp_trials = inp_trials    # no. of times the same ansatz+parameters is evaluated different input states
        self.err_trials = err_trials    # no. of times the same ansatz+parameters+input is evaluated with different errors
        self.sim_trials = sim_trials    # no. of times the same ansatz+parameters+input+errors is evaluated by statevector simulator to get measurement statistics
        
        self.qec_ansatz = QuantumCircuit(self.n + self.s, self.k)
        self.ansatz_init()

    """
    Generate the input circuit
    """
    def gen_inp_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        ax = np.random.rand() * 2 * np.pi
        qc.rx(ax, range(self.k))
        ay = np.random.rand() * 2 * np.pi    
        qc.ry(ay, range(self.k))
        qc.barrier()
        return qc
    
    """
    Generate the encoding circuit
    """
    def gen_enc_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        for i in range(self.s):
            qc.cx(self.k - 1, self.k + i)
        qc.barrier()
        return qc
    
    """
    Generate the error circuit
    """
    def gen_err_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        for i in range(self.n):     # TBD: Add error also on syndrome qubits?
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
    
    """
    Initialize agent's ansatz
    """
    def ansatz_init(self):
        p0 = Parameter('p0')
        self.qec_ansatz.rx(p0,3)
        p1 = Parameter('p1')
        self.qec_ansatz.rx(p1,4)
        return

    """
    Mutate agent's ansatz
    """
    def ansatz_mutate(self):
        return 

    """
    Generate the QEC circuit    
    """
    def gen_qec_circ(self, ansatz_params):
        qc = QuantumCircuit(self.n + self.s, self.k)
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.barrier()
        qc.x(4)
        qc.mcx([3,4],0)     # Only s1 fired, Bit-flip on q0
        qc.x(4)
        qc.mcx([3,4],1)     # Both s1, s2 fired, Bit-flip on q1
        qc.x(3)
        qc.mcx([3,4],2)     # Only s2 fired, Bit-flip on q2
        qc.x(3)
        qc.barrier()

        # TBD: Choose a random qubit between q1:(0,n), and a random target between q2:(0,s) and apply a CNOT from q1 to n+q2. Repeat this for 4 times
        qc_params = self.qec_ansatz.parameters
        qec_circ = self.qec_ansatz
        for i in range(len(qc_params)):
            qec_circ = qec_circ.assign_parameters({qc_params[i]: ansatz_params[i]})
        # print(qec_circ.draw())
        
        qc = qc.compose(qec_circ)
        return qc
    
    """
    Generate the decoding circuit
    """
    def gen_dec_circ(self):
        qc = QuantumCircuit(self.n + self.s, self.k)
        for i in range(0, self.n - self.k):
            qc.cx(self.k - 1, self.k + i)
        qc.barrier()
        return qc
    
    """
    Evaluate the agent's ansatz for a given set of parameters
    """
    def run_param(self, ansatz_params):

        qec_circ  = self.gen_qec_circ(ansatz_params)
        enc_circ  = self.gen_enc_circ()
        dec_circ  = self.gen_dec_circ()

        penalties = []

        for ti in range(self.inp_trials): 
            qc = self.gen_inp_circ()
            qc_noerr = copy.deepcopy(qc)
            qc_noerr.measure(range(self.k), range(self.k))
            # print(qc_noerr.draw())
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
                # print(err_circ.draw())
                res_qec = execute(qc_dec, QC_SIM, shots=self.sim_trials).result()
                m_qec = res_qec.get_counts(qc_dec)
                # print(m_noerr, m_qec)
                penalty = sum(abs(m_noerr.get(key, 0) - m_qec.get(key, 0)) / self.sim_trials for key in set(m_noerr) | set(m_qec))
                penalties.append(penalty)
                # print("Input:",ti,"Error",ei, "Penalty", penalty)

        param_penalty = sum(penalties) / (self.inp_trials * self.err_trials)
        return param_penalty
    
    """
    Optimize the agent's ansatz parameters
    """
    def eval_agent(self):
        score = 0
        no_params = len(self.qec_ansatz.parameters)
        best_score = 1000
        best_params = []

        ''' Using Random Search '''
        # for vi in range(self.vqc_trials):
        #     params = []
        #     for i in range(no_params):
        #         params.append(np.random.rand() * 2 * np.pi)
        #     score = self.run_param(params)
        #     print("\t\tAgent ID:",self.agent_id,"Param",vi,params,"Score",score)
        #     if score < best_score:
        #         best_score = score
        #         best_params = params
        # print("\tAgent ID",self.agent_id,"Best score:", best_score, "Best params:", best_params)

        ''' Using SciPy optimizer '''
        init_params = np.random.uniform(0, 2*np.pi, no_params)
        max_iter = self.vqc_trials
        res = minimize(self.run_param, init_params, method='nelder-mead', options={'xatol': 1e-8, 'disp': False, 'maxiter': max_iter})
        print("\tAgent ID",self.agent_id,"Best score:", res.fun, "Best params:", res.x)
        
        return best_score


"""
Genetic Algorithm that uses a population of qecc_agents for evolving the best ansatz
"""

pop_sz = 1
no_gen = 1
mut_prob = 0.1
pop = {}

for gi in range(no_gen):
    print("Generation", gi)
    for ai in range(pop_sz):
        pop[ai] = qecc_agent(ai, 1, 3)
        # print(pop[i].qec_ansatz.draw())
        score = pop[ai].eval_agent()