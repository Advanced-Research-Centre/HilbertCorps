#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

# Invoke a primitive. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
#result = Sampler("ibmq_qasm_simulator").run(circuits).result()


# In[2]:


from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import random
import copy
from scipy.linalg import polar
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt


k = 1
n = 3
s = n - k
trials = 15
shots = 100

perr_x = 0.4
perr_y = 0
perr_z = 0

simulator = Aer.get_backend('qasm_simulator')

def prep_state(qc, ax, ay):
    qc.rx(ax, range(k))
    qc.ry(ay, range(k))
    return qc

def add_enc_circ(ax, ay):
    qc = QuantumCircuit(n + s, k)
    qc.rx(ax, range(k))
    qc.ry(ay, range(k))
    qc.barrier()
    for i in range(0, n - k):
        qc.cx(k - 1, k + i)
    qc.barrier()
    return qc

def add_err_circ(qc):
    for i in range(n):
        px = np.random.rand()
        py = np.random.rand()
        pz = np.random.rand()
        if px < perr_x:
            qc.x(i)
        if py < perr_y:
            qc.y(i)
        if pz < perr_z:
            qc.z(i)
    qc.barrier()
    return qc

def add_dec_circ(qc):
    for i in range(0, n - k):
        qc.cx(k - 1, k + i)
    qc.barrier()
    return qc

def create_initial_population(population_size):
    population = []
    for _ in range(population_size):
        syndrome_circuit = QuantumCircuit(n + s, k)
        
        for _ in range(s):
            control_qubit = random.randint(0, n - 1)
            target_qubit = random.randint(n, n + s - 1)
            syndrome_circuit.cx(control_qubit, target_qubit)
        population.append(syndrome_circuit)
    return population

def evaluate_fitness(syndrome_circuit):
    penalties = []
    for _ in range(trials):
        penalty = run_episode(syndrome_circuit)  
        penalties.append(penalty)
   
    return np.mean(penalties)

def select_parents(population, num_parents):
    parents = random.sample(population, num_parents)
    parents.append(parents)
    return parents

def crossover(parent1, parent2):    
    child = copy.deepcopy(parent1)  
    return child

def mutate(syndrome_circuit, mutation_probability):
    mutated_circuit = copy.deepcopy(syndrome_circuit)

    for i in range(len(mutated_circuit.data)):
        if random.random() < mutation_probability:
            random_gate = QuantumGate()
            position = random.randint(0, len(mutated_circuit.data))
            mutated_circuit.data
            
            
def QuantumGate():  
    matrix_size = 5

    random_matrix = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
    U, P = polar(random_matrix)


    qr = QuantumRegister(matrix_size)
    random_gate_circuit = QuantumCircuit(qr)

    gate_type = np.random.choice(['CNOT', 'Toffoli' , 'X'])

    if gate_type == 'CNOT':
        control_qubit, target_qubit = np.random.choice(range(matrix_size), size=2, replace=False)
        random_gate_circuit.cx(qr[control_qubit], qr[target_qubit])
    elif gate_type == 'Toffoli':
        control_qubits1, control_qubits2,  target_qubit = np.random.choice(range(matrix_size), size=3, replace=False)
        random_gate_circuit.ccx(qr[control_qubits1], qr[control_qubits2], qr[target_qubit])
    elif gate_type == 'X':
        gate = np.random.choice(range(matrix_size), size=1)[0]
        random_gate_circuit.x(qr[gate])

    return(random_gate_circuit)


def run_episode(syn_qc):
    penalties = []

    for _ in range(trials):
        qc = QuantumCircuit(k, k)
        ax = np.random.rand() * 2 * np.pi
        ay = np.random.rand() * 2 * np.pi
        q_state = prep_state(qc, ax, ay)
        q_state.measure(range(k), range(k))
        result = execute(q_state, simulator, shots=shots).result()
        m1 = result.get_counts(q_state)

        ec = add_enc_circ(ax, ay)

        err_trials = 1

        for _ in range(err_trials):
            enc_circ = copy.deepcopy(ec)
            err_circ = add_err_circ(enc_circ)
            syn_circ = err_circ.compose(syn_qc)
            dec_circ = add_dec_circ(syn_circ)
            dec_circ.measure(range(k), range(k))
            result = execute(dec_circ, simulator, shots=shots).result()
            m2 = result.get_counts(dec_circ)

            penalty = sum(abs(m1.get(key, 0) - m2.get(key, 0))/shots for key in set(m1) | set(m2)) 
            penalties.append(penalty)

    return penalties



population_size = 10
num_generations = 5
num_parents_mating = 4
mutation_probability = 0.1
population = []

best_syndrome_circuit = None
best_fitness = float('inf')

for generation in range(num_generations):
    fitness_scores = []

    population = create_initial_population(population_size)
    for circuit in population:
        fitness = np.mean(evaluate_fitness(circuit))
        fitness_scores.append(fitness)

        if fitness < best_fitness:
            best_fitness = fitness
            best_syndrome_circuit = copy.deepcopy(circuit)

    parents = random.sample(population, num_parents_mating)
    parents.append(best_syndrome_circuit)

    children = []

    while len(children) < population_size - num_parents_mating:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = parent1
        child = mutate(child, mutation_probability)
        children.append(child)

    new_population = parents + children
    population = new_population

print("Best Fitness:", best_fitness)
print("Best Syndrome Circuit:")
with open('best_syndrome_circuit_as.txt', 'w',encoding='utf-8') as f:
    f.write(str(best_syndrome_circuit.draw(output='text')))


#print("Population Circuits:")



