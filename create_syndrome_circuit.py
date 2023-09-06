#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

# Invoke a primitive. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
# result = Sampler("ibmq_qasm_simulator").run(circuits).result()


# In[ ]:


from qiskit import QuantumCircuit


n = 3  
s = 2  
num_syndrome_gates = 4 
num_recovery_gates = 3  
def generate_syndrome_circuit():
    syndrome_circuit = QuantumCircuit(n + s, n)
    
    for _ in range(num_syndrome_gates):
        data_qubit = np.random.randint(n)
        syndrome_qubit = np.random.randint(n, n + s)
        syndrome_circuit.cx(data_qubit, syndrome_qubit)
        
    return syndrome_circuit

def generate_recovery_circuit():
    recovery_circuit = QuantumCircuit(n + s, n)
    
    for _ in range(num_recovery_gates):
        target_qubit = np.random.randint(n)
        num_controls = np.random.randint(1, s + 1) 
        control_qubits = np.random.choice(np.arange(n, n + s), size=num_controls)
        recovery_circuit.mcx(control_qubits, target_qubit)
        
    return recovery_circuit


syndrome_circuit = generate_syndrome_circuit()
recovery_circuit = generate_recovery_circuit()

print("Syndrome Circuit:")
print(syndrome_circuit)

print("\nRecovery Circuit:")
print(recovery_circuit)

