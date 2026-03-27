"""
Quantum Computing Module for QGIAL

This module implements variational quantum circuits (VQCs) for real-time
protein-ligand interaction energy and conformational entropy calculations
using IBM's 27-qubit Falcon processor.
"""

from .vqc_molecular import VariationalQuantumCircuit
from .quantum_descriptors import QuantumDescriptorCalculator
from .quantum_optimizer import QuantumOptimizer

__all__ = [
    'VariationalQuantumCircuit',
    'QuantumDescriptorCalculator', 
    'QuantumOptimizer'
]
