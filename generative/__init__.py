"""
Hybrid Quantum-Classical Generative Models

This module implements HQGAN (Hybrid Quantum-Classical Generative Adversarial Network)
for novel molecular scaffold generation using quantum-enhanced representations.
"""

from .hqgan import HybridQuantumGAN
from .quantum_generator import QuantumGenerator
from .classical_discriminator import ClassicalDiscriminator
from .molecule_generator import MoleculeGenerator

__all__ = [
    'HybridQuantumGAN',
    'QuantumGenerator', 
    'ClassicalDiscriminator',
    'MoleculeGenerator'
]
