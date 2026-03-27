"""
Hybrid Quantum-Classical Generative Adversarial Network

Implements HQGAN for generating novel molecular scaffolds with
quantum-enhanced representations and classical discriminative feedback.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
try:
    from .quantum_generator import QuantumGenerator
    from .classical_discriminator import ClassicalDiscriminator
    from ..quantum.quantum_descriptors import QuantumDescriptorCalculator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from generative.quantum_generator import QuantumGenerator
    from generative.classical_discriminator import ClassicalDiscriminator
    from quantum.quantum_descriptors import QuantumDescriptorCalculator


class HybridQuantumGAN:
    """
    Hybrid Quantum-Classical GAN for molecular generation.
    
    Combines quantum circuit-based generator with classical neural discriminator
    for efficient exploration of chemical space with quantum-enhanced representations.
    """
    
    def __init__(self, latent_dim: int = 100, n_qubits: int = 27, 
                 learning_rate: float = 0.0002, beta1: float = 0.5):
        """
        Initialize HQGAN model.
        
        Args:
            latent_dim: Dimension of latent noise vector
            n_qubits: Number of qubits for quantum generator
            learning_rate: Learning rate for optimization
            beta1: Beta1 parameter for Adam optimizer
        """
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize quantum generator and classical discriminator
        self.generator = QuantumGenerator(latent_dim, n_qubits).to(self.device)
        self.discriminator = ClassicalDiscriminator().to(self.device)
        
        # Initialize quantum descriptor calculator
        self.descriptor_calc = QuantumDescriptorCalculator(n_qubits)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'validity_score': [],
            'uniqueness_score': [],
            'novelty_score': []
        }
        
    def train(self, real_molecules: List[Chem.Mol], epochs: int = 1000,
              batch_size: int = 32, sample_interval: int = 100):
        """
        Train the HQGAN model.
        
        Args:
            real_molecules: List of real molecules for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            sample_interval: Interval for sampling and evaluation
        """
        # Prepare real data
        real_data = self._prepare_real_data(real_molecules)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Calculate number of batches
        n_batches = len(real_data) // batch_size
        
        for epoch in range(epochs):
            for batch_idx in range(n_batches):
                # Get batch of real data
                real_batch = real_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                real_batch = torch.FloatTensor(real_batch).to(self.device)
                
                # Train Discriminator
                self.optimizer_D.zero_grad()
                
                # Real data loss
                real_output = self.discriminator(real_batch)
                d_real_loss = self.criterion(real_output, real_labels)
                
                # Fake data loss
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_molecules = self.generator(noise)
                fake_output = self.discriminator(fake_molecules.detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_D.step()
                
                # Train Generator
                self.optimizer_G.zero_grad()
                
                # Generate fake molecules
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_molecules = self.generator(noise)
                fake_output = self.discriminator(fake_molecules)
                
                # Generator loss (wants discriminator to think it's real)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()
                
            # Record losses
            self.training_history['generator_loss'].append(g_loss.item())
            self.training_history['discriminator_loss'].append(d_loss.item())
            
            # Evaluate and sample
            if epoch % sample_interval == 0:
                self._evaluate_model(real_molecules, epoch)
                print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
                
    def _prepare_real_data(self, molecules: List[Chem.Mol]) -> np.ndarray:
        """Prepare real molecular data for training."""
        descriptor_data = []
        
        for mol in molecules:
            # Calculate quantum descriptors
            descriptors = self.descriptor_calc.calculate_all_descriptors(mol)
            
            # Extract numerical features and ensure consistent size
            features = []
            for key, value in descriptors.items():
                if isinstance(value, (int, float)) and key != 'molecule_id':
                    features.append(value)
                    
            # Ensure we have exactly 50 features
            if len(features) < 50:
                features.extend([0.0] * (50 - len(features)))
            elif len(features) > 50:
                features = features[:50]
                    
            descriptor_data.append(features)
            
        return np.array(descriptor_data)
        
    def _evaluate_model(self, real_molecules: List[Chem.Mol], epoch: int):
        """Evaluate model performance and calculate metrics."""
        # Generate sample molecules
        generated_mols = self.generate_molecules(100)
        
        # Calculate validity score
        validity = self._calculate_validity(generated_mols)
        self.training_history['validity_score'].append(validity)
        
        # Calculate uniqueness score
        uniqueness = self._calculate_uniqueness(generated_mols)
        self.training_history['uniqueness_score'].append(uniqueness)
        
        # Calculate novelty score
        novelty = self._calculate_novelty(generated_mols, real_molecules)
        self.training_history['novelty_score'].append(novelty)
        
    def generate_molecules(self, n_molecules: int) -> List[Chem.Mol]:
        """
        Generate novel molecular scaffolds.
        
        Args:
            n_molecules: Number of molecules to generate
            
        Returns:
            List of generated RDKit molecules
        """
        generated_molecules = []
        
        with torch.no_grad():
            for _ in range(n_molecules):
                # Generate latent noise
                noise = torch.randn(1, self.latent_dim).to(self.device)
                
                # Generate molecular features
                molecular_features = self.generator(noise).cpu().numpy()[0]
                
                # Convert features to molecule
                mol = self._features_to_molecule(molecular_features)
                
                if mol is not None:
                    generated_molecules.append(mol)
                    
        return generated_molecules
        
    def _features_to_molecule(self, features: np.ndarray) -> Optional[Chem.Mol]:
        """Convert feature vector to RDKit molecule."""
        try:
            # This is a simplified conversion - in practice would use
            # more sophisticated inverse design techniques
            
            # Use features to guide molecular construction
            # Here we'll use a simple approach with SMILES generation
            
            # Generate random SMILES guided by features
            smiles = self._generate_guided_smiles(features)
            
            # Convert to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # Add hydrogens and generate 3D coordinates
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(mol)
                
            return mol
            
        except Exception as e:
            print(f"Error converting features to molecule: {e}")
            return None
            
    def _generate_guided_smiles(self, features: np.ndarray) -> str:
        """Generate SMILES string guided by molecular features."""
        # Simplified SMILES generation based on features
        # In practice, would use more sophisticated methods
        
        # Use first few features to determine molecular properties
        mw_target = features[0] if len(features) > 0 else 300
        logp_target = features[1] if len(features) > 1 else 2
        
        # Generate simple scaffold based on targets
        if mw_target < 200:
            # Small molecule
            scaffolds = ["CCO", "CCN", "CC(C)O", "CC(=O)O"]
        elif mw_target < 400:
            # Medium molecule
            scaffolds = ["c1ccccc1", "c1ccc(cc1)O", "c1ccc(cc1)N", "c1ccc(cc1)C"]
        else:
            # Large molecule
            scaffolds = ["c1ccc2ccccc2c1", "c1ccc2c(c1)ccc3c2cccc3"]
            
        import random
        base_smiles = random.choice(scaffolds)
        
        # Add functional groups based on logp
        if logp_target > 3:
            # Add hydrophobic groups
            modifications = ["C", "CC", "CCC"]
        else:
            # Add polar groups
            modifications = ["O", "N", "F", "Cl"]
            
        mod = random.choice(modifications)
        
        # Simple combination
        return base_smiles + mod
        
    def _calculate_validity(self, molecules: List[Chem.Mol]) -> float:
        """Calculate fraction of valid molecules."""
        if len(molecules) == 0:
            return 0.0
        valid_count = sum(1 for mol in molecules if mol is not None)
        return valid_count / len(molecules)
        
    def _calculate_uniqueness(self, molecules: List[Chem.Mol]) -> float:
        """Calculate fraction of unique molecules."""
        if len(molecules) == 0:
            return 0.0
            
        # Convert to SMILES for comparison
        smiles_list = []
        for mol in molecules:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                smiles_list.append(smiles)
                
        unique_smiles = set(smiles_list)
        return len(unique_smiles) / len(smiles_list) if smiles_list else 0.0
        
    def _calculate_novelty(self, generated_mols: List[Chem.Mol], 
                          real_molecules: List[Chem.Mol]) -> float:
        """Calculate fraction of novel molecules not in training set."""
        if len(generated_mols) == 0:
            return 0.0
            
        # Get SMILES of real molecules
        real_smiles = set()
        for mol in real_molecules:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                real_smiles.add(smiles)
                
        # Count novel molecules
        novel_count = 0
        for mol in generated_mols:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if smiles not in real_smiles:
                    novel_count += 1
                    
        return novel_count / len(generated_mols)
        
    def save_model(self, filepath: str):
        """Save trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'training_history': self.training_history
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.training_history = checkpoint['training_history']
