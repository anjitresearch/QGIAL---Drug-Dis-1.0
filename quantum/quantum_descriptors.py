"""
Quantum Descriptor Calculator

Computes quantum mechanical descriptors for molecules using
variational quantum circuits and quantum chemistry calculations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from .vqc_molecular import VariationalQuantumCircuit


class QuantumDescriptorCalculator:
    """
    Calculates quantum descriptors for molecular property prediction
    using hybrid quantum-classical approaches.
    """
    
    def __init__(self, n_qubits: int = 27):
        """
        Initialize quantum descriptor calculator.
        
        Args:
            n_qubits: Number of qubits for quantum circuit
        """
        self.vqc = VariationalQuantumCircuit(n_qubits)
        self.descriptor_cache = {}
        
    def calculate_all_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        Calculate comprehensive set of quantum and classical descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary containing all calculated descriptors
        """
        # Get molecular identifier for caching
        mol_id = Chem.MolToSmiles(mol)
        
        if mol_id in self.descriptor_cache:
            return self.descriptor_cache[mol_id]
            
        # Calculate quantum descriptors
        quantum_desc = self.vqc.get_quantum_descriptors(mol)
        
        # Calculate enhanced classical descriptors
        classical_desc = self._calculate_classical_descriptors(mol)
        
        # Calculate quantum-classical hybrid descriptors
        hybrid_desc = self._calculate_hybrid_descriptors(mol, quantum_desc, classical_desc)
        
        # Combine all descriptors
        all_descriptors = {
            **quantum_desc,
            **classical_desc,
            **hybrid_desc,
            'molecule_id': mol_id
        }
        
        # Cache results
        self.descriptor_cache[mol_id] = all_descriptors
        
        return all_descriptors
        
    def _calculate_classical_descriptors(self, mol: Chem.Mol) -> Dict:
        """Calculate classical molecular descriptors."""
        # Add hydrogens for accurate calculations
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        descriptors = {}
        
        # Basic molecular properties
        descriptors['molecular_weight'] = Descriptors.MolWt(mol)
        descriptors['logp'] = Descriptors.MolLogP(mol)
        descriptors['tpsa'] = Descriptors.TPSA(mol)
        descriptors['num_hbd'] = Descriptors.NumHDonors(mol)
        descriptors['num_hba'] = Descriptors.NumHAcceptors(mol)
        descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        descriptors['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
        
        # Electronic properties
        descriptors['formal_charge'] = Chem.GetFormalCharge(mol)
        descriptors['num_atoms'] = mol.GetNumAtoms()
        descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        
        # 3D descriptors
        if mol.GetNumConformers() > 0:
            conformer = mol.GetConformer()
            coords = conformer.GetPositions()
            
            # Calculate molecular volume (approximate)
            descriptors['molecular_volume'] = self._calculate_molecular_volume(coords)
            
            # Calculate radius of gyration
            descriptors['radius_of_gyration'] = self._calculate_radius_of_gyration(coords)
            
        return descriptors
        
    def _calculate_hybrid_descriptors(self, mol: Chem.Mol, 
                                    quantum_desc: Dict, 
                                    classical_desc: Dict) -> Dict:
        """Calculate hybrid quantum-classical descriptors."""
        hybrid_desc = {}
        
        # Quantum-enhanced lipophilicity
        hybrid_desc['quantum_logp'] = classical_desc['logp'] * quantum_desc['quantum_fidelity']
        
        # Quantum-corrected polar surface area
        hybrid_desc['quantum_tpsa'] = classical_desc['tpsa'] * (1 + quantum_desc['von_neumann_entropy'] / 10)
        
        # Entanglement-weighted molecular complexity
        hybrid_desc['entanglement_complexity'] = (classical_desc['num_rotatable_bonds'] * 
                                                 quantum_desc['quantum_entanglement'])
        
        # Quantum-enhanced drug-likeness score
        hybrid_desc['quantum_drug_score'] = self._calculate_quantum_drug_score(
            classical_desc, quantum_desc
        )
        
        return hybrid_desc
        
    def _calculate_molecular_volume(self, coords: np.ndarray) -> float:
        """Calculate approximate molecular volume from 3D coordinates."""
        # Use convex hull approach for volume approximation
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(coords)
            return hull.volume
        except:
            # Fallback to spherical approximation
            center = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - center, axis=1)
            radius = np.max(distances)
            return (4/3) * np.pi * radius**3
            
    def _calculate_radius_of_gyration(self, coords: np.ndarray) -> float:
        """Calculate radius of gyration."""
        center = np.mean(coords, axis=0)
        distances_squared = np.sum((coords - center)**2, axis=1)
        return np.sqrt(np.mean(distances_squared))
        
    def _calculate_quantum_drug_score(self, classical_desc: Dict, quantum_desc: Dict) -> float:
        """Calculate quantum-enhanced drug-likeness score."""
        # Lipinski's rule of five components
        mw_ok = classical_desc['molecular_weight'] <= 500
        logp_ok = classical_desc['logp'] <= 5
        hbd_ok = classical_desc['num_hbd'] <= 5
        hba_ok = classical_desc['num_hba'] <= 10
        
        # Basic Lipinski score
        lipinski_score = sum([mw_ok, logp_ok, hbd_ok, hba_ok]) / 4.0
        
        # Quantum enhancement factors
        fidelity_factor = quantum_desc['quantum_fidelity']
        entanglement_factor = quantum_desc['quantum_entanglement']
        
        # Combined quantum drug score
        quantum_drug_score = lipinski_score * fidelity_factor * (1 + entanglement_factor)
        
        return np.clip(quantum_drug_score, 0, 1)
        
    def batch_descriptors(self, mols: List[Chem.Mol]) -> List[Dict]:
        """Calculate descriptors for a batch of molecules."""
        return [self.calculate_all_descriptors(mol) for mol in mols]
        
    def get_descriptor_matrix(self, mols: List[Chem.Mol]) -> np.ndarray:
        """Get descriptor matrix for machine learning."""
        descriptors_list = self.batch_descriptors(mols)
        
        # Extract numerical features
        feature_names = []
        feature_values = []
        
        for desc in descriptors_list:
            features = []
            for key, value in desc.items():
                if isinstance(value, (int, float)) and key != 'molecule_id':
                    features.append(value)
                    if key not in feature_names:
                        feature_names.append(key)
            feature_values.append(features)
            
        return np.array(feature_values), feature_names
