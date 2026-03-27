"""
Property Calculator

Implements comprehensive molecular property calculations for drug discovery.
"""

import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


class PropertyCalculator:
    """
    Comprehensive molecular property calculator.
    
    Calculates various molecular properties including physicochemical,
    topological, and quantum-inspired descriptors.
    """
    
    def __init__(self):
        """Initialize property calculator."""
        self.descriptor_names = [
            'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
            'FractionCSP3', 'MolMR', 'LabuteASA', 'BalabanJ',
            'BertzCT', 'Chi0v', 'Chi1n', 'Kappa1', 'Kappa2', 'Kappa3'
        ]
        
    def calculate_all_properties(self, molecule: Chem.Mol) -> Dict[str, float]:
        """
        Calculate all molecular properties.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of calculated properties
        """
        if molecule is None:
            return {}
            
        properties = {}
        
        # Basic molecular properties
        properties['MolWt'] = Descriptors.MolWt(molecule)
        properties['MolLogP'] = Descriptors.MolLogP(molecule)
        properties['TPSA'] = Descriptors.TPSA(molecule)
        properties['NumHDonors'] = Descriptors.NumHDonors(molecule)
        properties['NumHAcceptors'] = Descriptors.NumHAcceptors(molecule)
        properties['NumRotatableBonds'] = Descriptors.NumRotatableBonds(molecule)
        properties['NumAromaticRings'] = Descriptors.NumAromaticRings(molecule)
        properties['NumSaturatedRings'] = Descriptors.NumSaturatedRings(molecule)
        
        # Advanced descriptors
        properties['FractionCSP3'] = Descriptors.FractionCSP3(molecule)
        properties['MolMR'] = Descriptors.MolMR(molecule)
        properties['LabuteASA'] = Descriptors.LabuteASA(molecule)
        properties['BalabanJ'] = Descriptors.BalabanJ(molecule)
        properties['BertzCT'] = Descriptors.BertzCT(molecule)
        properties['Chi0v'] = Descriptors.Chi0v(molecule)
        properties['Chi1n'] = Descriptors.Chi1n(molecule)
        properties['Kappa1'] = Descriptors.Kappa1(molecule)
        properties['Kappa2'] = Descriptors.Kappa2(molecule)
        properties['Kappa3'] = Descriptors.Kappa3(molecule)
        
        # Additional calculated properties
        properties['HeavyAtomCount'] = molecule.GetNumHeavyAtoms()
        properties['RingCount'] = Descriptors.RingCount(molecule)
        properties['FormalCharge'] = Chem.GetFormalCharge(molecule)
        
        # 3D properties (if coordinates available)
        if molecule.GetNumConformers() > 0:
            properties['MolecularVolume'] = self._calculate_molecular_volume(molecule)
            properties['RadiusOfGyration'] = self._calculate_radius_of_gyration(molecule)
        else:
            properties['MolecularVolume'] = 0.0
            properties['RadiusOfGyration'] = 0.0
            
        return properties
        
    def _calculate_molecular_volume(self, molecule: Chem.Mol) -> float:
        """Calculate approximate molecular volume."""
        try:
            from scipy.spatial import ConvexHull
            
            conformer = molecule.GetConformer()
            coords = conformer.GetPositions()
            
            if len(coords) < 4:
                return 0.0
                
            hull = ConvexHull(coords)
            return hull.volume
            
        except:
            # Fallback to spherical approximation
            conformer = molecule.GetConformer()
            coords = conformer.GetPositions()
            center = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - center, axis=1)
            radius = np.max(distances)
            return (4/3) * np.pi * radius**3
            
    def _calculate_radius_of_gyration(self, molecule: Chem.Mol) -> float:
        """Calculate radius of gyration."""
        try:
            conformer = molecule.GetConformer()
            coords = conformer.GetPositions()
            center = np.mean(coords, axis=0)
            distances_squared = np.sum((coords - center)**2, axis=1)
            return np.sqrt(np.mean(distances_squared))
        except:
            return 0.0
            
    def calculate_property_vector(self, molecule: Chem.Mol) -> np.ndarray:
        """
        Calculate property vector for machine learning.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Property vector as numpy array
        """
        properties = self.calculate_all_properties(molecule)
        
        # Create vector in consistent order
        vector = []
        for prop_name in self.descriptor_names:
            if prop_name in properties:
                vector.append(properties[prop_name])
            else:
                vector.append(0.0)
                
        return np.array(vector)
        
    def batch_calculate_properties(self, molecules: List[Chem.Mol]) -> List[Dict[str, float]]:
        """
        Calculate properties for a batch of molecules.
        
        Args:
            molecules: List of RDKit molecules
            
        Returns:
            List of property dictionaries
        """
        return [self.calculate_all_properties(mol) for mol in molecules]
        
    def get_property_statistics(self, molecules: List[Chem.Mol]) -> Dict:
        """
        Get property statistics for a set of molecules.
        
        Args:
            molecules: List of RDKit molecules
            
        Returns:
            Statistics dictionary
        """
        if not molecules:
            return {}
            
        # Calculate properties for all molecules
        all_properties = self.batch_calculate_properties(molecules)
        
        # Calculate statistics
        statistics = {}
        for prop_name in self.descriptor_names:
            values = [props.get(prop_name, 0.0) for props in all_properties if prop_name in props]
            if values:
                statistics[prop_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
                
        return statistics
