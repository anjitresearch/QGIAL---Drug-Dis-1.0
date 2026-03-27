"""
Molecule Generator

Implements molecular generation from quantum-enhanced features,
converting feature vectors back to valid molecular structures.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


class MoleculeGenerator:
    """
    Generates valid molecular structures from feature vectors.
    
    Converts quantum-enhanced feature representations back to
    chemically valid SMILES strings and RDKit molecules.
    """
    
    def __init__(self):
        """Initialize molecule generator."""
        # Common molecular scaffolds
        self.scaffolds = {
            'benzene': 'c1ccccc1',
            'pyridine': 'c1ccncc1',
            'pyrimidine': 'c1ncnc(n1)',
            'indole': 'c1ccc2c(c1)[nH]c3ccccc23',
            'quinoline': 'c1ccc2ccccc2n1',
            'pyrazole': 'c1cn[nH]c1',
            'imidazole': 'c1ncc[nH]1',
            'thiazole': 'c1ncsc1',
            'oxazole': 'c1cocn1',
            'furan': 'c1ccoc1',
            'thiophene': 'c1ccsc1',
            'pyrrolidine': 'C1CCCN1',
            'piperidine': 'C1CCNCC1',
            'morpholine': 'C1CCOCCN1',
            'piperazine': 'C1CCNCCN1'
        }
        
        # Functional groups for decoration
        self.functional_groups = {
            'hydroxyl': '[OH]',
            'amine': '[NH2]',
            'carboxyl': '[C](=O)O',
            'ester': '[C](=O)O[C]',
            'amide': '[C](=O)N',
            'ketone': '[C](=O)[C]',
            'aldehyde': '[C](=O)H',
            'nitrile': '[C]#N',
            'nitro': '[N+](=O)[O-]',
            'halogen': '[F,Cl,Br,I]',
            'methyl': '[CH3]',
            'ethyl': '[CH2][CH3]',
            'methoxy': '[OCH3]',
            'fluoro': '[F]',
            'chloro': '[Cl]',
            'bromo': '[Br]',
            'iodo': '[I]'
        }
        
        # Linkers for connecting fragments
        self.linkers = {
            'single': '-',
            'double': '=',
            'triple': '#',
            'aromatic': ':'
        }
        
    def features_to_molecule(self, features: np.ndarray) -> Optional[Chem.Mol]:
        """
        Convert feature vector to RDKit molecule.
        
        Args:
            features: Feature vector from quantum generator
            
        Returns:
            RDKit molecule or None if conversion fails
        """
        try:
            # Determine molecular properties from features
            mol_properties = self._extract_properties_from_features(features)
            
            # Generate scaffold based on properties
            scaffold = self._select_scaffold(mol_properties)
            
            # Add functional groups
            molecule = self._decorate_scaffold(scaffold, mol_properties)
            
            # Validate and optimize molecule
            if self._validate_molecule(molecule):
                optimized_mol = self._optimize_molecule(molecule)
                return optimized_mol
            else:
                return None
                
        except Exception as e:
            print(f"Error converting features to molecule: {e}")
            return None
            
    def _extract_properties_from_features(self, features: np.ndarray) -> Dict:
        """Extract molecular properties from feature vector."""
        # Normalize features to [0, 1] range
        normalized_features = np.clip(features, -1, 1)
        normalized_features = (normalized_features + 1) / 2
        
        # Determine molecular properties
        properties = {
            'molecular_weight_target': normalized_features[0] * 600 + 100,  # 100-700 Da
            'logp_target': normalized_features[1] * 6 - 2,  # -2 to 4
            'tpsa_target': normalized_features[2] * 200,  # 0-200 Ų
            'hbd_target': int(normalized_features[3] * 6),  # 0-6 HBD
            'hba_target': int(normalized_features[4] * 12),  # 0-12 HBA
            'rotatable_bonds_target': int(normalized_features[5] * 10),  # 0-10
            'aromatic_rings_target': int(normalized_features[6] * 4),  # 0-4 aromatic rings
            'complexity': normalized_features[7],  # 0-1 complexity score
            'hetero_atoms_target': int(normalized_features[8] * 10),  # 0-10 hetero atoms
            'functional_groups': int(normalized_features[9] * 5)  # 0-5 functional groups
        }
        
        return properties
        
    def _select_scaffold(self, properties: Dict) -> str:
        """Select molecular scaffold based on properties."""
        # Choose scaffold based on aromatic ring count and complexity
        aromatic_rings = properties['aromatic_rings_target']
        complexity = properties['complexity']
        
        if aromatic_rings == 0:
            # Aliphatic scaffold
            if complexity < 0.3:
                return self.scaffolds['pyrrolidine']
            elif complexity < 0.6:
                return self.scaffolds['piperidine']
            else:
                return self.scaffolds['morpholine']
                
        elif aromatic_rings == 1:
            # Single aromatic ring
            if complexity < 0.3:
                return self.scaffolds['benzene']
            elif complexity < 0.6:
                return self.scaffolds['pyridine']
            else:
                return self.scaffolds['pyrimidine']
                
        elif aromatic_rings == 2:
            # Two aromatic rings
            if complexity < 0.5:
                return self.scaffolds['indole']
            else:
                return self.scaffolds['quinoline']
                
        else:
            # Multiple aromatic rings
            return self.scaffolds['quinoline']
            
    def _decorate_scaffold(self, scaffold: str, properties: Dict) -> Chem.Mol:
        """Decorate scaffold with functional groups."""
        try:
            # Start with scaffold
            mol = Chem.MolFromSmiles(scaffold)
            if mol is None:
                return None
                
            rw_mol = Chem.RWMol(mol)
            
            # Add functional groups
            n_groups = min(properties['functional_groups'], 3)  # Limit to 3 groups
            
            # Select functional groups based on properties
            selected_groups = self._select_functional_groups(properties, n_groups)
            
            # Add groups to molecule
            for group_smarts in selected_groups:
                group_mol = Chem.MolFromSmarts(group_smarts)
                if group_mol is not None:
                    # Combine molecules (simplified approach)
                    combined = Chem.CombineMols(rw_mol, group_mol)
                    rw_mol = Chem.RWMol(combined)
                    
            # Convert back to Mol
            final_mol = rw_mol.GetMol()
            
            return final_mol
            
        except Exception as e:
            print(f"Error decorating scaffold: {e}")
            return Chem.MolFromSmiles(scaffold)
            
    def _select_functional_groups(self, properties: Dict, n_groups: int) -> List[str]:
        """Select functional groups based on molecular properties."""
        selected_groups = []
        
        # Determine HBD/HBA requirements
        hbd_needed = properties['hbd_target']
        hba_needed = properties['hba_target']
        
        # Add groups to meet HBD/HBA requirements
        if hbd_needed > 0 and len(selected_groups) < n_groups:
            selected_groups.append(self.functional_groups['hydroxyl'])
            hbd_needed -= 1
            
        if hbd_needed > 0 and len(selected_groups) < n_groups:
            selected_groups.append(self.functional_groups['amine'])
            hbd_needed -= 1
            
        # Add groups to meet HBA requirements
        if hba_needed > 0 and len(selected_groups) < n_groups:
            selected_groups.append(self.functional_groups['carboxyl'])
            hba_needed -= 2
            
        if hba_needed > 0 and len(selected_groups) < n_groups:
            selected_groups.append(self.functional_groups['ester'])
            hba_needed -= 2
            
        # Add additional groups based on LogP target
        logp_target = properties['logp_target']
        if logp_target > 2 and len(selected_groups) < n_groups:
            # Add hydrophobic groups
            selected_groups.append(self.functional_groups['methyl'])
        elif logp_target < 1 and len(selected_groups) < n_groups:
            # Add polar groups
            selected_groups.append(self.functional_groups['hydroxyl'])
            
        # Fill remaining slots with random groups
        available_groups = list(self.functional_groups.values())
        import random
        
        while len(selected_groups) < n_groups:
            group = random.choice(available_groups)
            if group not in selected_groups:
                selected_groups.append(group)
                
        return selected_groups[:n_groups]
        
    def _validate_molecule(self, molecule: Chem.Mol) -> bool:
        """Validate molecule for chemical correctness."""
        if molecule is None:
            return False
            
        try:
            # Check for basic chemical validity
            Chem.SanitizeMol(molecule)
            
            # Check molecular properties
            mw = Descriptors.MolWt(molecule)
            if mw < 50 or mw > 1000:  # Reasonable MW range
                return False
                
            # Check for valid valence
            for atom in molecule.GetAtoms():
                if atom.GetExplicitValence() > atom.GetImplicitValence() + atom.GetFormalCharge():
                    return False
                    
            return True
            
        except:
            return False
            
    def _optimize_molecule(self, molecule: Chem.Mol) -> Chem.Mol:
        """Optimize molecule geometry and properties."""
        try:
            # Add hydrogens
            mol = Chem.AddHs(molecule)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            # Optimize geometry
            AllChem.UFFOptimizeMolecule(mol)
            
            # Remove hydrogens for final output
            mol = Chem.RemoveHs(mol)
            
            return mol
            
        except Exception as e:
            print(f"Error optimizing molecule: {e}")
            return molecule
            
    def generate_batch(self, features_batch: np.ndarray) -> List[Chem.Mol]:
        """
        Generate molecules from a batch of feature vectors.
        
        Args:
            features_batch: Batch of feature vectors
            
        Returns:
            List of generated molecules
        """
        molecules = []
        
        for features in features_batch:
            mol = self.features_to_molecule(features)
            if mol is not None:
                molecules.append(mol)
                
        return molecules
        
    def generate_smiles(self, features: np.ndarray) -> Optional[str]:
        """
        Generate SMILES string from features.
        
        Args:
            features: Feature vector
            
        Returns:
            SMILES string or None if generation fails
        """
        mol = self.features_to_molecule(features)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
        
    def get_molecule_properties(self, molecule: Chem.Mol) -> Dict:
        """
        Get comprehensive molecular properties.
        
        Args:
            molecule: RDKit molecule
            
        Returns:
            Dictionary of molecular properties
        """
        if molecule is None:
            return {}
            
        try:
            properties = {
                'smiles': Chem.MolToSmiles(molecule, canonical=True),
                'molecular_weight': Descriptors.MolWt(molecule),
                'logp': Descriptors.MolLogP(molecule),
                'tpsa': Descriptors.TPSA(molecule),
                'hbd': Descriptors.NumHDonors(molecule),
                'hba': Descriptors.NumHAcceptors(molecule),
                'rotatable_bonds': Descriptors.NumRotatableBonds(molecule),
                'aromatic_rings': Descriptors.NumAromaticRings(molecule),
                'saturated_rings': Descriptors.NumSaturatedRings(molecule),
                'formal_charge': Chem.GetFormalCharge(molecule),
                'num_atoms': molecule.GetNumAtoms(),
                'num_heavy_atoms': molecule.GetNumHeavyAtoms()
            }
            
            # Calculate molecular fingerprint
            fp = GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
            properties['fingerprint'] = list(fp)
            
            return properties
            
        except Exception as e:
            print(f"Error calculating molecule properties: {e}")
            return {}
            
    def filter_by_properties(self, molecules: List[Chem.Mol], 
                           min_mw: float = 100, max_mw: float = 600,
                           min_logp: float = -2, max_logp: float = 4,
                           max_tpsa: float = 140) -> List[Chem.Mol]:
        """
        Filter molecules by property ranges.
        
        Args:
            molecules: List of molecules to filter
            min_mw: Minimum molecular weight
            max_mw: Maximum molecular weight
            min_logp: Minimum LogP
            max_logp: Maximum LogP
            max_tpsa: Maximum TPSA
            
        Returns:
            Filtered list of molecules
        """
        filtered_molecules = []
        
        for mol in molecules:
            if mol is not None:
                try:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    
                    if (min_mw <= mw <= max_mw and 
                        min_logp <= logp <= max_logp and 
                        tpsa <= max_tpsa):
                        filtered_molecules.append(mol)
                        
                except:
                    continue
                    
        return filtered_molecules
