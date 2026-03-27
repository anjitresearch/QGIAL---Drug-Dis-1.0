"""
PAINS Detector

Implements detection of Pan-Assay Interference Compounds (PAINS) patterns
to identify and filter out problematic molecular scaffolds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


class PAINSDetector:
    """
    Pan-Assay Interference Compounds (PAINS) detector.
    
    Identifies molecular substructures known to cause assay interference
    and false positives in drug discovery screening.
    """
    
    def __init__(self):
        """Initialize PAINS detector with known PAINS patterns."""
        # PAINS SMARTS patterns (simplified subset)
        self.pains_patterns = {
            'A1': {'smarts': '[#6]=[#6]([#6])[#6]', 'description': 'Enone Michael acceptor'},
            'A2': {'smarts': '[#6]=[#6]([#6])[#6]=[#6]', 'description': 'Dienone Michael acceptor'},
            'B1': {'smarts': '[#6]=[#6]([#7])', 'description': 'Quinone imine'},
            'B2': {'smarts': '[#6]=[#6]([#8])', 'description': 'Quinone'},
            'C1': {'smarts': '[#7]=[#7]', 'description': 'Diazo compound'},
            'C2': {'smarts': '[#7]=[#7+]', 'description': 'Diazo cation'},
            'D1': {'smarts': '[#16]=[#8]', 'description': 'Thioaldehyde'},
            'D2': {'smarts': '[#16]=[#6]', 'description': 'Thioketone'},
            'E1': {'smarts': '[#6]=[#7+]', 'description': 'Imine cation'},
            'E2': {'smarts': '[#7]=[#6]', 'description': 'Imine'},
            'F1': {'smarts': '[#6]1[#6][#6][#6][#6][#6]1', 'description': 'Benzene ring'},
            'F2': {'smarts': '[#6]1[#6][#6][#6][#6]1', 'description': 'Cyclopentadiene'},
            'G1': {'smarts': '[#8]=[#8]', 'description': 'Peroxide'},
            'G2': {'smarts': '[#8-][#8]', 'description': 'Peroxide anion'},
            'H1': {'smarts': '[#7]C(=O)', 'description': 'Amide'},
            'H2': {'smarts': '[#7]C(=S)', 'description': 'Thioamide'},
            'I1': {'smarts': '[#6](=O)[#7]', 'description': 'Lactam'},
            'I2': {'smarts': '[#6](=S)[#7]', 'description': 'Thiolactam'},
            'J1': {'smarts': '[#6]=[#6][#6]=[#6]', 'description': 'Conjugated diene'},
            'J2': {'smarts': '[#6]=[#6][#6]=[#6][#6]=[#6]', 'description': 'Conjugated triene'},
            'K1': {'smarts': '[#7]C#N', 'description': 'Cyanamide'},
            'K2': {'smarts': '[#6]C#N', 'description': 'Nitrile'},
            'L1': {'smarts': '[#16]C(=O)', 'description': 'Thioester'},
            'L2': {'smarts': '[#8]C(=O)', 'description': 'Ester'},
            'M1': {'smarts': '[#7]C(=O)[#8]', 'description': 'Carbamate'},
            'M2': {'smarts': '[#7]C(=O)[#7]', 'description': 'Urea'},
            'N1': {'smarts': '[#6]1[#6]([#6])[#6][#6][#6]1', 'description': 'Phenol'},
            'N2': {'smarts': '[#6]1[#6]([#7])[#6][#6][#6]1', 'description': 'Aniline'},
            'O1': {'smarts': '[#6]=[#6][#6]=[#6][#6]=[#6][#6]=[#6]', 'description': 'Conjugated tetraene'},
            'O2': {'smarts': '[#6]=[#6][#6]=[#6][#6]=[#6]', 'description': 'Conjugated pentaene'},
            'P1': {'smarts': '[#7]1[#6][#6][#6][#6][#6]1', 'description': 'Pyridine'},
            'P2': {'smarts': '[#7]1[#6][#6][#6][#6]1', 'description': 'Pyridine (5-membered)'},
            'Q1': {'smarts': '[#16]1[#6][#6][#6][#6][#6]1', 'description': 'Thiophene'},
            'Q2': {'smarts': '[#8]1[#6][#6][#6][#6][#6]1', 'description': 'Furan'},
            'R1': {'smarts': '[#6]=[#6][#6]=[#6][#6]=[#6][#6]=[#6][#6]=[#6]', 'description': 'Extended conjugation'},
            'R2': {'smarts': '[#6]=[#6][#6]=[#6][#6]=[#6][#6]=[#6]', 'description': 'Highly conjugated system'}
        }
        
        # Compile SMARTS patterns
        self.compiled_patterns = {}
        for pattern_id, pattern_info in self.pains_patterns.items():
            try:
                mol = Chem.MolFromSmarts(pattern_info['smarts'])
                if mol is not None:
                    self.compiled_patterns[pattern_id] = {
                        'mol': mol,
                        'description': pattern_info['description']
                    }
            except Exception as e:
                print(f"Error compiling PAINS pattern {pattern_id}: {e}")
                
        # PAINS severity levels
        self.severity_levels = {
            'high': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],  # Highly problematic
            'medium': ['D1', 'D2', 'E1', 'E2', 'G1', 'G2'],  # Moderately problematic
            'low': ['F1', 'F2', 'H1', 'H2', 'I1', 'I2']  # Mildly problematic
        }
        
    def detect_pains(self, molecule: Chem.Mol) -> Dict:
        """
        Detect PAINS patterns in a molecule.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary containing PAINS detection results
        """
        if molecule is None:
            return {
                'is_pains': False,
                'pains_score': 0.0,
                'detected_patterns': [],
                'severity': 'none',
                'pattern_details': {}
            }
            
        detected_patterns = []
        pattern_details = {}
        total_score = 0.0
        
        # Check each PAINS pattern
        for pattern_id, pattern_info in self.compiled_patterns.items():
            try:
                # Check if pattern matches
                matches = molecule.GetSubstructMatches(pattern_info['mol'])
                if matches:
                    detected_patterns.append(pattern_id)
                    
                    # Calculate pattern score
                    pattern_score = len(matches)
                    total_score += pattern_score
                    
                    # Store pattern details
                    pattern_details[pattern_id] = {
                        'description': pattern_info['description'],
                        'matches': len(matches),
                        'match_atoms': matches
                    }
                    
            except Exception as e:
                print(f"Error checking PAINS pattern {pattern_id}: {e}")
                
        # Determine severity
        severity = self._determine_severity(detected_patterns)
        
        # Normalize PAINS score
        max_possible_score = len(self.compiled_patterns) * 3  # Max 3 matches per pattern
        normalized_score = min(total_score / max_possible_score, 1.0)
        
        return {
            'is_pains': len(detected_patterns) > 0,
            'pains_score': normalized_score,
            'detected_patterns': detected_patterns,
            'severity': severity,
            'pattern_details': pattern_details
        }
        
    def _determine_severity(self, detected_patterns: List[str]) -> str:
        """Determine PAINS severity based on detected patterns."""
        if not detected_patterns:
            return 'none'
            
        # Check for high severity patterns
        for pattern in detected_patterns:
            if pattern in self.severity_levels['high']:
                return 'high'
                
        # Check for medium severity patterns
        for pattern in detected_patterns:
            if pattern in self.severity_levels['medium']:
                return 'medium'
                
        # Otherwise low severity
        return 'low'
        
    def filter_pains(self, molecules: List[Chem.Mol], 
                    max_score: float = 0.1) -> List[Chem.Mol]:
        """
        Filter out PAINS compounds from a list of molecules.
        
        Args:
            molecules: List of molecules to filter
            max_score: Maximum PAINS score threshold
            
        Returns:
            Filtered list of molecules
        """
        filtered_molecules = []
        
        for mol in molecules:
            if mol is not None:
                pains_result = self.detect_pains(mol)
                if pains_result['pains_score'] <= max_score:
                    filtered_molecules.append(mol)
                    
        return filtered_molecules
        
    def get_pains_statistics(self, molecules: List[Chem.Mol]) -> Dict:
        """
        Get PAINS statistics for a set of molecules.
        
        Args:
            molecules: List of molecules
            
        Returns:
            Statistics dictionary
        """
        if not molecules:
            return {}
            
        total_molecules = len(molecules)
        pains_counts = {'none': 0, 'low': 0, 'medium': 0, 'high': 0}
        pattern_counts = {}
        total_pains_score = 0.0
        
        for mol in molecules:
            if mol is not None:
                result = self.detect_pains(mol)
                pains_counts[result['severity']] += 1
                total_pains_score += result['pains_score']
                
                # Count individual patterns
                for pattern in result['detected_patterns']:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    
        # Calculate percentages
        pains_percentages = {
            severity: (count / total_molecules) * 100 
            for severity, count in pains_counts.items()
        }
        
        return {
            'total_molecules': total_molecules,
            'pains_distribution': pains_counts,
            'pains_percentages': pains_percentages,
            'pattern_frequencies': pattern_counts,
            'average_pains_score': total_pains_score / total_molecules,
            'pains_detection_rate': (total_molecules - pains_counts['none']) / total_molecules
        }
        
    def highlight_pains_patterns(self, molecule: Chem.Mol) -> Optional[object]:
        """
        Highlight PAINS patterns in a molecule for visualization.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            RDKit molecule with highlighted atoms or None if no PAINS
        """
        if molecule is None:
            return None
            
        pains_result = self.detect_pains(molecule)
        
        if not pains_result['is_pains']:
            return molecule
            
        # Create a copy for highlighting
        mol_copy = Chem.Mol(molecule)
        
        # Highlight atoms involved in PAINS patterns
        highlight_atoms = set()
        for pattern_id, pattern_details in pains_result['pattern_details'].items():
            for match_atoms in pattern_details['match_atoms']:
                highlight_atoms.update(match_atoms)
                
        # Add highlight information
        if highlight_atoms:
            mol_copy.__sssAtoms = list(highlight_atoms)
            
        return mol_copy
        
    def get_pains_summary(self, molecule: Chem.Mol) -> str:
        """
        Get a human-readable summary of PAINS detection.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Summary string
        """
        if molecule is None:
            return "Invalid molecule"
            
        result = self.detect_pains(molecule)
        
        if not result['is_pains']:
            return "No PAINS patterns detected"
            
        summary = f"PAINS detected (Severity: {result['severity'].upper()}, Score: {result['pains_score']:.3f})\n"
        summary += "Detected patterns:\n"
        
        for pattern_id in result['detected_patterns']:
            pattern_info = self.compiled_patterns.get(pattern_id, {})
            description = pattern_info.get('description', 'Unknown pattern')
            matches = result['pattern_details'][pattern_id]['matches']
            summary += f"  - {pattern_id}: {description} ({matches} matches)\n"
            
        return summary
        
    def batch_detect_pains(self, molecules: List[Chem.Mol]) -> List[Dict]:
        """
        Detect PAINS in a batch of molecules.
        
        Args:
            molecules: List of molecules
            
        Returns:
            List of PAINS detection results
        """
        return [self.detect_pains(mol) for mol in molecules]
        
    def remove_problematic_patterns(self, molecule: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Attempt to remove problematic PAINS patterns from a molecule.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Modified molecule or None if removal failed
        """
        if molecule is None:
            return None
            
        pains_result = self.detect_pains(molecule)
        
        if not pains_result['is_pains']:
            return molecule
            
        # This is a simplified approach - in practice would need
        # more sophisticated pattern removal strategies
        try:
            rw_mol = Chem.RWMol(molecule)
            
            # Remove atoms involved in high-severity patterns
            atoms_to_remove = set()
            for pattern_id in pains_result['detected_patterns']:
                if pattern_id in self.severity_levels['high']:
                    for match_atoms in pains_result['pattern_details'][pattern_id]['match_atoms']:
                        atoms_to_remove.update(match_atoms)
                        
            # Remove atoms (in reverse order to maintain indices)
            for atom_idx in sorted(atoms_to_remove, reverse=True):
                if atom_idx < rw_mol.GetNumAtoms():
                    rw_mol.RemoveAtom(atom_idx)
                    
            # Convert back to Mol
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except Exception as e:
            print(f"Error removing PAINS patterns: {e}")
            return molecule
