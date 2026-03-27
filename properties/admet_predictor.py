"""
ADMET Predictor

Implements comprehensive ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
prediction for molecules using machine learning models and rule-based methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Fingerprints import FingerprintMols


class ADMETPredictor:
    """
    Comprehensive ADMET predictor for drug discovery.
    
    Predicts various ADMET properties including bioavailability,
    toxicity, metabolic stability, and overall drug-likeness.
    """
    
    def __init__(self):
        """Initialize ADMET predictor."""
        # ADMET property ranges and thresholds
        self.property_thresholds = {
            'bioavailability': {'min': 0.0, 'max': 1.0, 'optimal': 0.8},
            'caco2_permeability': {'min': -2.0, 'max': 2.0, 'optimal': 0.5},
            'pampa_permeability': {'min': -2.0, 'max': 2.0, 'optimal': 0.0},
            'plasma_protein_binding': {'min': 0.0, 'max': 1.0, 'optimal': 0.3},
            'volume_distribution': {'min': 0.1, 'max': 10.0, 'optimal': 1.0},
            'blood_brain_barrier': {'min': -2.0, 'max': 2.0, 'optimal': 0.0},
            'cytochrome_inhibition': {'min': 0.0, 'max': 1.0, 'optimal': 0.0},
            'clearance': {'min': 0.1, 'max': 10.0, 'optimal': 1.0},
            'half_life': {'min': 0.1, 'max': 24.0, 'optimal': 8.0},
            'toxicity_score': {'min': 0.0, 'max': 1.0, 'optimal': 0.0},
            'hepatotoxicity': {'min': 0.0, 'max': 1.0, 'optimal': 0.0},
            'cardiotoxicity': {'min': 0.0, 'max': 1.0, 'optimal': 0.0},
            'mutagenicity': {'min': 0.0, 'max': 1.0, 'optimal': 0.0},
            'carcinogenicity': {'min': 0.0, 'max': 1.0, 'optimal': 0.0}
        }
        
        # Molecular descriptors for ADMET prediction
        self.descriptor_names = [
            'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
            'FractionCSP3', 'MolMR', 'LabuteASA', 'BalabanJ',
            'BertzCT', 'Chi0v', 'Chi1n', 'Kappa1', 'Kappa2', 'Kappa3'
        ]
        
    def predict_properties(self, molecule: Chem.Mol) -> Dict[str, float]:
        """
        Predict comprehensive ADMET properties for a molecule.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of predicted ADMET properties
        """
        if molecule is None:
            return self._get_default_properties()
            
        # Calculate molecular descriptors
        descriptors = self._calculate_descriptors(molecule)
        
        # Predict absorption properties
        absorption = self._predict_absorption(descriptors)
        
        # Predict distribution properties
        distribution = self._predict_distribution(descriptors)
        
        # Predict metabolism properties
        metabolism = self._predict_metabolism(descriptors)
        
        # Predict excretion properties
        excretion = self._predict_excretion(descriptors)
        
        # Predict toxicity properties
        toxicity = self._predict_toxicity(descriptors)
        
        # Calculate overall ADMET score
        overall_score = self._calculate_overall_score(
            absorption, distribution, metabolism, excretion, toxicity
        )
        
        # Combine all properties
        admet_properties = {
            **absorption,
            **distribution,
            **metabolism,
            **excretion,
            **toxicity,
            'overall_score': overall_score
        }
        
        return admet_properties
        
    def _get_default_properties(self) -> Dict[str, float]:
        """Get default ADMET properties for invalid molecules."""
        return {prop: 0.0 for prop in self.property_thresholds.keys()}
        
    def _calculate_descriptors(self, molecule: Chem.Mol) -> np.ndarray:
        """Calculate molecular descriptors for ADMET prediction."""
        try:
            # Add hydrogens for accurate calculations
            mol = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            
            # Calculate descriptors
            descriptors = []
            
            # Basic molecular properties
            descriptors.append(Descriptors.MolWt(mol))
            descriptors.append(Descriptors.MolLogP(mol))
            descriptors.append(Descriptors.TPSA(mol))
            descriptors.append(Descriptors.NumHDonors(mol))
            descriptors.append(Descriptors.NumHAcceptors(mol))
            descriptors.append(Descriptors.NumRotatableBonds(mol))
            descriptors.append(Descriptors.NumAromaticRings(mol))
            descriptors.append(Descriptors.NumSaturatedRings(mol))
            
            # Advanced descriptors
            descriptors.append(Descriptors.FractionCSP3(mol))
            descriptors.append(Descriptors.MolMR(mol))
            descriptors.append(Descriptors.LabuteASA(mol))
            descriptors.append(Descriptors.BalabanJ(mol))
            descriptors.append(Descriptors.BertzCT(mol))
            descriptors.append(Descriptors.Chi0v(mol))
            descriptors.append(Descriptors.Chi1n(mol))
            descriptors.append(Descriptors.Kappa1(mol))
            descriptors.append(Descriptors.Kappa2(mol))
            descriptors.append(Descriptors.Kappa3(mol))
            
            return np.array(descriptors)
            
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return np.zeros(len(self.descriptor_names))
            
    def _predict_absorption(self, descriptors: np.ndarray) -> Dict[str, float]:
        """Predict absorption properties."""
        mw, logp, tpsa, hbd, hba, rot_bonds = descriptors[:6]
        
        # Bioavailability prediction (simplified)
        bioavailability = 1.0
        
        # Lipinski's rule of five
        if mw > 500:
            bioavailability -= 0.2
        if logp > 5:
            bioavailability -= 0.2
        if hbd > 5:
            bioavailability -= 0.2
        if hba > 10:
            bioavailability -= 0.2
            
        # Veber's rules
        if rot_bonds > 10:
            bioavailability -= 0.1
        if tpsa > 140:
            bioavailability -= 0.1
            
        bioavailability = np.clip(bioavailability, 0.0, 1.0)
        
        # Caco-2 permeability prediction
        caco2_perm = 0.5 - (tpsa / 200) + (logp / 4)
        caco2_perm = np.clip(caco2_perm, -2.0, 2.0)
        
        # PAMPA permeability prediction
        pampa_perm = 0.3 - (tpsa / 150) + (logp / 3)
        pampa_perm = np.clip(pampa_perm, -2.0, 2.0)
        
        return {
            'bioavailability': bioavailability,
            'caco2_permeability': caco2_perm,
            'pampa_permeability': pampa_perm
        }
        
    def _predict_distribution(self, descriptors: np.ndarray) -> Dict[str, float]:
        """Predict distribution properties."""
        mw, logp, tpsa = descriptors[:3]
        
        # Plasma protein binding prediction
        ppb = 1.0 / (1.0 + np.exp(-(logp - 2)))
        ppb = np.clip(ppb, 0.0, 1.0)
        
        # Volume of distribution prediction
        vdist = 0.5 + (logp / 4) - (mw / 1000)
        vdist = np.clip(vdist, 0.1, 10.0)
        
        # Blood-brain barrier penetration
        bbb = 0.5 - (tpsa / 100) + (logp / 3)
        bbb = np.clip(bbb, -2.0, 2.0)
        
        return {
            'plasma_protein_binding': ppb,
            'volume_distribution': vdist,
            'blood_brain_barrier': bbb
        }
        
    def _predict_metabolism(self, descriptors: np.ndarray) -> Dict[str, float]:
        """Predict metabolism properties."""
        mw, logp, tpsa, hbd, hba = descriptors[:5]
        
        # Cytochrome P450 inhibition prediction
        cyp_inhibition = 0.3 + (logp / 10) + (tpsa / 200)
        cyp_inhibition = np.clip(cyp_inhibition, 0.0, 1.0)
        
        # Metabolic stability prediction
        metabolic_stability = 1.0 - (hbd / 10) - (hba / 20)
        metabolic_stability = np.clip(metabolic_stability, 0.0, 1.0)
        
        return {
            'cytochrome_inhibition': cyp_inhibition,
            'metabolic_stability': metabolic_stability
        }
        
    def _predict_excretion(self, descriptors: np.ndarray) -> Dict[str, float]:
        """Predict excretion properties."""
        mw, logp, tpsa = descriptors[:3]
        
        # Clearance prediction
        clearance = 0.5 + (mw / 500) + (tpsa / 100)
        clearance = np.clip(clearance, 0.1, 10.0)
        
        # Half-life prediction
        half_life = 8.0 - (logp * 2) + (tpsa / 50)
        half_life = np.clip(half_life, 0.1, 24.0)
        
        return {
            'clearance': clearance,
            'half_life': half_life
        }
        
    def _predict_toxicity(self, descriptors: np.ndarray) -> Dict[str, float]:
        """Predict toxicity properties."""
        mw, logp, tpsa, hbd, hba, rot_bonds = descriptors[:6]
        
        # General toxicity score
        toxicity = 0.1
        
        # High molecular weight increases toxicity risk
        if mw > 600:
            toxicity += 0.2
            
        # High LogP increases toxicity risk
        if logp > 5:
            toxicity += 0.2
            
        # High TPSA can affect toxicity
        if tpsa > 150:
            toxicity += 0.1
            
        # Too many H-bond donors/acceptors
        if hbd > 6 or hba > 12:
            toxicity += 0.1
            
        toxicity = np.clip(toxicity, 0.0, 1.0)
        
        # Specific toxicity predictions
        hepatotoxicity = toxicity * 0.8
        cardiotoxicity = toxicity * 0.6
        mutagenicity = toxicity * 0.4
        carcinogenicity = toxicity * 0.3
        
        return {
            'toxicity_score': toxicity,
            'hepatotoxicity': hepatotoxicity,
            'cardiotoxicity': cardiotoxicity,
            'mutagenicity': mutagenicity,
            'carcinogenicity': carcinogenicity
        }
        
    def _calculate_overall_score(self, absorption: Dict, distribution: Dict,
                                metabolism: Dict, excretion: Dict, 
                                toxicity: Dict) -> float:
        """Calculate overall ADMET score."""
        # Weight different categories
        weights = {
            'absorption': 0.3,
            'distribution': 0.2,
            'metabolism': 0.2,
            'excretion': 0.1,
            'toxicity': 0.2
        }
        
        # Calculate category scores
        absorption_score = (absorption['bioavailability'] + 
                           self._normalize_score(absorption['caco2_permeability'], -2, 2) +
                           self._normalize_score(absorption['pampa_permeability'], -2, 2)) / 3
        
        distribution_score = (1 - distribution['plasma_protein_binding'] +
                             self._normalize_score(distribution['volume_distribution'], 0.1, 10) +
                             self._normalize_score(distribution['blood_brain_barrier'], -2, 2)) / 3
        
        metabolism_score = (1 - metabolism['cytochrome_inhibition'] +
                           metabolism['metabolic_stability']) / 2
        
        excretion_score = (1 - self._normalize_score(excretion['clearance'], 0.1, 10) +
                          self._normalize_score(excretion['half_life'], 0.1, 24)) / 2
        
        toxicity_score = 1 - toxicity['toxicity_score']
        
        # Calculate weighted overall score
        overall_score = (weights['absorption'] * absorption_score +
                       weights['distribution'] * distribution_score +
                       weights['metabolism'] * metabolism_score +
                       weights['excretion'] * excretion_score +
                       weights['toxicity'] * toxicity_score)
        
        return np.clip(overall_score, 0.0, 1.0)
        
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        return (value - min_val) / (max_val - min_val)
        
    def predict_batch(self, molecules: List[Chem.Mol]) -> List[Dict[str, float]]:
        """
        Predict ADMET properties for a batch of molecules.
        
        Args:
            molecules: List of RDKit molecules
            
        Returns:
            List of ADMET property dictionaries
        """
        return [self.predict_properties(mol) for mol in molecules]
        
    def filter_by_admet(self, molecules: List[Chem.Mol], 
                       min_score: float = 0.5) -> List[Chem.Mol]:
        """
        Filter molecules by ADMET score.
        
        Args:
            molecules: List of molecules to filter
            min_score: Minimum ADMET score threshold
            
        Returns:
            Filtered list of molecules
        """
        filtered_molecules = []
        
        for mol in molecules:
            if mol is not None:
                admet_props = self.predict_properties(mol)
                if admet_props['overall_score'] >= min_score:
                    filtered_molecules.append(mol)
                    
        return filtered_molecules
        
    def get_property_summary(self, molecules: List[Chem.Mol]) -> Dict:
        """
        Get summary statistics of ADMET properties for a set of molecules.
        
        Args:
            molecules: List of molecules
            
        Returns:
            Summary statistics dictionary
        """
        if not molecules:
            return {}
            
        # Predict properties for all molecules
        all_properties = self.predict_batch(molecules)
        
        # Calculate statistics
        summary = {}
        for prop_name in self.property_thresholds.keys():
            values = [props.get(prop_name, 0.0) for props in all_properties]
            if values:
                summary[prop_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
                
        return summary
