"""
Multi-Objective Reward Functions

Implements reward functions for multi-objective optimization in drug discovery,
combining multiple objectives into a single reward signal.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from rdkit import Chem
from rdkit.Chem import Descriptors


class MultiObjectiveRewardFunction:
    """
    Multi-objective reward function for drug discovery optimization.
    
    Combines multiple objectives (binding affinity, ADMET, synthetic accessibility,
    selectivity) into a single reward signal using weighted sum or Pareto optimization.
    """
    
    def __init__(self, objectives: List[str], weights: Optional[Dict[str, float]] = None,
                 reward_type: str = 'weighted_sum'):
        """
        Initialize multi-objective reward function.
        
        Args:
            objectives: List of optimization objectives
            weights: Optional weights for each objective
            reward_type: Type of reward combination ('weighted_sum', 'pareto', 'product')
        """
        self.objectives = objectives
        self.reward_type = reward_type
        
        # Default weights if not provided
        if weights is None:
            self.weights = {obj: 1.0 for obj in objectives}
        else:
            self.weights = weights
            
        # Objective functions
        self.objective_functions = {
            'binding_affinity': self._calculate_binding_affinity,
            'admet_score': self._calculate_admet_score,
            'synthetic_accessibility': self._calculate_synthetic_accessibility,
            'selectivity': self._calculate_selectivity,
            'drug_likeness': self._calculate_drug_likeness,
            'novelty': self._calculate_novelty,
            'diversity': self._calculate_diversity
        }
        
        # Reward statistics
        self.reward_stats = {
            'total_rewards': [],
            'objective_rewards': {obj: [] for obj in objectives},
            'reward_components': []
        }
        
    def calculate_reward(self, molecule: Chem.Mol, target_info: Dict,
                        reference_molecules: List[Chem.Mol] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward for a molecule.
        
        Args:
            molecule: RDKit molecule object
            target_info: Target-specific information
            reference_molecules: Reference molecules for novelty/diversity calculation
            
        Returns:
            Total reward and individual objective scores
        """
        if molecule is None:
            return 0.0, {obj: 0.0 for obj in self.objectives}
            
        # Calculate individual objective scores
        objective_scores = {}
        for obj in self.objectives:
            if obj in self.objective_functions:
                score = self.objective_functions[obj](molecule, target_info, reference_molecules)
                objective_scores[obj] = np.clip(score, 0.0, 1.0)
            else:
                objective_scores[obj] = 0.0
                
        # Combine objectives into total reward
        if self.reward_type == 'weighted_sum':
            total_reward = self._weighted_sum(objective_scores)
        elif self.reward_type == 'pareto':
            total_reward = self._pareto_optimization(objective_scores)
        elif self.reward_type == 'product':
            total_reward = self._product_combination(objective_scores)
        else:
            total_reward = self._weighted_sum(objective_scores)
            
        # Apply bonus for multi-objective balance
        balance_bonus = self._calculate_balance_bonus(objective_scores)
        total_reward += balance_bonus
        
        # Record statistics
        self.reward_stats['total_rewards'].append(total_reward)
        for obj, score in objective_scores.items():
            if obj in self.reward_stats['objective_rewards']:
                self.reward_stats['objective_rewards'][obj].append(score)
                
        return total_reward, objective_scores
        
    def _weighted_sum(self, objective_scores: Dict[str, float]) -> float:
        """Calculate weighted sum of objective scores."""
        total_reward = 0.0
        total_weight = 0.0
        
        for obj, score in objective_scores.items():
            if obj in self.weights:
                total_reward += self.weights[obj] * score
                total_weight += self.weights[obj]
                
        return total_reward / total_weight if total_weight > 0 else 0.0
        
    def _pareto_optimization(self, objective_scores: Dict[str, float]) -> float:
        """Calculate Pareto-optimal reward."""
        # Use minimum of all objectives (conservative approach)
        return min(objective_scores.values()) if objective_scores else 0.0
        
    def _product_combination(self, objective_scores: Dict[str, float]) -> float:
        """Calculate product of objective scores."""
        product = 1.0
        for score in objective_scores.values():
            product *= score
        return product ** (1.0 / len(objective_scores)) if objective_scores else 0.0
        
    def _calculate_balance_bonus(self, objective_scores: Dict[str, float]) -> float:
        """Calculate bonus for balanced objective performance."""
        if len(objective_scores) < 2:
            return 0.0
            
        scores = list(objective_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Bonus for low standard deviation (balanced performance)
        balance_bonus = 0.1 * (1.0 - std_score) * mean_score
        
        return np.clip(balance_bonus, 0.0, 0.1)
        
    def _calculate_binding_affinity(self, molecule: Chem.Mol, target_info: Dict,
                                   reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate binding affinity score."""
        # Simplified binding affinity calculation
        # In practice, would use molecular docking or quantum calculations
        
        # Molecular properties that influence binding
        mw = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        tpsa = Descriptors.TPSA(molecule)
        hbd = Descriptors.NumHDonors(molecule)
        hba = Descriptors.NumHAcceptors(molecule)
        
        # Target-specific considerations
        pocket_volume = target_info.get('binding_pocket_volume', 1000)
        pocket_hydrophobicity = target_info.get('hydrophobicity', 0.5)
        
        # Simple scoring function
        # Optimal molecular weight range: 200-500 Da
        mw_score = 1.0 - abs(mw - 350) / 350
        mw_score = np.clip(mw_score, 0.0, 1.0)
        
        # Optimal LogP range: 1-3
        logp_score = 1.0 - abs(logp - 2) / 2
        logp_score = np.clip(logp_score, 0.0, 1.0)
        
        # Optimal TPSA range: 20-120 Ų
        tpsa_score = 1.0 - abs(tpsa - 70) / 70
        tpsa_score = np.clip(tpsa_score, 0.0, 1.0)
        
        # Hydrogen bond considerations
        hbd_score = 1.0 if hbd <= 5 else 1.0 / hbd
        hba_score = 1.0 if hba <= 10 else 1.0 / hba
        
        # Combine scores
        binding_affinity = (mw_score * 0.3 + logp_score * 0.25 + 
                          tpsa_score * 0.2 + hbd_score * 0.125 + 
                          hba_score * 0.125)
        
        return binding_affinity
        
    def _calculate_admet_score(self, molecule: Chem.Mol, target_info: Dict,
                              reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate ADMET score."""
        # Lipinski's rule of five
        mw = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        hbd = Descriptors.NumHDonors(molecule)
        hba = Descriptors.NumHAcceptors(molecule)
        tpsa = Descriptors.TPSA(molecule)
        
        # Rule of five compliance
        lipinski_score = 0
        if mw <= 500:
            lipinski_score += 0.25
        if logp <= 5:
            lipinski_score += 0.25
        if hbd <= 5:
            lipinski_score += 0.25
        if hba <= 10:
            lipinski_score += 0.25
            
        # Veber's rules (oral bioavailability)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        veber_score = 0
        if rotatable_bonds <= 10:
            veber_score += 0.5
        if tpsa <= 140:
            veber_score += 0.5
            
        # Simple ADMET score
        admet_score = (lipinski_score * 0.6 + veber_score * 0.4)
        
        return admet_score
        
    def _calculate_synthetic_accessibility(self, molecule: Chem.Mol, target_info: Dict,
                                        reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate synthetic accessibility score."""
        # Simplified synthetic accessibility calculation
        # In practice, would use more sophisticated methods
        
        # Molecular complexity factors
        mw = Descriptors.MolWt(molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        aromatic_rings = Descriptors.NumAromaticRings(molecule)
        hetero_atoms = sum(1 for atom in molecule.GetAtoms() 
                          if atom.GetAtomicNum() not in [1, 6])
        
        # Complexity score (lower is better for synthesis)
        complexity = (mw / 500 + rotatable_bonds / 10 + 
                     aromatic_rings / 5 + hetero_atoms / 10)
        
        # Convert to accessibility score (higher is better)
        accessibility = 1.0 / (1.0 + complexity)
        
        return np.clip(accessibility, 0.0, 1.0)
        
    def _calculate_selectivity(self, molecule: Chem.Mol, target_info: Dict,
                              reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate selectivity score."""
        # Simplified selectivity calculation
        # In practice, would use off-target predictions
        
        # Molecular properties that influence selectivity
        logp = Descriptors.MolLogP(molecule)
        tpsa = Descriptors.TPSA(molecule)
        aromatic_rings = Descriptors.NumAromaticRings(molecule)
        
        # Target class considerations
        target_class = target_info.get('target_class', 'enzyme')
        
        # Simple selectivity scoring
        if target_class == 'kinase':
            # Kinases prefer specific properties
            selectivity = 1.0 - abs(logp - 2.5) / 2.5
        elif target_class == 'gpcr':
            # GPCRs prefer different properties
            selectivity = 1.0 - abs(logp - 3.0) / 3.0
        else:
            # General selectivity
            selectivity = 1.0 - abs(logp - 2.0) / 2.0
            
        # Adjust for aromatic content
        aromatic_factor = 1.0 - abs(aromatic_rings - 2) / 4
        selectivity *= aromatic_factor
        
        return np.clip(selectivity, 0.0, 1.0)
        
    def _calculate_drug_likeness(self, molecule: Chem.Mol, target_info: Dict,
                                reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate drug-likeness score."""
        # Combine multiple drug-likeness metrics
        admet_score = self._calculate_admet_score(molecule, target_info)
        sa_score = self._calculate_synthetic_accessibility(molecule, target_info)
        
        # Additional drug-likeness considerations
        mw = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        
        # Drug-likeness score
        drug_likeness = (admet_score * 0.5 + sa_score * 0.3)
        
        # Bonus for optimal molecular weight and LogP
        if 200 <= mw <= 500:
            drug_likeness += 0.1
        if 1 <= logp <= 3:
            drug_likeness += 0.1
            
        return np.clip(drug_likeness, 0.0, 1.0)
        
    def _calculate_novelty(self, molecule: Chem.Mol, target_info: Dict,
                          reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate novelty score compared to reference molecules."""
        if reference_molecules is None or len(reference_molecules) == 0:
            return 0.5  # Default novelty
            
        # Calculate molecular fingerprint
        try:
            from rdkit.Chem import rdMolDescriptors
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
            
            # Compare with reference molecules
            similarities = []
            for ref_mol in reference_molecules:
                if ref_mol is not None:
                    ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        ref_mol, 2, nBits=1024
                    )
                    similarity = Chem.DataStructs.TanimotoSimilarity(fp, ref_fp)
                    similarities.append(similarity)
                    
            if similarities:
                # Novelty is inversely related to maximum similarity
                max_similarity = max(similarities)
                novelty = 1.0 - max_similarity
                return np.clip(novelty, 0.0, 1.0)
            else:
                return 0.5
                
        except:
            return 0.5
            
    def _calculate_diversity(self, molecule: Chem.Mol, target_info: Dict,
                           reference_molecules: List[Chem.Mol] = None) -> float:
        """Calculate diversity score."""
        # Similar to novelty but considers overall diversity
        return self._calculate_novelty(molecule, target_info, reference_molecules)
        
    def update_weights(self, new_weights: Dict[str, float]):
        """Update objective weights."""
        for obj, weight in new_weights.items():
            if obj in self.weights:
                self.weights[obj] = weight
                
    def get_reward_statistics(self) -> Dict:
        """Get reward calculation statistics."""
        return self.reward_stats.copy()
        
    def reset_statistics(self):
        """Reset reward statistics."""
        self.reward_stats = {
            'total_rewards': [],
            'objective_rewards': {obj: [] for obj in self.objectives},
            'reward_components': []
        }
