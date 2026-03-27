"""
Evolution Manager

Implements evolutionary operations for molecular population management
including selection, crossover, mutation, and diversity maintenance.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem


class EvolutionManager:
    """
    Evolution manager for molecular population evolution.
    
    Implements genetic algorithm operations including selection,
    crossover, mutation, and diversity maintenance.
    """
    
    def __init__(self, target, config: Dict):
        """
        Initialize evolution manager.
        
        Args:
            target: Target-specific module
            config: Configuration dictionary
        """
        self.target = target
        self.config = config
        
        # Evolution parameters
        self.mutation_rate = config.get('optimization', {}).get('mutation_rate', 0.1)
        self.crossover_rate = config.get('optimization', {}).get('crossover_rate', 0.7)
        self.selection_pressure = config.get('optimization', {}).get('selection_pressure', 0.8)
        
        # Diversity parameters
        self.diversity_threshold = 0.7
        self.max_similarity = 0.8
        
    def evolve_population(self, population: List[Chem.Mol], 
                         population_size: int) -> List[Chem.Mol]:
        """
        Evolve molecular population using genetic operations.
        
        Args:
            population: Current molecular population
            population_size: Target population size
            
        Returns:
            Evolved population
        """
        if len(population) < 2:
            return population
            
        # Calculate fitness for selection
        fitness_scores = [self._calculate_fitness(mol) for mol in population]
        
        # Selection
        selected_molecules = self._selection(population, fitness_scores)
        
        # Crossover
        offspring_molecules = self._crossover(selected_molecules)
        
        # Mutation
        mutated_molecules = self._mutation(offspring_molecules)
        
        # Diversity maintenance
        diverse_population = self._maintain_diversity(
            population + mutated_molecules, 
            population_size
        )
        
        return diverse_population
        
    def _calculate_fitness(self, mol: Chem.Mol) -> float:
        """Calculate fitness score for molecule."""
        if mol is None:
            return 0.0
            
        try:
            # Use target-specific binding affinity as primary fitness
            binding_affinity = self.target.evaluate_binding_affinity(mol)
            
            # Add basic drug-likeness score
            from rdkit.Chem import Descriptors
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # Lipinski's rule of five
            lipinski_score = 0.0
            if mw <= 500:
                lipinski_score += 0.25
            if logp <= 5:
                lipinski_score += 0.25
            if Descriptors.NumHDonors(mol) <= 5:
                lipinski_score += 0.25
            if Descriptors.NumHAcceptors(mol) <= 10:
                lipinski_score += 0.25
                
            # Combined fitness
            fitness = binding_affinity * 0.7 + lipinski_score * 0.3
            
            return np.clip(fitness, 0.0, 1.0)
            
        except:
            return 0.0
            
    def _selection(self, population: List[Chem.Mol], 
                  fitness_scores: List[float]) -> List[Chem.Mol]:
        """Select molecules using tournament selection."""
        selected = []
        
        # Select 75% of population
        selection_size = max(2, int(len(population) * 0.75))
        
        for _ in range(selection_size):
            # Tournament selection
            tournament_size = max(2, int(len(population) * self.selection_pressure))
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            # Select best from tournament
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx])
            
        return selected
        
    def _crossover(self, population: List[Chem.Mol]) -> List[Chem.Mol]:
        """Perform crossover operations on selected molecules."""
        offspring = []
        
        # Pair molecules for crossover
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            
            if random.random() < self.crossover_rate:
                # Perform crossover
                child1, child2 = self._crossover_molecules(parent1, parent2)
                
                if child1 is not None:
                    offspring.append(child1)
                if child2 is not None:
                    offspring.append(child2)
            else:
                # No crossover, add parents
                offspring.extend([parent1, parent2])
                
        return offspring
        
    def _crossover_molecules(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
        """Perform crossover between two molecules."""
        try:
            # Get SMILES representations
            smiles1 = Chem.MolToSmiles(mol1)
            smiles2 = Chem.MolToSmiles(mol2)
            
            # Find crossover points
            crossover_point1 = random.randint(1, len(smiles1) - 1)
            crossover_point2 = random.randint(1, len(smiles2) - 1)
            
            # Create offspring SMILES
            child1_smiles = smiles1[:crossover_point1] + smiles2[crossover_point2:]
            child2_smiles = smiles2[:crossover_point2] + smiles1[crossover_point1:]
            
            # Convert back to molecules
            child1 = Chem.MolFromSmiles(child1_smiles)
            child2 = Chem.MolFromSmiles(child2_smiles)
            
            # Validate molecules
            valid_child1 = self._validate_molecule(child1)
            valid_child2 = self._validate_molecule(child2)
            
            return valid_child1, valid_child2
            
        except Exception as e:
            print(f"Crossover failed: {e}")
            return mol1, mol2
            
    def _mutation(self, population: List[Chem.Mol]) -> List[Chem.Mol]:
        """Apply mutation operations to molecules."""
        mutated = []
        
        for mol in population:
            if random.random() < self.mutation_rate:
                # Apply mutation
                mutated_mol = self._mutate_molecule(mol)
                if mutated_mol is not None:
                    mutated.append(mutated_mol)
                else:
                    mutated.append(mol)
            else:
                mutated.append(mol)
                
        return mutated
        
    def _mutate_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Apply mutation to a single molecule."""
        try:
            # Get SMILES
            smiles = Chem.MolToSmiles(mol)
            
            # Mutation operations
            mutation_ops = [
                self._add_atom_mutation,
                self._remove_atom_mutation,
                self._change_bond_mutation,
                self._add_functional_group_mutation
            ]
            
            # Apply random mutation
            mutation_op = random.choice(mutation_ops)
            mutated_mol = mutation_op(mol)
            
            return self._validate_molecule(mutated_mol)
            
        except Exception as e:
            print(f"Mutation failed: {e}")
            return mol
            
    def _add_atom_mutation(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add atom mutation."""
        try:
            rw_mol = Chem.RWMol(mol)
            
            # Add carbon atom
            new_atom = Chem.Atom(6)  # Carbon
            atom_idx = rw_mol.AddAtom(new_atom)
            
            # Connect to random existing atom
            if rw_mol.GetNumAtoms() > 1:
                connect_to = random.randint(0, rw_mol.GetNumAtoms() - 2)
                rw_mol.AddBond(connect_to, atom_idx, Chem.BondType.SINGLE)
                
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except:
            return None
            
    def _remove_atom_mutation(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove atom mutation."""
        try:
            if mol.GetNumAtoms() <= 3:
                return None
                
            rw_mol = Chem.RWMol(mol)
            
            # Remove random atom (not the first one)
            atom_to_remove = random.randint(1, rw_mol.GetNumAtoms() - 1)
            rw_mol.RemoveAtom(atom_to_remove)
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except:
            return None
            
    def _change_bond_mutation(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Change bond order mutation."""
        try:
            rw_mol = Chem.RWMol(mol)
            
            bonds = list(rw_mol.GetBonds())
            if len(bonds) == 0:
                return None
                
            # Select random bond
            bond = random.choice(bonds)
            bond_idx = bonds.index(bond)
            
            # Change bond order
            current_order = bond.GetBondType()
            new_orders = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
            new_orders.remove(current_order)
            new_order = random.choice(new_orders)
            
            rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            rw_mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_order)
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except:
            return None
            
    def _add_functional_group_mutation(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add functional group mutation."""
        try:
            # Common functional groups
            functional_groups = ['[OH]', '[NH2]', '[F]', '[Cl]', '[CH3]', '[OCH3]']
            
            group_smarts = random.choice(functional_groups)
            group_mol = Chem.MolFromSmarts(group_smarts)
            
            if group_mol is not None:
                combined = Chem.CombineMols(mol, group_mol)
                return combined
                
        except:
            pass
            
        return None
        
    def _validate_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Validate molecule for chemical correctness."""
        if mol is None:
            return None
            
        try:
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            
            # Check molecular weight
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            
            if mw < 50 or mw > 1000:
                return None
                
            # Check for valid valence
            for atom in mol.GetAtoms():
                if atom.GetExplicitValence() > atom.GetImplicitValence() + atom.GetFormalCharge():
                    return None
                    
            # Optimize geometry
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            
            return mol
            
        except:
            return None
            
    def _maintain_diversity(self, population: List[Chem.Mol], 
                          target_size: int) -> List[Chem.Mol]:
        """Maintain diversity in molecular population."""
        if len(population) <= target_size:
            return population
            
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(population)
        
        # Select diverse molecules
        diverse_molecules = []
        remaining_indices = list(range(len(population)))
        
        while len(diverse_molecules) < target_size and remaining_indices:
            # Select most diverse molecule
            if not diverse_molecules:
                # First molecule: select random
                selected_idx = random.choice(remaining_indices)
            else:
                # Select molecule with minimum similarity to already selected
                selected_idx = self._select_most_diverse(
                    remaining_indices, diverse_molecules, similarity_matrix
                )
                
            diverse_molecules.append(population[selected_idx])
            remaining_indices.remove(selected_idx)
            
        return diverse_molecules
        
    def _calculate_similarity_matrix(self, population: List[Chem.Mol]) -> np.ndarray:
        """Calculate Tanimoto similarity matrix."""
        n = len(population)
        similarity_matrix = np.zeros((n, n))
        
        try:
            from rdkit.Chem import rdMolDescriptors
            from rdkit import DataStructs
            
            # Calculate fingerprints
            fingerprints = []
            for mol in population:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(fp)
                
            # Calculate similarities
            for i in range(n):
                for j in range(i, n):
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
                    
        except Exception as e:
            print(f"Error calculating similarity matrix: {e}")
            # Return identity matrix
            np.fill_diagonal(similarity_matrix, 1.0)
            
        return similarity_matrix
        
    def _select_most_diverse(self, candidate_indices: List[int], 
                           selected_indices: List[int], 
                           similarity_matrix: np.ndarray) -> int:
        """Select most diverse molecule from candidates."""
        best_idx = candidate_indices[0]
        best_score = float('inf')
        
        for candidate_idx in candidate_indices:
            # Calculate average similarity to selected molecules
            similarities = [similarity_matrix[candidate_idx][sel_idx] 
                           for sel_idx in selected_indices]
            avg_similarity = np.mean(similarities)
            
            # Select molecule with minimum average similarity
            if avg_similarity < best_score:
                best_score = avg_similarity
                best_idx = candidate_idx
                
        return best_idx
