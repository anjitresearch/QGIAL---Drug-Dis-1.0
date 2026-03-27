"""
Drug Discovery Environment

Implements the environment for multi-objective reinforcement learning
in drug discovery, providing state representation, action space, and reward calculation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from .reward_functions import MultiObjectiveRewardFunction


class DrugDiscoveryEnvironment:
    """
    Environment for drug discovery reinforcement learning.
    
    Provides the interface between the MODRL agent and the molecular
    optimization problem, handling state representation, actions, and rewards.
    """
    
    def __init__(self, initial_molecule: Chem.Mol, target_info: Dict,
                 objectives: List[str], max_steps: int = 50):
        """
        Initialize drug discovery environment.
        
        Args:
            initial_molecule: Starting molecule
            target_info: Target-specific information
            objectives: List of optimization objectives
            max_steps: Maximum steps per episode
        """
        self.initial_molecule = initial_molecule
        self.current_molecule = initial_molecule
        self.target_info = target_info
        self.objectives = objectives
        self.max_steps = max_steps
        
        # Multi-objective reward function
        self.reward_function = MultiObjectiveRewardFunction(objectives)
        
        # Action space definition
        self.action_space = self._define_action_space()
        
        # State space dimension
        self.state_dim = self._calculate_state_dim()
        
        # Episode tracking
        self.current_step = 0
        self.best_molecule = initial_molecule
        self.best_score = 0.0
        
        # History tracking
        self.history = {
            'molecules': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'objective_scores': []
        }
        
    def _define_action_space(self) -> Dict:
        """Define the action space for molecular modifications."""
        return {
            'add_atom': list(range(1, 9)),  # Add atoms 1-8 (H, C, N, O, etc.)
            'remove_atom': list(range(1, 20)),  # Remove atom indices
            'add_bond': [(i, j) for i in range(10) for j in range(i+1, 10)],  # Add bonds
            'remove_bond': [(i, j) for i in range(10) for j in range(i+1, 10)],  # Remove bonds
            'change_bond_order': [(i, j, order) for i in range(10) for j in range(i+1, 10) 
                                 for order in [1, 2, 3]],  # Change bond order
            'add_ring': list(range(3, 8)),  # Add rings of size 3-7
            'functional_group': ['OH', 'NH2', 'COOH', 'CH3', 'Cl', 'F', 'Br', 'I']  # Add functional groups
        }
        
    def _calculate_state_dim(self) -> int:
        """Calculate state space dimension."""
        # Molecular features: 50 dimensions
        # Target features: 20 dimensions
        # Step progress: 1 dimension
        # Historical performance: 10 dimensions
        return 50 + 20 + 1 + 10
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_molecule = self.initial_molecule
        self.current_step = 0
        self.best_molecule = self.initial_molecule
        self.best_score = 0.0
        
        # Clear history
        self.history = {
            'molecules': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'objective_scores': []
        }
        
        return self._get_state()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            next_state, reward, done, info
        """
        # Execute action on current molecule
        new_molecule, action_success = self._execute_action(action)
        
        if action_success:
            self.current_molecule = new_molecule
            
        # Get current state
        next_state = self._get_state()
        
        # Calculate reward
        reward, objective_scores = self.reward_function.calculate_reward(
            self.current_molecule, self.target_info
        )
        
        # Update best molecule
        total_score = sum(objective_scores.values()) / len(objective_scores)
        if total_score > self.best_score:
            self.best_molecule = self.current_molecule
            self.best_score = total_score
            improved = True
        else:
            improved = False
            
        # Update history
        self.history['molecules'].append(self.current_molecule)
        self.history['states'].append(next_state)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['objective_scores'].append(objective_scores)
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'molecule': self.current_molecule,
            'action_success': action_success,
            'improved': improved,
            'objective_scores': objective_scores,
            'best_score': self.best_score,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
        
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Molecular features (50 dimensions)
        mol_features = self._extract_molecular_features()
        
        # Target features (20 dimensions)
        target_features = self._extract_target_features()
        
        # Step progress (1 dimension)
        step_progress = self.current_step / self.max_steps
        
        # Historical performance (10 dimensions)
        historical_features = self._extract_historical_features()
        
        # Combine all features
        state = np.concatenate([
            mol_features,
            target_features,
            [step_progress],
            historical_features
        ])
        
        return state
        
    def _extract_molecular_features(self) -> np.ndarray:
        """Extract molecular features for state representation."""
        if self.current_molecule is None:
            return np.zeros(50)
            
        # Basic molecular properties (10 dimensions)
        mw = Descriptors.MolWt(self.current_molecule)
        logp = Descriptors.MolLogP(self.current_molecule)
        tpsa = Descriptors.TPSA(self.current_molecule)
        hbd = Descriptors.NumHDonors(self.current_molecule)
        hba = Descriptors.NumHAcceptors(self.current_molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(self.current_molecule)
        aromatic_rings = Descriptors.NumAromaticRings(self.current_molecule)
        saturated_rings = Descriptors.NumSaturatedRings(self.current_molecule)
        formal_charge = Chem.GetFormalCharge(self.current_molecule)
        num_atoms = self.current_molecule.GetNumAtoms()
        
        basic_props = [mw, logp, tpsa, hbd, hba, rotatable_bonds, 
                      aromatic_rings, saturated_rings, formal_charge, num_atoms]
        
        # Extended molecular features (40 dimensions)
        extended_props = self._extract_extended_features()
        
        # Combine and pad to 50 dimensions
        all_features = basic_props + extended_props
        if len(all_features) < 50:
            all_features.extend([0.0] * (50 - len(all_features)))
        else:
            all_features = all_features[:50]
            
        return np.array(all_features)
        
    def _extract_extended_features(self) -> List[float]:
        """Extract extended molecular features."""
        features = []
        
        # Fragment counts (20 dimensions)
        fragments = {
            'aromatic': 0, 'aliphatic': 0, 'hetero_aromatic': 0,
            'carbonyl': 0, 'hydroxyl': 0, 'amine': 0, 'amide': 0,
            'ester': 0, 'ether': 0, 'halogen': 0, 'nitrile': 0,
            'nitro': 0, 'sulfur': 0, 'phosphorus': 0, 'boron': 0,
            'silicon': 0, 'metal': 0, 'other': 0
        }
        
        # Simplified fragment counting
        for atom in self.current_molecule.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 6:  # Carbon
                if atom.GetIsAromatic():
                    fragments['aromatic'] += 1
                else:
                    fragments['aliphatic'] += 1
            elif atomic_num in [7, 8, 16]:  # N, O, S
                if atom.GetIsAromatic():
                    fragments['hetero_aromatic'] += 1
            elif atomic_num in [9, 17, 35, 53]:  # Halogens
                fragments['halogen'] += 1
                
        features.extend(list(fragments.values()))
        
        # Topological features (20 dimensions)
        # Simplified topological descriptors
        try:
            from rdkit.Chem import rdMolDescriptors
            
            # Molecular fingerprints (simplified)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                self.current_molecule, 2, nBits=20
            )
            fp_list = list(fp)
            features.extend(fp_list)
            
        except:
            features.extend([0.0] * 20)
            
        return features
        
    def _extract_target_features(self) -> np.ndarray:
        """Extract target-specific features."""
        features = []
        
        # Target properties (10 dimensions)
        target_props = [
            self.target_info.get('binding_pocket_volume', 0),
            self.target_info.get('hydrophobicity', 0),
            self.target_info.get('polarity', 0),
            self.target_info.get('flexibility', 0),
            self.target_info.get('accessible_surface_area', 0),
            self.target_info.get('electrostatic_potential', 0),
            self.target_info.get('hydrogen_bond_donors', 0),
            self.target_info.get('hydrogen_bond_acceptors', 0),
            self.target_info.get('metal_binding_sites', 0),
            self.target_info.get('allosteric_sites', 0)
        ]
        features.extend(target_props)
        
        # Target class features (10 dimensions)
        target_classes = [
            'kinase', 'gpcr', 'ion_channel', 'nuclear_receptor',
            'enzyme', 'transporter', 'transcription_factor',
            'viral_protein', 'bacterial_protein', 'other'
        ]
        
        target_class = self.target_info.get('target_class', 'other')
        class_encoding = [1.0 if cls == target_class else 0.0 for cls in target_classes]
        features.extend(class_encoding)
        
        return np.array(features)
        
    def _extract_historical_features(self) -> np.ndarray:
        """Extract historical performance features."""
        if len(self.history['rewards']) == 0:
            return np.zeros(10)
            
        # Recent performance metrics (10 dimensions)
        recent_rewards = self.history['rewards'][-10:]
        if len(recent_rewards) < 10:
            recent_rewards.extend([0.0] * (10 - len(recent_rewards)))
            
        return np.array(recent_rewards)
        
    def _execute_action(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Execute action on current molecule."""
        try:
            # Convert action to discrete action type
            action_type_idx = int(np.argmax(action[:len(self.action_space)]))
            action_types = list(self.action_space.keys())
            action_type = action_types[action_type_idx]
            
            # Execute specific action
            if action_type == 'add_atom':
                return self._add_atom(action)
            elif action_type == 'remove_atom':
                return self._remove_atom(action)
            elif action_type == 'add_bond':
                return self._add_bond(action)
            elif action_type == 'remove_bond':
                return self._remove_bond(action)
            elif action_type == 'change_bond_order':
                return self._change_bond_order(action)
            elif action_type == 'add_ring':
                return self._add_ring(action)
            elif action_type == 'functional_group':
                return self._add_functional_group(action)
            else:
                return self.current_molecule, False
                
        except Exception as e:
            print(f"Action execution failed: {e}")
            return self.current_molecule, False
            
    def _add_atom(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Add atom to molecule."""
        # Simplified atom addition
        rw_mol = Chem.RWMol(self.current_molecule)
        
        # Get atom type from action
        atom_types = [6, 7, 8, 9, 17, 35, 53, 16]  # C, N, O, F, Cl, Br, I, S
        atom_type_idx = int(action[len(self.action_space)]) % len(atom_types)
        atom_type = atom_types[atom_type_idx]
        
        # Add atom
        new_atom = Chem.Atom(atom_type)
        rw_mol.AddAtom(new_atom)
        
        # Try to connect to existing atom
        if rw_mol.GetNumAtoms() > 1:
            connect_to = np.random.randint(0, rw_mol.GetNumAtoms() - 1)
            rw_mol.AddBond(connect_to, rw_mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
            
        # Convert back to Mol
        try:
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol, True
        except:
            return self.current_molecule, False
            
    def _remove_atom(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Remove atom from molecule."""
        if self.current_molecule.GetNumAtoms() <= 1:
            return self.current_molecule, False
            
        rw_mol = Chem.RWMol(self.current_molecule)
        
        # Get atom to remove
        atom_idx = int(action[len(self.action_space)]) % rw_mol.GetNumAtoms()
        rw_mol.RemoveAtom(atom_idx)
        
        try:
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol, True
        except:
            return self.current_molecule, False
            
    def _add_bond(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Add bond to molecule."""
        rw_mol = Chem.RWMol(self.current_molecule)
        
        if rw_mol.GetNumAtoms() < 2:
            return self.current_molecule, False
            
        # Get atoms to connect
        atom1_idx = int(action[len(self.action_space)]) % rw_mol.GetNumAtoms()
        atom2_idx = int(action[len(self.action_space) + 1]) % rw_mol.GetNumAtoms()
        
        if atom1_idx == atom2_idx:
            return self.current_molecule, False
            
        # Check if bond already exists
        existing_bond = rw_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if existing_bond is None:
            rw_mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
            
            try:
                new_mol = rw_mol.GetMol()
                Chem.SanitizeMol(new_mol)
                return new_mol, True
            except:
                return self.current_molecule, False
        else:
            return self.current_molecule, False
            
    def _remove_bond(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Remove bond from molecule."""
        rw_mol = Chem.RWMol(self.current_molecule)
        
        # Get bond to remove
        bonds = list(rw_mol.GetBonds())
        if len(bonds) == 0:
            return self.current_molecule, False
            
        bond_idx = int(action[len(self.action_space)]) % len(bonds)
        bond = bonds[bond_idx]
        
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
        try:
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol, True
        except:
            return self.current_molecule, False
            
    def _change_bond_order(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Change bond order."""
        rw_mol = Chem.RWMol(self.current_molecule)
        
        bonds = list(rw_mol.GetBonds())
        if len(bonds) == 0:
            return self.current_molecule, False
            
        # Get bond to modify
        bond_idx = int(action[len(self.action_space)]) % len(bonds)
        bond = bonds[bond_idx]
        
        # Get new bond order
        bond_orders = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
        order_idx = int(action[len(self.action_space) + 1]) % len(bond_orders)
        new_order = bond_orders[order_idx]
        
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        rw_mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_order)
        
        try:
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol, True
        except:
            return self.current_molecule, False
            
    def _add_ring(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Add ring to molecule."""
        # Simplified ring addition
        return self.current_molecule, False
        
    def _add_functional_group(self, action: np.ndarray) -> Tuple[Optional[Chem.Mol], bool]:
        """Add functional group to molecule."""
        # Simplified functional group addition
        return self.current_molecule, False
        
    def get_best_molecule(self) -> Chem.Mol:
        """Get best molecule found during episode."""
        return self.best_molecule
        
    def get_history(self) -> Dict:
        """Get episode history."""
        return self.history.copy()
