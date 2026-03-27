"""
Optimization Loop

Implements the core optimization loop that coordinates HQGAN and DRL agents
for iterative molecular design and optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from ..generative.hqgan import HybridQuantumGAN
from ..reinforcement.modrl_agent import MultiObjectiveDRLAgent


class OptimizationLoop:
    """
    Core optimization loop that coordinates generative modeling and
    reinforcement learning for molecular optimization.
    """
    
    def __init__(self, target, hqgan: HybridQuantumGAN, 
                 drl_agent: MultiObjectiveDRLAgent, config: Dict):
        """
        Initialize optimization loop.
        
        Args:
            target: Target-specific module
            hqgan: Hybrid quantum-classical GAN
            drl_agent: Multi-objective DRL agent
            config: Configuration dictionary
        """
        self.target = target
        self.hqgan = hqgan
        self.drl_agent = drl_agent
        self.config = config
        
        # Optimization parameters
        self.max_iterations = config.get('reinforcement', {}).get('max_iterations', 100)
        self.population_size = config.get('optimization', {}).get('population_size', 100)
        
        # Performance tracking
        self.optimization_history = []
        self.best_fitness = 0.0
        self.best_molecules = []
        
    def generate_new_molecules(self, elite_molecules: List[Chem.Mol], 
                              population_size: int) -> List[Chem.Mol]:
        """
        Generate new molecules using HQGAN trained on elite molecules.
        
        Args:
            elite_molecules: List of elite molecules from previous generation
            population_size: Number of molecules to generate
            
        Returns:
            List of newly generated molecules
        """
        if len(elite_molecules) < 5:
            # Not enough elite molecules, generate random molecules
            return self._generate_random_molecules(population_size)
            
        # Train HQGAN on elite molecules
        print(f"  🎨 Training HQGAN on {len(elite_molecules)} elite molecules...")
        self.hqgan.train(elite_molecules, epochs=20, batch_size=8)
        
        # Generate new molecules
        print(f"  🌟 Generating {population_size} new molecules...")
        new_molecules = self.hqgan.generate_molecules(population_size)
        
        # Filter valid molecules
        valid_molecules = [mol for mol in new_molecules if mol is not None]
        
        print(f"  ✓ Generated {len(valid_molecules)} valid molecules")
        
        return valid_molecules
        
    def _generate_random_molecules(self, n_molecules: int) -> List[Chem.Mol]:
        """Generate random molecules when elite pool is insufficient."""
        molecules = []
        
        for _ in range(n_molecules):
            # Use target-specific design
            mol = self.target.design_molecules(1)
            if mol:
                molecules.extend(mol)
                
        return molecules[:n_molecules]
        
    def optimize_molecules(self, molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        """
        Optimize molecules using DRL agent.
        
        Args:
            molecules: List of molecules to optimize
            
        Returns:
            List of optimized molecules
        """
        optimized_molecules = []
        
        print(f"  🤖 Optimizing {len(molecules)} molecules with DRL agent...")
        
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
                
            try:
                # Get target information
                target_info = self.target.get_target_info()
                
                # Optimize molecule
                optimized_mol, history = self.drl_agent.optimize_molecule(
                    mol, target_info, max_iterations=10
                )
                
                if optimized_mol is not None:
                    optimized_molecules.append(optimized_mol)
                    
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i + 1}/{len(molecules)} molecules")
                    
            except Exception as e:
                print(f"    Optimization failed for molecule {i}: {e}")
                optimized_molecules.append(mol)
                
        print(f"  ✓ Optimized {len(optimized_molecules)} molecules")
        
        return optimized_molecules
        
    def evaluate_molecules(self, molecules: List[Chem.Mol]) -> List[Dict]:
        """
        Evaluate molecules and return fitness scores.
        
        Args:
            molecules: List of molecules to evaluate
            
        Returns:
            List of evaluation results
        """
        evaluations = []
        
        for mol in molecules:
            if mol is None:
                continue
                
            try:
                # Evaluate binding affinity
                binding_affinity = self.target.evaluate_binding_affinity(mol)
                
                # Calculate fitness
                fitness = self._calculate_fitness(mol, binding_affinity)
                
                evaluations.append({
                    'molecule': mol,
                    'smiles': Chem.MolToSmiles(mol, canonical=True),
                    'fitness': fitness,
                    'binding_affinity': binding_affinity
                })
                
            except Exception as e:
                print(f"    Evaluation failed: {e}")
                continue
                
        return evaluations
        
    def _calculate_fitness(self, mol: Chem.Mol, binding_affinity: float) -> float:
        """Calculate fitness score for molecule."""
        # Simple fitness calculation combining binding affinity and basic properties
        from rdkit.Chem import Descriptors
        
        # Basic molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Property-based scoring
        mw_score = 1.0 if 200 <= mw <= 600 else 0.5
        logp_score = 1.0 if 1 <= logp <= 4 else 0.5
        
        # Combined fitness
        fitness = (binding_affinity * 0.6 + mw_score * 0.2 + logp_score * 0.2)
        
        return np.clip(fitness, 0.0, 1.0)
        
    def select_elite(self, evaluations: List[Dict], elite_size: int) -> List[Chem.Mol]:
        """Select elite molecules from evaluations."""
        # Sort by fitness
        sorted_evals = sorted(evaluations, key=lambda x: x['fitness'], reverse=True)
        
        # Select elite molecules
        elite_molecules = [eval_['molecule'] for eval_ in sorted_evals[:elite_size]]
        
        # Update best molecules
        if sorted_evals:
            best_fitness = sorted_evals[0]['fitness']
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_molecules = [eval_ for eval_ in sorted_evals[:5]]
                
        return elite_molecules
        
    def run_iteration(self, population: List[Chem.Mol]) -> Tuple[List[Chem.Mol], Dict]:
        """
        Run one optimization iteration.
        
        Args:
            population: Current molecular population
            
        Returns:
            New population and iteration statistics
        """
        # Evaluate current population
        evaluations = self.evaluate_molecules(population)
        
        # Select elite molecules
        elite_size = max(5, len(population) // 4)
        elite_molecules = self.select_elite(evaluations, elite_size)
        
        # Generate new molecules
        new_molecules = self.generate_new_molecules(
            elite_molecules, 
            population_size=len(population) // 2
        )
        
        # Optimize molecules
        optimized_molecules = self.optimize_molecules(new_molecules)
        
        # Create new population
        new_population = elite_molecules + optimized_molecules
        
        # Calculate statistics
        iteration_stats = {
            'population_size': len(population),
            'elite_size': len(elite_molecules),
            'new_molecules': len(new_molecules),
            'optimized_molecules': len(optimized_molecules),
            'avg_fitness': np.mean([eval_['fitness'] for eval_ in evaluations]),
            'max_fitness': max([eval_['fitness'] for eval_ in evaluations]) if evaluations else 0.0,
            'best_overall_fitness': self.best_fitness
        }
        
        # Track history
        self.optimization_history.append(iteration_stats)
        
        return new_population, iteration_stats
        
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self.optimization_history.copy()
        
    def get_best_molecules(self, n_molecules: int = 5) -> List[Dict]:
        """Get best molecules found during optimization."""
        return self.best_molecules[:n_molecules]
        
    def reset(self):
        """Reset optimization state."""
        self.optimization_history = []
        self.best_fitness = 0.0
        self.best_molecules = []
