"""
QGIAL Main Pipeline

Implements the complete QGIAL (Quantum Generative Intelligence and Adaptive Learning)
pipeline that integrates quantum simulation, generative modeling, reinforcement learning,
and target-specific optimization for autonomous drug discovery.
"""

import os
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
import matplotlib.pyplot as plt
import pandas as pd

# Import QGIAL components
from ..quantum.vqc_molecular import VariationalQuantumCircuit
from ..quantum.quantum_descriptors import QuantumDescriptorCalculator
from ..generative.hqgan import HybridQuantumGAN
from ..reinforcement.modrl_agent import MultiObjectiveDRLAgent
from ..targets.kras_g12d import KRASG12DTarget
from ..targets.pd_l1 import PDL1Target
from ..targets.sars_cov_2_mpro import SARSCoV2MProTarget
from ..properties.admet_predictor import ADMETPredictor
from ..properties.pains_detector import PAINSDetector
from .optimization_loop import OptimizationLoop
from .evolution_manager import EvolutionManager
from .performance_tracker import PerformanceTracker


class QGIALPipeline:
    """
    Main QGIAL pipeline for autonomous drug discovery.
    
    Integrates all components into a unified framework for:
    - Quantum molecular simulation
    - Hybrid quantum-classical generative modeling
    - Multi-objective reinforcement learning optimization
    - Target-specific molecular design
    - ADMET prediction and PAINS detection
    - Closed-loop evolutionary optimization
    """
    
    def __init__(self, target_name: str = 'KRAS_G12D', config: Optional[Dict] = None):
        """
        Initialize QGIAL pipeline.
        
        Args:
            target_name: Name of target protein ('KRAS_G12D', 'PD_L1', 'SARS_COV_2_MPRO')
            config: Configuration dictionary
        """
        self.target_name = target_name
        self.config = config or self._get_default_config()
        
        # Initialize target-specific module
        self.target = self._initialize_target()
        
        # Initialize core components
        self.vqc = VariationalQuantumCircuit(
            n_qubits=self.config['quantum']['n_qubits'],
            backend=self.config['quantum']['backend']
        )
        
        self.descriptor_calc = QuantumDescriptorCalculator(
            n_qubits=self.config['quantum']['n_qubits']
        )
        
        self.hqgan = HybridQuantumGAN(
            latent_dim=self.config['generative']['latent_dim'],
            n_qubits=self.config['quantum']['n_qubits'],
            learning_rate=self.config['generative']['learning_rate']
        )
        
        self.drl_agent = MultiObjectiveDRLAgent(
            state_dim=self.config['reinforcement']['state_dim'],
            action_dim=self.config['reinforcement']['action_dim'],
            objectives=self.config['reinforcement']['objectives'],
            learning_rate=self.config['reinforcement']['learning_rate']
        )
        
        self.admet_predictor = ADMETPredictor()
        self.pains_detector = PAINSDetector()
        
        # Initialize pipeline components
        self.optimization_loop = OptimizationLoop(
            target=self.target,
            hqgan=self.hqgan,
            drl_agent=self.drl_agent,
            config=self.config
        )
        
        self.evolution_manager = EvolutionManager(
            target=self.target,
            config=self.config
        )
        
        self.performance_tracker = PerformanceTracker()
        
        # Pipeline state
        self.current_generation = 0
        self.best_molecules = []
        self.optimization_history = []
        self.pipeline_stats = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for QGIAL pipeline."""
        return {
            'quantum': {
                'n_qubits': 27,
                'backend': 'ibmq_falcon',
                'shots': 1000
            },
            'generative': {
                'latent_dim': 100,
                'learning_rate': 0.0002,
                'epochs': 1000,
                'batch_size': 32
            },
            'reinforcement': {
                'state_dim': 81,
                'action_dim': 20,
                'objectives': ['binding_affinity', 'admet_score', 'synthetic_accessibility', 'selectivity'],
                'learning_rate': 0.0003,
                'max_iterations': 100
            },
            'optimization': {
                'max_generations': 96,
                'population_size': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'selection_pressure': 0.8
            },
            'filtering': {
                'min_admet_score': 0.5,
                'max_pains_score': 0.1,
                'min_binding_affinity': 0.3
            }
        }
        
    def _initialize_target(self):
        """Initialize target-specific module."""
        target_map = {
            'KRAS_G12D': KRASG12DTarget,
            'PD_L1': PDL1Target,
            'SARS_COV_2_MPRO': SARSCoV2MProTarget
        }
        
        if self.target_name not in target_map:
            raise ValueError(f"Unknown target: {self.target_name}")
            
        return target_map[self.target_name]()
        
    def run_pipeline(self, max_generations: Optional[int] = None) -> Dict:
        """
        Run the complete QGIAL pipeline.
        
        Args:
            max_generations: Maximum number of evolutionary generations
            
        Returns:
            Pipeline results and statistics
        """
        print(f"🚀 Starting QGIAL Pipeline for {self.target_name}")
        print("=" * 60)
        
        max_gen = max_generations or self.config['optimization']['max_generations']
        
        # Initialize population
        print("📊 Initializing molecular population...")
        initial_population = self._initialize_population()
        
        # Run evolutionary optimization
        print(f"🧬 Running evolutionary optimization for {max_gen} generations...")
        results = self._run_evolutionary_optimization(initial_population, max_gen)
        
        # Final evaluation and ranking
        print("🏆 Final evaluation and ranking...")
        final_results = self._final_evaluation(results)
        
        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        report = self._generate_report(final_results)
        
        print("✅ QGIAL Pipeline completed successfully!")
        return report
        
    def _initialize_population(self) -> List[Chem.Mol]:
        """Initialize initial molecular population."""
        population_size = self.config['optimization']['population_size']
        
        # Generate initial molecules using target-specific design
        target_molecules = self.target.design_molecules(population_size // 2)
        
        # Generate additional molecules using HQGAN
        if len(target_molecules) < population_size:
            additional_needed = population_size - len(target_molecules)
            
            # Train HQGAN briefly on target molecules
            if len(target_molecules) > 5:
                self.hqgan.train(target_molecules, epochs=10, batch_size=4)
                
            # Generate additional molecules
            generated_molecules = self.hqgan.generate_molecules(additional_needed)
            target_molecules.extend(generated_molecules)
            
        # Filter and validate population
        valid_population = []
        for mol in target_molecules:
            if mol is not None:
                # Check for PAINS
                pains_result = self.pains_detector.detect_pains(mol)
                if not pains_result['is_pains'] or pains_result['pains_score'] < 0.3:
                    valid_population.append(mol)
                    
        return valid_population[:population_size]
        
    def _run_evolutionary_optimization(self, initial_population: List[Chem.Mol], 
                                     max_generations: int) -> Dict:
        """Run evolutionary optimization loop."""
        population = initial_population.copy()
        generation_results = []
        
        for generation in range(max_generations):
            self.current_generation = generation
            
            print(f"Generation {generation + 1}/{max_generations}")
            
            # Evaluate current population
            evaluated_population = self._evaluate_population(population)
            
            # Select best performers
            elite_molecules = self._select_elite(evaluated_population)
            
            # Generate new molecules using HQGAN
            new_molecules = self.optimization_loop.generate_new_molecules(
                elite_molecules, population_size=len(population) // 2
            )
            
            # Optimize molecules using DRL agent
            optimized_molecules = self._optimize_with_drl(new_molecules)
            
            # Apply evolutionary operations
            population = self.evolution_manager.evolve_population(
                elite_molecules + optimized_molecules,
                population_size=len(population)
            )
            
            # Track performance
            gen_stats = self._track_generation_performance(evaluated_population)
            generation_results.append(gen_stats)
            
            # Update best molecules
            self._update_best_molecules(evaluated_population)
            
            # Print progress
            if generation % 10 == 0:
                self._print_generation_summary(generation, gen_stats)
                
        return {
            'final_population': population,
            'generation_results': generation_results,
            'best_molecules': self.best_molecules
        }
        
    def _evaluate_population(self, population: List[Chem.Mol]) -> List[Dict]:
        """Evaluate molecular population."""
        evaluated = []
        
        for mol in population:
            if mol is None:
                continue
                
            # Calculate quantum descriptors
            quantum_desc = self.descriptor_calc.calculate_all_descriptors(mol)
            
            # Evaluate binding affinity
            binding_affinity = self.target.evaluate_binding_affinity(mol)
            
            # Predict ADMET properties
            admet_props = self.admet_predictor.predict_properties(mol)
            
            # Detect PAINS
            pains_result = self.pains_detector.detect_pains(mol)
            
            # Calculate overall fitness
            fitness = self._calculate_fitness(
                binding_affinity, admet_props, pains_result, quantum_desc
            )
            
            evaluated.append({
                'molecule': mol,
                'smiles': Chem.MolToSmiles(mol, canonical=True),
                'quantum_descriptors': quantum_desc,
                'binding_affinity': binding_affinity,
                'admet_properties': admet_props,
                'pains_result': pains_result,
                'fitness': fitness
            })
            
        return evaluated
        
    def _calculate_fitness(self, binding_affinity: float, admet_props: Dict,
                          pains_result: Dict, quantum_desc: Dict) -> float:
        """Calculate overall fitness score."""
        # Weight different components
        weights = {
            'binding_affinity': 0.3,
            'admet_score': 0.25,
            'pains_penalty': 0.2,
            'quantum_score': 0.15,
            'drug_likeness': 0.1
        }
        
        # Component scores
        binding_score = binding_affinity
        admet_score = admet_props.get('overall_score', 0.0)
        pains_score = 1.0 - pains_result.get('pains_score', 0.0)
        quantum_score = quantum_desc.get('quantum_fidelity', 0.5)
        drug_score = quantum_desc.get('quantum_drug_score', 0.5)
        
        # Calculate weighted fitness
        fitness = (
            weights['binding_affinity'] * binding_score +
            weights['admet_score'] * admet_score +
            weights['pains_penalty'] * pains_score +
            weights['quantum_score'] * quantum_score +
            weights['drug_likeness'] * drug_score
        )
        
        return np.clip(fitness, 0.0, 1.0)
        
    def _select_elite(self, evaluated_population: List[Dict]) -> List[Chem.Mol]:
        """Select elite molecules from population."""
        # Sort by fitness
        sorted_population = sorted(
            evaluated_population, 
            key=lambda x: x['fitness'], 
            reverse=True
        )
        
        # Select top performers
        elite_size = max(5, len(evaluated_population) // 4)
        elite_molecules = [item['molecule'] for item in sorted_population[:elite_size]]
        
        return elite_molecules
        
    def _optimize_with_drl(self, molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        """Optimize molecules using DRL agent."""
        optimized_molecules = []
        
        for mol in molecules:
            if mol is None:
                continue
                
            try:
                # Get target info
                target_info = self.target.get_target_info()
                
                # Optimize molecule
                optimized_mol, history = self.drl_agent.optimize_molecule(
                    mol, target_info, max_iterations=20
                )
                
                if optimized_mol is not None:
                    optimized_molecules.append(optimized_mol)
                    
            except Exception as e:
                print(f"DRL optimization failed: {e}")
                optimized_molecules.append(mol)
                
        return optimized_molecules
        
    def _track_generation_performance(self, evaluated_population: List[Dict]) -> Dict:
        """Track performance metrics for current generation."""
        if not evaluated_population:
            return {}
            
        # Extract metrics
        fitness_scores = [item['fitness'] for item in evaluated_population]
        binding_scores = [item['binding_affinity'] for item in evaluated_population]
        admet_scores = [item['admet_properties'].get('overall_score', 0) 
                       for item in evaluated_population]
        
        # Calculate statistics
        stats = {
            'generation': self.current_generation,
            'population_size': len(evaluated_population),
            'avg_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'avg_binding_affinity': np.mean(binding_scores),
            'max_binding_affinity': np.max(binding_scores),
            'avg_admet_score': np.mean(admet_scores),
            'max_admet_score': np.max(admet_scores),
            'pains_free_rate': sum(1 for item in evaluated_population 
                                 if not item['pains_result']['is_pains']) / len(evaluated_population)
        }
        
        # Track performance
        self.performance_tracker.update(stats)
        
        return stats
        
    def _update_best_molecules(self, evaluated_population: List[Dict]):
        """Update best molecules list."""
        # Sort by fitness
        sorted_population = sorted(
            evaluated_population,
            key=lambda x: x['fitness'],
            reverse=True
        )
        
        # Keep top 10 molecules
        self.best_molecules = sorted_population[:10]
        
    def _print_generation_summary(self, generation: int, stats: Dict):
        """Print generation summary."""
        print(f"  📊 Generation {generation + 1} Summary:")
        print(f"    Avg Fitness: {stats['avg_fitness']:.3f}")
        print(f"    Max Fitness: {stats['max_fitness']:.3f}")
        print(f"    Avg Binding: {stats['avg_binding_affinity']:.3f}")
        print(f"    Max Binding: {stats['max_binding_affinity']:.3f}")
        print(f"    PAINS-Free: {stats['pains_free_rate']:.1%}")
        
    def _final_evaluation(self, results: Dict) -> Dict:
        """Perform final evaluation of results."""
        final_population = results['final_population']
        
        # Comprehensive evaluation
        final_evaluated = self._evaluate_population(final_population)
        
        # Rank molecules
        ranked_molecules = sorted(final_evaluated, key=lambda x: x['fitness'], reverse=True)
        
        # Generate final statistics
        final_stats = {
            'total_molecules_generated': len(final_evaluated),
            'top_molecules': ranked_molecules[:20],
            'performance_summary': self.performance_tracker.get_summary(),
            'optimization_efficiency': self._calculate_optimization_efficiency(results),
            'target_specific_metrics': self._calculate_target_metrics(ranked_molecules)
        }
        
        return final_stats
        
    def _calculate_optimization_efficiency(self, results: Dict) -> Dict:
        """Calculate optimization efficiency metrics."""
        generation_results = results['generation_results']
        
        if not generation_results:
            return {}
            
        # Calculate improvement over time
        initial_fitness = generation_results[0]['avg_fitness']
        final_fitness = generation_results[-1]['avg_fitness']
        improvement = (final_fitness - initial_fitness) / initial_fitness if initial_fitness > 0 else 0
        
        # Calculate convergence rate
        convergence_gen = self._find_convergence_generation(generation_results)
        
        return {
            'fitness_improvement': improvement,
            'convergence_generation': convergence_gen,
            'generations_to_convergence': convergence_gen + 1,
            'optimization_rate': improvement / (convergence_gen + 1) if convergence_gen >= 0 else 0
        }
        
    def _find_convergence_generation(self, generation_results: List[Dict]) -> int:
        """Find generation where optimization converged."""
        if len(generation_results) < 10:
            return len(generation_results) - 1
            
        # Check for plateau in fitness improvement
        recent_fitness = [gen['max_fitness'] for gen in generation_results[-10:]]
        fitness_std = np.std(recent_fitness)
        
        if fitness_std < 0.01:  # Convergence threshold
            return len(generation_results) - 10
            
        return len(generation_results) - 1
        
    def _calculate_target_metrics(self, ranked_molecules: List[Dict]) -> Dict:
        """Calculate target-specific metrics."""
        if not ranked_molecules:
            return {}
            
        top_molecules = ranked_molecules[:10]
        
        # Calculate target-specific statistics
        binding_scores = [mol['binding_affinity'] for mol in top_molecules]
        admet_scores = [mol['admet_properties'].get('overall_score', 0) for mol in top_molecules]
        
        return {
            'avg_top_binding': np.mean(binding_scores),
            'max_top_binding': np.max(binding_scores),
            'avg_top_admet': np.mean(admet_scores),
            'max_top_admet': np.max(admet_scores),
            'sub_nanomolar_count': sum(1 for score in binding_scores if score > 0.8),
            'favorable_admet_count': sum(1 for score in admet_scores if score > 0.7)
        }
        
    def _generate_report(self, final_results: Dict) -> Dict:
        """Generate comprehensive pipeline report."""
        report = {
            'pipeline_info': {
                'target': self.target_name,
                'config': self.config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': final_results,
            'best_molecules': [
                {
                    'rank': i + 1,
                    'smiles': mol['smiles'],
                    'fitness': mol['fitness'],
                    'binding_affinity': mol['binding_affinity'],
                    'admet_score': mol['admet_properties'].get('overall_score', 0),
                    'pains_score': mol['pains_result'].get('pains_score', 0)
                }
                for i, mol in enumerate(final_results['top_molecules'])
            ],
            'performance_metrics': final_results['performance_summary'],
            'optimization_efficiency': final_results['optimization_efficiency'],
            'target_specific_metrics': final_results['target_specific_metrics']
        }
        
        # Save report
        self._save_report(report)
        
        return report
        
    def _save_report(self, report: Dict):
        """Save pipeline report to file."""
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'data', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        import json
        report_file = os.path.join(output_dir, f'qgial_report_{self.target_name}_{int(time.time())}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save CSV summary
        if 'best_molecules' in report:
            df = pd.DataFrame(report['best_molecules'])
            csv_file = os.path.join(output_dir, f'qgial_summary_{self.target_name}_{int(time.time())}.csv')
            df.to_csv(csv_file, index=False)
            
        print(f"📄 Report saved to: {report_file}")
        print(f"📊 Summary saved to: {csv_file}")
        
    def get_best_molecules(self, n_molecules: int = 10) -> List[Dict]:
        """Get best molecules from pipeline run."""
        if not self.best_molecules:
            return []
            
        return self.best_molecules[:n_molecules]
        
    def visualize_optimization_progress(self):
        """Visualize optimization progress."""
        self.performance_tracker.plot_progress()
        
    def save_pipeline_state(self, filepath: str):
        """Save complete pipeline state."""
        state = {
            'target_name': self.target_name,
            'config': self.config,
            'current_generation': self.current_generation,
            'best_molecules': self.best_molecules,
            'performance_history': self.performance_tracker.get_history()
        }
        
        torch.save(state, filepath)
        print(f"💾 Pipeline state saved to: {filepath}")
        
    def load_pipeline_state(self, filepath: str):
        """Load pipeline state from file."""
        state = torch.load(filepath, map_location='cpu')
        
        self.target_name = state['target_name']
        self.config = state['config']
        self.current_generation = state['current_generation']
        self.best_molecules = state['best_molecules']
        
        # Restore performance tracker
        self.performance_tracker.load_history(state['performance_history'])
        
        print(f"📂 Pipeline state loaded from: {filepath}")
