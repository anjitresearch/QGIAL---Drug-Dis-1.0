"""
Performance Tracker

Tracks and visualizes performance metrics throughout the QGIAL optimization process.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import time
import os


class PerformanceTracker:
    """
    Performance tracker for QGIAL optimization metrics.
    
    Tracks fitness, binding affinity, ADMET scores, and other metrics
    throughout the optimization process.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.history = []
        self.start_time = time.time()
        self.metrics = {
            'generation': [],
            'avg_fitness': [],
            'max_fitness': [],
            'avg_binding_affinity': [],
            'max_binding_affinity': [],
            'avg_admet_score': [],
            'max_admet_score': [],
            'pains_free_rate': [],
            'population_size': [],
            'diversity_score': []
        }
        
    def update(self, generation_stats: Dict):
        """Update performance metrics."""
        self.history.append(generation_stats)
        
        # Extract metrics
        generation = generation_stats.get('generation', len(self.history))
        avg_fitness = generation_stats.get('avg_fitness', 0.0)
        max_fitness = generation_stats.get('max_fitness', 0.0)
        avg_binding = generation_stats.get('avg_binding_affinity', 0.0)
        max_binding = generation_stats.get('max_binding_affinity', 0.0)
        avg_admet = generation_stats.get('avg_admet_score', 0.0)
        max_admet = generation_stats.get('max_admet_score', 0.0)
        pains_free = generation_stats.get('pains_free_rate', 0.0)
        pop_size = generation_stats.get('population_size', 0)
        
        # Store metrics
        self.metrics['generation'].append(generation)
        self.metrics['avg_fitness'].append(avg_fitness)
        self.metrics['max_fitness'].append(max_fitness)
        self.metrics['avg_binding_affinity'].append(avg_binding)
        self.metrics['max_binding_affinity'].append(max_binding)
        self.metrics['avg_admet_score'].append(avg_admet)
        self.metrics['max_admet_score'].append(max_admet)
        self.metrics['pains_free_rate'].append(pains_free)
        self.metrics['population_size'].append(pop_size)
        self.metrics['diversity_score'].append(self._calculate_diversity_score(generation_stats))
        
    def _calculate_diversity_score(self, generation_stats: Dict) -> float:
        """Calculate diversity score for generation."""
        # Simple diversity metric based on fitness distribution
        avg_fitness = generation_stats.get('avg_fitness', 0.0)
        max_fitness = generation_stats.get('max_fitness', 0.0)
        
        if max_fitness == 0:
            return 0.0
            
        # Diversity as ratio of average to maximum fitness
        diversity = avg_fitness / max_fitness
        
        return np.clip(diversity, 0.0, 1.0)
        
    def get_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.metrics['generation']:
            return {}
            
        summary = {}
        
        # Calculate statistics for each metric
        for metric_name, values in self.metrics.items():
            if values and isinstance(values[0], (int, float)):
                summary[f'{metric_name}_final'] = values[-1]
                summary[f'{metric_name}_best'] = max(values)
                summary[f'{metric_name}_worst'] = min(values)
                summary[f'{metric_name}_average'] = np.mean(values)
                summary[f'{metric_name}_improvement'] = values[-1] - values[0] if len(values) > 1 else 0.0
                
        # Calculate optimization efficiency
        if len(self.metrics['avg_fitness']) > 1:
            initial_fitness = self.metrics['avg_fitness'][0]
            final_fitness = self.metrics['avg_fitness'][-1]
            improvement = (final_fitness - initial_fitness) / initial_fitness if initial_fitness > 0 else 0.0
            summary['fitness_improvement_percentage'] = improvement * 100
            
        # Calculate convergence metrics
        convergence_gen = self._find_convergence_generation()
        summary['convergence_generation'] = convergence_gen
        summary['generations_to_convergence'] = convergence_gen + 1 if convergence_gen >= 0 else len(self.metrics['generation'])
        
        # Time metrics
        summary['total_time_seconds'] = time.time() - self.start_time
        summary['avg_time_per_generation'] = summary['total_time_seconds'] / len(self.metrics['generation'])
        
        return summary
        
    def _find_convergence_generation(self) -> int:
        """Find generation where optimization converged."""
        if len(self.metrics['max_fitness']) < 10:
            return len(self.metrics['max_fitness']) - 1
            
        # Check for plateau in max fitness
        recent_fitness = self.metrics['max_fitness'][-10:]
        fitness_std = np.std(recent_fitness)
        
        if fitness_std < 0.01:  # Convergence threshold
            return len(self.metrics['max_fitness']) - 10
            
        return len(self.metrics['max_fitness']) - 1
        
    def plot_progress(self, save_path: Optional[str] = None):
        """Plot optimization progress."""
        if not self.metrics['generation']:
            print("No data to plot")
            return
            
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        generations = self.metrics['generation']
        
        # Plot 1: Fitness Progress
        ax1.plot(generations, self.metrics['avg_fitness'], 'b-', linewidth=2, label='Average Fitness', marker='o')
        ax1.plot(generations, self.metrics['max_fitness'], 'r-', linewidth=2, label='Maximum Fitness', marker='s')
        ax1.set_title('Fitness Progress')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot 2: Binding Affinity Progress
        ax2.plot(generations, self.metrics['avg_binding_affinity'], 'g-', linewidth=2, label='Average Binding', marker='o')
        ax2.plot(generations, self.metrics['max_binding_affinity'], 'orange', linewidth=2, label='Maximum Binding', marker='s')
        ax2.set_title('Binding Affinity Progress')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Binding Affinity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: ADMET Score Progress
        ax3.plot(generations, self.metrics['avg_admet_score'], 'purple', linewidth=2, label='Average ADMET', marker='o')
        ax3.plot(generations, self.metrics['max_admet_score'], 'brown', linewidth=2, label='Maximum ADMET', marker='s')
        ax3.set_title('ADMET Score Progress')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('ADMET Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # Plot 4: Population Metrics
        ax4_twin = ax4.twinx()
        
        # Plot PAINS-free rate
        line1 = ax4.plot(generations, self.metrics['pains_free_rate'], 'cyan', linewidth=2, label='PAINS-Free Rate', marker='o')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('PAINS-Free Rate', color='cyan')
        ax4.tick_params(axis='y', labelcolor='cyan')
        ax4.set_ylim([0, 1])
        
        # Plot diversity score on secondary axis
        line2 = ax4_twin.plot(generations, self.metrics['diversity_score'], 'magenta', linewidth=2, label='Diversity Score', marker='s')
        ax4_twin.set_ylabel('Diversity Score', color='magenta')
        ax4_twin.tick_params(axis='y', labelcolor='magenta')
        ax4_twin.set_ylim([0, 1])
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        ax4.set_title('Population Quality Metrics')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            output_dir = os.path.join(os.getcwd(), 'data', 'results')
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'qgial_progress_{int(time.time())}.png')
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Progress plot saved to: {save_path}")
        
        plt.show()
        
    def plot_correlation_matrix(self, save_path: Optional[str] = None):
        """Plot correlation matrix of metrics."""
        if not self.metrics['generation']:
            print("No data to plot")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        # Calculate correlation matrix
        correlation_matrix = df_numeric.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        
        plt.title('QGIAL Performance Metrics Correlation Matrix')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            output_dir = os.path.join(os.getcwd(), 'data', 'results')
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'qgial_correlation_{int(time.time())}.png')
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Correlation matrix saved to: {save_path}")
        
        plt.show()
        
    def export_metrics(self, save_path: Optional[str] = None):
        """Export metrics to CSV file."""
        if not self.metrics['generation']:
            print("No data to export")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save to CSV
        if save_path is None:
            output_dir = os.path.join(os.getcwd(), 'data', 'results')
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'qgial_metrics_{int(time.time())}.csv')
            
        df.to_csv(save_path, index=False)
        print(f"📊 Metrics exported to: {save_path}")
        
    def get_history(self) -> List[Dict]:
        """Get complete optimization history."""
        return self.history.copy()
        
    def load_history(self, history: List[Dict]):
        """Load optimization history."""
        self.history = history.copy()
        
        # Rebuild metrics from history
        self.metrics = {
            'generation': [],
            'avg_fitness': [],
            'max_fitness': [],
            'avg_binding_affinity': [],
            'max_binding_affinity': [],
            'avg_admet_score': [],
            'max_admet_score': [],
            'pains_free_rate': [],
            'population_size': [],
            'diversity_score': []
        }
        
        for generation_stats in history:
            self.update(generation_stats)
            
    def reset(self):
        """Reset performance tracker."""
        self.history = []
        self.start_time = time.time()
        self.metrics = {
            'generation': [],
            'avg_fitness': [],
            'max_fitness': [],
            'avg_binding_affinity': [],
            'max_binding_affinity': [],
            'avg_admet_score': [],
            'max_admet_score': [],
            'pains_free_rate': [],
            'population_size': [],
            'diversity_score': []
        }
