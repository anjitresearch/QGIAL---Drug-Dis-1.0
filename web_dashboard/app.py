"""
QGIAL Web Dashboard

Interactive web interface for monitoring QGIAL drug discovery pipeline.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import QGIAL components with error handling
try:
    from pipeline.qgial_pipeline import QGIALPipeline
    from quantum.vqc_molecular import VariationalQuantumCircuit
    from generative.hqgan import HybridQuantumGAN
    from reinforcement.modrl_agent import MultiObjectiveDRLAgent
    from targets.kras_g12d import KRASG12DTarget
    from properties.admet_predictor import ADMETPredictor
    from properties.pains_detector import PAINSDetector
    QGIAL_AVAILABLE = True
except ImportError as e:
    print(f"QGIAL components not available: {e}")
    QGIAL_AVAILABLE = False

app = Flask(__name__)

# Global variables for pipeline state
pipeline = None
current_results = {}
optimization_history = []

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current pipeline status."""
    global pipeline, current_results
    
    if pipeline is None:
        return jsonify({
            'status': 'idle',
            'message': 'Pipeline not initialized',
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({
        'status': 'running' if pipeline.current_generation > 0 else 'initialized',
        'generation': pipeline.current_generation,
        'best_molecules': len(pipeline.best_molecules),
        'target': pipeline.target_name,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_pipeline():
    """Initialize QGIAL pipeline."""
    global pipeline, current_results, optimization_history
    
    if not QGIAL_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'QGIAL components not available'
        })
    
    try:
        data = request.get_json()
        target = data.get('target', 'KRAS_G12D')
        
        # Initialize pipeline
        pipeline = QGIALPipeline(target_name=target)
        
        # Reset results
        current_results = {}
        optimization_history = []
        
        return jsonify({
            'success': True,
            'message': f'Pipeline initialized for {target}',
            'target': target,
            'config': pipeline.config
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """Run QGIAL pipeline."""
    global pipeline, current_results, optimization_history
    
    if not QGIAL_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'QGIAL components not available'
        })
    
    try:
        if pipeline is None:
            return jsonify({
                'success': False,
                'error': 'Pipeline not initialized'
            })
        
        data = request.get_json()
        max_generations = data.get('max_generations', 5)
        
        # Run pipeline
        results = pipeline.run_pipeline(max_generations=max_generations)
        current_results = results
        
        # Extract optimization history
        if 'performance_summary' in results:
            optimization_history = results['performance_summary']
        
        return jsonify({
            'success': True,
            'message': 'Pipeline completed successfully',
            'results': {
                'total_molecules': results.get('total_molecules_generated', 0),
                'top_molecules': len(results.get('top_molecules', [])),
                'efficiency': results.get('optimization_efficiency', {}),
                'target_metrics': results.get('target_specific_metrics', {})
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/molecules')
def get_molecules():
    """Get best molecules."""
    global current_results
    
    if not current_results:
        return jsonify({'molecules': []})
    
    molecules = current_results.get('best_molecules', [])
    
    # Limit to top 20 for display
    display_molecules = molecules[:20]
    
    return jsonify({
        'molecules': display_molecules,
        'total': len(molecules)
    })

@app.route('/api/performance')
def get_performance():
    """Get performance metrics."""
    global optimization_history
    
    if not optimization_history:
        return jsonify({'metrics': {}})
    
    return jsonify({
        'metrics': optimization_history,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/visualization/fitness')
def get_fitness_plot():
    """Get fitness progression plot."""
    global optimization_history
    
    if not optimization_history:
        return jsonify({'plot': None})
    
    # Extract fitness data
    generations = list(range(len(optimization_history)))
    avg_fitness = [gen.get('avg_fitness', 0) for gen in optimization_history]
    max_fitness = [gen.get('max_fitness', 0) for gen in optimization_history]
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=generations,
        y=avg_fitness,
        mode='lines+markers',
        name='Average Fitness',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=generations,
        y=max_fitness,
        mode='lines+markers',
        name='Maximum Fitness',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='QGIAL Fitness Progression',
        xaxis_title='Generation',
        yaxis_title='Fitness Score',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return jsonify({
        'plot': json.loads(fig.to_json())
    })

@app.route('/api/visualization/metrics')
def get_metrics_plot():
    """Get comprehensive metrics plot."""
    global optimization_history
    
    if not optimization_history:
        return jsonify({'plot': None})
    
    # Extract metrics data
    generations = list(range(len(optimization_history)))
    binding_affinity = [gen.get('avg_binding_affinity', 0) for gen in optimization_history]
    admet_score = [gen.get('avg_admet_score', 0) for gen in optimization_history]
    pains_free = [gen.get('pains_free_rate', 0) for gen in optimization_history]
    
    # Create subplot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=binding_affinity,
        mode='lines+markers',
        name='Binding Affinity',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=admet_score,
        mode='lines+markers',
        name='ADMET Score',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=pains_free,
        mode='lines+markers',
        name='PAINS-Free Rate',
        line=dict(color='purple', width=2)
    ))
    
    fig.update_layout(
        title='QGIAL Optimization Metrics',
        xaxis_title='Generation',
        yaxis_title='Score',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return jsonify({
        'plot': json.loads(fig.to_json())
    })

@app.route('/api/targets')
def get_targets():
    """Get available targets."""
    targets = {
        'KRAS_G12D': {
            'name': 'KRAS G12D',
            'description': 'Oncogenic KRAS G12D mutant protein',
            'disease': 'Various cancers',
            'binding_pocket': 450.0
        },
        'PD_L1': {
            'name': 'PD-L1',
            'description': 'Programmed Death-Ligand 1 immune checkpoint',
            'disease': 'Cancer immunotherapy',
            'binding_pocket': 680.0
        },
        'SARS_COV_2_MPRO': {
            'name': 'SARS-CoV-2 Mpro',
            'description': 'Main protease for COVID-19',
            'disease': 'COVID-19',
            'binding_pocket': 380.0
        }
    }
    
    return jsonify(targets)

@app.route('/api/demo')
def run_demo():
    """Run quick demo with sample data."""
    global pipeline, current_results, optimization_history
    
    try:
        # Initialize with KRAS G12D
        pipeline = QGIALPipeline(target_name='KRAS_G12D')
        
        # Create sample optimization history
        optimization_history = []
        for i in range(10):
            optimization_history.append({
                'generation': i,
                'avg_fitness': 0.3 + (i * 0.05) + np.random.normal(0, 0.02),
                'max_fitness': 0.5 + (i * 0.04) + np.random.normal(0, 0.01),
                'avg_binding_affinity': 0.2 + (i * 0.06) + np.random.normal(0, 0.02),
                'max_binding_affinity': 0.4 + (i * 0.05) + np.random.normal(0, 0.01),
                'avg_admet_score': 0.6 + (i * 0.02) + np.random.normal(0, 0.01),
                'max_admet_score': 0.7 + (i * 0.02) + np.random.normal(0, 0.01),
                'pains_free_rate': 0.8 + (i * 0.02) + np.random.normal(0, 0.01)
            })
        
        # Create sample results
        current_results = {
            'total_molecules_generated': 100,
            'top_molecules': [
                {
                    'rank': 1,
                    'smiles': 'CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2',
                    'fitness': 0.85,
                    'binding_affinity': 0.82,
                    'admet_score': 0.78,
                    'pains_score': 0.05
                },
                {
                    'rank': 2,
                    'smiles': 'C1=CC=C(C=C1)C2=NC=NC(=N2)N',
                    'fitness': 0.82,
                    'binding_affinity': 0.79,
                    'admet_score': 0.81,
                    'pains_score': 0.03
                },
                {
                    'rank': 3,
                    'smiles': 'CC1=CC(=NC=C1)N2C=NC3=CC=CC=N32',
                    'fitness': 0.79,
                    'binding_affinity': 0.76,
                    'admet_score': 0.83,
                    'pains_score': 0.02
                }
            ],
            'optimization_efficiency': {
                'fitness_improvement': 0.55,
                'convergence_generation': 7,
                'generations_to_convergence': 8
            },
            'target_specific_metrics': {
                'avg_top_binding': 0.79,
                'max_top_binding': 0.82,
                'sub_nanomolar_count': 2,
                'favorable_admet_count': 3
            }
        }
        
        return jsonify({
            'success': True,
            'message': 'Demo data loaded successfully',
            'results': current_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
