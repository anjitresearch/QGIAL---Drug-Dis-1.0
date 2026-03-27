"""
Closed-Loop Optimization Pipeline

This module implements the complete closed-loop optimization pipeline
that integrates all QGIAL components for autonomous molecular design.
"""

from .qgial_pipeline import QGIALPipeline
from .optimization_loop import OptimizationLoop
from .evolution_manager import EvolutionManager
from .performance_tracker import PerformanceTracker

__all__ = [
    'QGIALPipeline',
    'OptimizationLoop',
    'EvolutionManager',
    'PerformanceTracker'
]
