"""
Base Target Class

Provides the base interface and common functionality for all target-specific
molecular design modules in the QGIAL framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from rdkit import Chem


class BaseTarget(ABC):
    """
    Abstract base class for target-specific molecular design modules.
    
    Provides common interface and functionality that all target modules
    should implement for consistency within the QGIAL framework.
    """
    
    def __init__(self):
        """Initialize base target module."""
        self.target_name = ""
        self.target_class = ""
        self.disease_indications = []
        
    @abstractmethod
    def get_target_info(self) -> Dict:
        """
        Get comprehensive target information.
        
        Returns:
            Dictionary containing target-specific information
        """
        pass
        
    @abstractmethod
    def design_molecules(self, n_molecules: int = 10) -> List[Chem.Mol]:
        """
        Design molecules specifically for this target.
        
        Args:
            n_molecules: Number of molecules to design
            
        Returns:
            List of designed RDKit molecules
        """
        pass
        
    @abstractmethod
    def evaluate_binding_affinity(self, molecule: Chem.Mol) -> float:
        """
        Evaluate predicted binding affinity for this target.
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Predicted binding affinity score (0-1, higher is better)
        """
        pass
        
    def get_binding_pocket_features(self) -> Dict:
        """
        Get detailed binding pocket features.
        
        Returns:
            Dictionary containing binding pocket information
        """
        return {}
        
    def generate_virtual_library(self, n_compounds: int = 1000) -> List[Chem.Mol]:
        """
        Generate virtual library of target-specific compounds.
        
        Args:
            n_compounds: Number of compounds to generate
            
        Returns:
            List of virtual compounds
        """
        return []
        
    def filter_library(self, library: List[Chem.Mol], 
                      threshold: float = 0.5) -> List[Chem.Mol]:
        """
        Filter virtual library based on target-specific criteria.
        
        Args:
            library: List of molecules to filter
            threshold: Minimum affinity threshold
            
        Returns:
            Filtered list of molecules
        """
        return []
