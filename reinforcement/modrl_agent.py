"""
Multi-Objective Deep Reinforcement Learning Agent

Implements MODRL agent for simultaneous optimization of multiple drug discovery
objectives including binding affinity, ADMET properties, synthetic accessibility,
and target selectivity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
from .environment import DrugDiscoveryEnvironment
from .reward_functions import MultiObjectiveRewardFunction
from .policy_networks import PolicyNetwork, ValueNetwork


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class MultiObjectiveDRLAgent:
    """
    Multi-Objective Deep Reinforcement Learning Agent for drug discovery.
    
    Uses actor-critic architecture with multi-objective optimization to
    simultaneously optimize multiple drug discovery objectives.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 objectives: List[str], learning_rate: float = 0.0003,
                 gamma: float = 0.99, tau: float = 0.005, 
                 buffer_size: int = 100000, batch_size: int = 64):
        """
        Initialize MODRL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            objectives: List of optimization objectives
            learning_rate: Learning rate for neural networks
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Replay buffer size
            batch_size: Training batch size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize actor and critic networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, action_dim).to(self.device)
        
        # Target networks for stable training
        self.target_policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_value_net = ValueNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize target networks
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Multi-objective reward function
        self.reward_function = MultiObjectiveRewardFunction(objectives)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'rewards': [],
            'objective_scores': {obj: [] for obj in objectives}
        }
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action using policy network with epsilon-greedy exploration.
        
        Args:
            state: Current state
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if explore and random.random() < self.epsilon:
            # Random exploration
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # Greedy action from policy
            with torch.no_grad():
                action = self.policy_net(state).cpu().numpy()[0]
                
        return action
        
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                       reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        
    def train(self) -> float:
        """
        Train the agent using experiences from replay buffer.
        
        Returns:
            Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
            
        # Sample batch of experiences
        experiences = random.sample(self.replay_buffer, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # Train value network
        value_loss = self._train_value_network(states, actions, rewards, next_states, dones)
        
        # Train policy network
        policy_loss = self._train_policy_network(states, actions)
        
        # Update target networks
        self._update_target_networks()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Record training statistics
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['rewards'].append(rewards.mean().item())
        
        return policy_loss.item() + value_loss.item()
        
    def _train_value_network(self, states: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_states: torch.Tensor,
                           dones: torch.Tensor) -> torch.Tensor:
        """Train value network using TD learning."""
        with torch.no_grad():
            # Target Q-values
            next_actions = self.target_policy_net(next_states)
            target_q_values = self.target_value_net(next_states, next_actions)
            target_values = rewards + (1 - dones) * self.gamma * target_q_values
            
        # Current Q-values
        current_q_values = self.value_net(states, actions)
        
        # Value loss
        value_loss = nn.MSELoss()(current_q_values, target_values)
        
        # Optimize value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss
        
    def _train_policy_network(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Train policy network using policy gradient."""
        # Current actions from policy
        current_actions = self.policy_net(states)
        
        # Q-values for current actions
        q_values = self.value_net(states, current_actions)
        
        # Policy loss (maximize Q-values)
        policy_loss = -q_values.mean()
        
        # Optimize policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss
        
    def _update_target_networks(self):
        """Soft update of target networks."""
        # Update target policy network
        for target_param, param in zip(self.target_policy_net.parameters(),
                                     self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
            
        # Update target value network
        for target_param, param in zip(self.target_value_net.parameters(),
                                     self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
            
    def evaluate_objectives(self, molecule, target_info: Dict) -> Dict[str, float]:
        """
        Evaluate multiple objectives for a given molecule.
        
        Args:
            molecule: RDKit molecule object
            target_info: Target-specific information
            
        Returns:
            Dictionary of objective scores
        """
        objective_scores = {}
        
        # Binding affinity (higher is better)
        if 'binding_affinity' in self.objectives:
            # Simplified binding affinity calculation
            objective_scores['binding_affinity'] = self._calculate_binding_affinity(
                molecule, target_info
            )
            
        # ADMET score (higher is better)
        if 'admet_score' in self.objectives:
            objective_scores['admet_score'] = self._calculate_admet_score(molecule)
            
        # Synthetic accessibility (higher is better)
        if 'synthetic_accessibility' in self.objectives:
            objective_scores['synthetic_accessibility'] = self._calculate_sa_score(molecule)
            
        # Selectivity (higher is better)
        if 'selectivity' in self.objectives:
            objective_scores['selectivity'] = self._calculate_selectivity(molecule, target_info)
            
        return objective_scores
        
    def _calculate_binding_affinity(self, molecule, target_info: Dict) -> float:
        """Calculate binding affinity score."""
        # Simplified binding affinity calculation
        # In practice, would use molecular docking or quantum calculations
        
        # Use molecular weight and logP as proxy
        from rdkit.Chem import Descriptors
        mw = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        
        # Simple scoring function (higher is better)
        affinity = 1.0 / (1.0 + abs(mw - 300) / 100 + abs(logp - 2) / 2)
        
        return np.clip(affinity, 0, 1)
        
    def _calculate_admet_score(self, molecule) -> float:
        """Calculate ADMET score."""
        from rdkit.Chem import Descriptors
        
        # Basic ADMET properties
        mw = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        tpsa = Descriptors.TPSA(molecule)
        hbd = Descriptors.NumHDonors(molecule)
        hba = Descriptors.NumHAcceptors(molecule)
        
        # Lipinski's rule of five compliance
        lipinski_score = 0
        if mw <= 500:
            lipinski_score += 1
        if logp <= 5:
            lipinski_score += 1
        if hbd <= 5:
            lipinski_score += 1
        if hba <= 10:
            lipinski_score += 1
            
        # Normalize to 0-1 range
        admet_score = lipinski_score / 4.0
        
        return admet_score
        
    def _calculate_sa_score(self, molecule) -> float:
        """Calculate synthetic accessibility score."""
        # Simplified SA score calculation
        # In practice, would use more sophisticated methods
        
        from rdkit.Chem import Descriptors
        
        # Use molecular complexity as proxy
        mw = Descriptors.MolWt(molecule)
        n_rotatable = Descriptors.NumRotatableBonds(molecule)
        n_rings = Descriptors.NumRings(molecule)
        
        # Simple SA score (higher is better, more accessible)
        complexity = mw / 100 + n_rotatable / 5 + n_rings / 3
        sa_score = 1.0 / (1.0 + complexity / 10)
        
        return np.clip(sa_score, 0, 1)
        
    def _calculate_selectivity(self, molecule, target_info: Dict) -> float:
        """Calculate selectivity score."""
        # Simplified selectivity calculation
        # In practice, would use off-target predictions
        
        # Use molecular properties as proxy for selectivity
        from rdkit.Chem import Descriptors
        
        logp = Descriptors.MolLogP(molecule)
        tpsa = Descriptors.TPSA(molecule)
        
        # Simple selectivity scoring
        selectivity = 1.0 / (1.0 + abs(logp - 2) + abs(tpsa - 80) / 50)
        
        return np.clip(selectivity, 0, 1)
        
    def optimize_molecule(self, initial_molecule, target_info: Dict, 
                         max_iterations: int = 100) -> Tuple[Any, Dict]:
        """
        Optimize a molecule using multi-objective reinforcement learning.
        
        Args:
            initial_molecule: Starting molecule
            target_info: Target-specific information
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized molecule and optimization history
        """
        # Initialize environment
        env = DrugDiscoveryEnvironment(initial_molecule, target_info, self.objectives)
        
        # Optimization history
        optimization_history = {
            'rewards': [],
            'objective_scores': {obj: [] for obj in self.objectives},
            'molecules': []
        }
        
        # Current state
        state = env.reset()
        current_molecule = initial_molecule
        
        for iteration in range(max_iterations):
            # Select action
            action = self.select_action(state, explore=True)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            if iteration > 10:
                loss = self.train()
                
            # Update current molecule if improvement
            if info.get('improved', False):
                current_molecule = info.get('molecule', current_molecule)
                
            # Record optimization progress
            optimization_history['rewards'].append(reward)
            optimization_history['molecules'].append(current_molecule)
            
            for obj, score in info.get('objective_scores', {}).items():
                if obj in optimization_history['objective_scores']:
                    optimization_history['objective_scores'][obj].append(score)
                    
            # Update state
            state = next_state
            
            # Check convergence
            if done:
                break
                
        return current_molecule, optimization_history
        
    def get_training_statistics(self) -> Dict:
        """Get training statistics."""
        return self.training_stats.copy()
        
    def save_model(self, filepath: str):
        """Save trained model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'target_policy_net_state_dict': self.target_policy_net.state_dict(),
            'target_value_net_state_dict': self.target_value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'epsilon': self.epsilon
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.target_policy_net.load_state_dict(checkpoint['target_policy_net_state_dict'])
        self.target_value_net.load_state_dict(checkpoint['target_value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.epsilon = checkpoint['epsilon']
