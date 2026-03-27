"""
Policy and Value Networks for Multi-Objective DRL

Implements neural network architectures for policy (actor) and value (critic) networks
used in the multi-objective deep reinforcement learning agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Policy network (actor) for multi-objective DRL agent.
    
    Takes state representation and outputs action probabilities
    for molecular modifications.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.
        
        Args:
            state: State tensor
            
        Returns:
            Action logits
        """
        logits = self.network(state)
        
        # Apply tanh activation to bound actions [-1, 1]
        actions = torch.tanh(logits)
        
        return actions
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)


class ValueNetwork(nn.Module):
    """
    Value network (critic) for multi-objective DRL agent.
    
    Takes state and action and outputs Q-value for state-action pair.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State processing layers
        state_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:2]:  # Use first 2 hidden dims for state
            state_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        self.state_encoder = nn.Sequential(*state_layers)
        
        # Action processing layers
        action_layers = []
        input_dim = action_dim
        
        for hidden_dim in hidden_dims[:2]:  # Use first 2 hidden dims for action
            action_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        self.action_encoder = nn.Sequential(*action_layers)
        
        # Combined processing layers
        combined_input_dim = hidden_dims[1] * 2  # State and action encoders output
        combined_layers = []
        
        for hidden_dim in hidden_dims[2:]:  # Use remaining hidden dims
            combined_layers.extend([
                nn.Linear(combined_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            combined_input_dim = hidden_dim
            
        # Output layer
        combined_layers.append(nn.Linear(combined_input_dim, 1))
        
        self.combined_network = nn.Sequential(*combined_layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value
        """
        # Encode state and action
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        
        # Combine features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Get Q-value
        q_value = self.combined_network(combined_features)
        
        return q_value
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)


class MultiHeadValueNetwork(nn.Module):
    """
    Multi-head value network for multi-objective optimization.
    
    Outputs separate Q-values for each objective.
    """
    
    def __init__(self, state_dim: int, action_dim: int, n_objectives: int,
                 hidden_dims: list = [256, 128, 64]):
        """
        Initialize multi-head value network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            n_objectives: Number of objectives
            hidden_dims: List of hidden layer dimensions
        """
        super(MultiHeadValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        
        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Objective-specific heads
        self.objective_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1) for _ in range(n_objectives)
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head value network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-values for each objective
        """
        # Combine state and action
        combined_input = torch.cat([state, action], dim=-1)
        
        # Extract shared features
        shared_features = self.shared_network(combined_input)
        
        # Get objective-specific Q-values
        q_values = []
        for head in self.objective_heads:
            q_value = head(shared_features)
            q_values.append(q_value)
            
        # Stack Q-values
        q_values = torch.stack(q_values, dim=-1)  # [batch, 1, n_objectives]
        q_values = q_values.squeeze(dim=-2)  # [batch, n_objectives]
        
        return q_values
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)


class AttentionPolicyNetwork(nn.Module):
    """
    Policy network with attention mechanism for better molecular feature learning.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128],
                 n_heads: int = 8):
        """
        Initialize attention policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            n_heads: Number of attention heads
        """
        super(AttentionPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_heads = n_heads
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, hidden_dims[0])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0])
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dims[0])
        self.norm2 = nn.LayerNorm(hidden_dims[0])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[0], action_dim)
        
        # Positional encoding (optional)
        self.positional_encoding = self._create_positional_encoding(100, hidden_dims[0])
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention policy network.
        
        Args:
            state: State tensor
            
        Returns:
            Action logits
        """
        # Project input
        x = self.input_projection(state)
        
        # Add positional encoding
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
            x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # Global pooling
        x = x.mean(dim=1)  # [batch, hidden_dim]
        
        # Output layer
        logits = self.output_layer(x)
        actions = torch.tanh(logits)
        
        return actions
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Initialize residual block.
        
        Args:
            dim: Dimension of the block
            dropout: Dropout rate
        """
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.block(x)
        out = self.relu(out + residual)
        return out


class DeepPolicyNetwork(nn.Module):
    """
    Deep policy network with residual connections.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256, 256],
                 n_residual_blocks: int = 3):
        """
        Initialize deep policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            n_residual_blocks: Number of residual blocks
        """
        super(DeepPolicyNetwork, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0]) for _ in range(n_residual_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[0], action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through deep policy network."""
        x = self.input_layer(state)
        x = torch.relu(x)
        
        for block in self.residual_blocks:
            x = block(x)
            
        logits = self.output_layer(x)
        actions = torch.tanh(logits)
        
        return actions
