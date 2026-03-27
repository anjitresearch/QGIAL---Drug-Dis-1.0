"""
Classical Discriminator for HQGAN

Implements the classical discriminator component of the Hybrid Quantum-Classical
Generative Adversarial Network using deep neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ClassicalDiscriminator(nn.Module):
    """
    Classical discriminator using deep neural networks.
    
    Distinguishes between real molecular features and generated features
    from the quantum generator.
    """
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [128, 64, 32]):
        """
        Initialize classical discriminator.
        
        Args:
            input_dim: Dimension of input molecular features
            hidden_dims: Hidden layer dimensions
        """
        super(ClassicalDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        
        # Build discriminator network
        self.network = self._build_network(input_dim, hidden_dims)
        
        # Spectral normalization for stability
        self.apply_spectral_norm = True
        
        if self.apply_spectral_norm:
            self._apply_spectral_normalization()
            
        # Initialize weights
        self._initialize_weights()
        
    def _build_network(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build discriminator network layers."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
        
    def _apply_spectral_normalization(self):
        """Apply spectral normalization to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module = nn.utils.spectral_norm(module)
                
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input molecular features
            
        Returns:
            Discriminator output (probability of being real)
        """
        return self.network(x)
        
    def compute_discriminator_loss(self, real_output: torch.Tensor, 
                                  fake_output: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_output: Discriminator output for real samples
            fake_output: Discriminator output for fake samples
            
        Returns:
            Discriminator loss
        """
        # Real samples should have output 1
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        
        # Fake samples should have output 0
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        
        # Total loss
        total_loss = (real_loss + fake_loss) / 2
        
        return total_loss
        
    def compute_generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss (wants discriminator to think fake is real).
        
        Args:
            fake_output: Discriminator output for fake samples
            
        Returns:
            Generator loss
        """
        # Generator wants discriminator to output 1 for fake samples
        return F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
        
    def get_feature_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate feature representation.
        
        Args:
            x: Input molecular features
            
        Returns:
            Intermediate features
        """
        # Pass through all layers except the final sigmoid
        features = x
        for i, layer in enumerate(self.network):
            if i < len(self.network) - 1:  # Exclude final sigmoid
                features = layer(features)
                
        return features
        
    def gradient_penalty(self, real_samples: torch.Tensor, 
                        fake_samples: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real molecular features
            fake_samples: Generated molecular features
            device: Device to compute on
            
        Returns:
            Gradient penalty
        """
        batch_size = real_samples.size(0)
        
        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1).to(device)
        
        # Interpolate between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Compute discriminator output for interpolated samples
        d_interpolated = self(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return penalty
        
    def get_discriminator_statistics(self) -> dict:
        """Get discriminator statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dimension': self.input_dim,
            'spectral_norm': self.apply_spectral_norm
        }
