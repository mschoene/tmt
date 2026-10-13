"""MLP Encoder wrapper for policies."""

from flax import linen as nn
from typing import Sequence
import jax.numpy as jnp


class MLPEncoderPolicy(nn.Module):
    """Wraps a policy with an MLP encoder and decoder."""
    hidden_dims: Sequence[int]
    latent_dim: int
    original_obs_dim: int
    inner_policy: nn.Module
    stack_size: int = 4
    init_decoder: bool = True
    action_dim: int = 0  # NEW: needed for decode_to next state

    
    @nn.compact
    def __call__(self, observations, *args, **kwargs):
        """Forward pass through encoder -> inner policy."""
        if self.init_decoder and self.is_initializing():
            dummy_latent = jnp.zeros((1, self.latent_dim))
            dummy_action = jnp.zeros((1, self.action_dim))
            
            # Initialize all decoder variants
            _ = self.decode(dummy_latent)
            _ = self.decode_full_stack(dummy_latent)
            if self.action_dim > 0:
                _ = self.decode_with_action(dummy_latent, dummy_action)
        
        x = self.encode(observations)
        return self.inner_policy(x, *args, **kwargs)
    
    @nn.compact
    def encode(self, observations):
        """Encode stacked observations to latent space."""
        x = observations
        
        # Encoder MLP
        for i, h in enumerate(self.hidden_dims):
            x = nn.Dense(h, name=f'encoder_dense_{i}')(x)
            x = nn.LayerNorm(name=f'encoder_ln_{i}')(x)
            x = nn.relu(x)
        
        # Project to latent dimension
        x = nn.Dense(self.latent_dim, name='enc_proj')(x)
        # x = nn.LayerNorm(name='enc_proj_ln')(x)

        return x
    
    @nn.compact
    def decode(self, latent):
        """Decode latent back to observation space."""
        x = latent
        
        # Decoder MLP (reverse of encoder)
        for i, h in enumerate(reversed(self.hidden_dims)):
            x = nn.Dense(h, name=f'decoder_dense_{i}')(x)
            x = nn.LayerNorm(name=f'decoder_ln_{i}')(x)
            x = nn.relu(x)
        
        # Project to original observation dimension
        x = nn.Dense(self.original_obs_dim, name='dec_proj')(x)
        
        return x
    
    @nn.compact
    def decode_full_stack(self, latent):
        """
        Decode latent to full stacked observation.
        Used when recon_target='full_stack'.
        """
        x = latent
        
        # Decoder MLP
        for i, h in enumerate(reversed(self.hidden_dims)):
            x = nn.Dense(h, name=f'decoder_full_dense_{i}')(x)
            x = nn.LayerNorm(name=f'decoder_full_ln_{i}')(x)
            x = nn.relu(x)
        
        # Output full stacked observation
        full_obs_dim = self.original_obs_dim * self.stack_size
        x = nn.Dense(full_obs_dim, name='dec_full_proj')(x)
        
        return x
    
    @nn.compact
    def decode_with_action(self, latents: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Decoder with action input for next-frame prediction.
        Args:
            latents: Latent representation (batch, latent_dim)
            actions: Actions taken (batch, action_dim)
        Returns:
            next_obs: Predicted next observation - single frame (batch, original_obs_dim)
        """
        # Concatenate latents and actions as input
        decoder_input = jnp.concatenate([latents, actions], axis=-1)
        
        x = decoder_input
        
        # MLP decoder (with action conditioning)
        for i, h in enumerate(reversed(self.hidden_dims)):
            x = nn.Dense(h, name=f'decoder_action_dense_{i}')(x)
            x = nn.LayerNorm(name=f'decoder_action_ln_{i}')(x)

            x = nn.relu(x)
        
        # Output: next observation (single frame)
        x = nn.Dense(self.original_obs_dim, name='dec_action_proj')(x)
        
        return x
    
    def apply_policy(self, latent, *args, **kwargs):
        """Apply inner policy to latent representation."""
        return self.inner_policy(latent, *args, **kwargs)
    
    def deterministic_action(self, observations, temperature: float = 1.0):
        dist = self(observations, temperature)
        return self.mean_action_from_dist(dist)
    
    def deterministic_action_from_latent(self, latent):
        return self.inner_policy.deterministic_action(latent)
    
    # def deterministic_action(self, latent, *args, **kwargs):
    #     """
    #     Deterministic action from wrapped policy.
    #     Handles encoding before calling inner policy.
    #     """
    #     return self.inner_policy.deterministic_action(latent, *args, **kwargs)


def wrap_with_encoder(
    policy: nn.Module,
    input_dim: int,
    latent_dim: int,
    hidden_dims: Sequence[int] = (128, 128),
    stack_size: int = 4,
    action_dim: int = 0
):
    """
    Wrap a policy with an MLP encoder.
    
    Args:
        policy: The inner policy module
        input_dim: Input observation dimension
        latent_dim: Latent representation dimension
        hidden_dims: Hidden dimensions for encoder/decoder MLPs
        stack_size: Number of stacked frames (for computing original obs dim)
    
    Returns:
        MLPEncoderPolicy wrapping the input policy
    """
    original_obs_dim = input_dim // stack_size
    
    print(f"Wrapping Policy: Input dim ({input_dim}) != Target Rep dim ({latent_dim}).")
    print(f"  -> Injecting MLP Encoder: {input_dim} -> {hidden_dims} -> {latent_dim}")
    print(f"  -> Injecting MLP Decoder: {latent_dim} -> {hidden_dims} -> {original_obs_dim}")
    if action_dim > 0:
        print(f"  -> Injecting MLP Decoder (next_frame): {latent_dim}+{action_dim} -> {hidden_dims} -> {original_obs_dim}")
    
    return MLPEncoderPolicy(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        original_obs_dim=original_obs_dim,
        inner_policy=policy,
        stack_size=stack_size,
        action_dim=action_dim,
    )