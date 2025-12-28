"""
Metacontroller Architecture for Internal RL

This implements the metacontroller from "Emergent temporal abstractions in
autoregressive models enable hierarchical reinforcement learning" (Kobayashi et al., 2025).

The metacontroller:
1. Reads residual stream activations from a frozen base model
2. Generates controller codes z_t (latent abstract actions)
3. Decides when to switch abstract actions (switching gate β_t)
4. Produces controller matrices U_t that modify the residual stream

Architecture Overview:
=====================

   e_{1:T} ──► [Sequence Embedder] ──► s(e_{1:T})
                                           │
                                           ▼
   e_t, h_{t-1} ──► [Controller Encoder] ──► μ_t, Σ_t
                           │
                           ▼
                    z̃_t ~ N(μ_t, Σ_t)
                           │
                           ▼
   e_t, h_{t-1}, z_{t-1} ──► [Switching Unit] ──► β_t
                           │
                           ▼
            z_t = β_t * z̃_t + (1-β_t) * z_{t-1}
                           │
                           ▼
                [Controller Decoder] ──► U_t
                           │
                           ▼
            e_{t,l} ← e_{t,l} + U_t @ e_{t,l}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit for maintaining history state.

    The GRU compresses information from past residual activations:
        h_t = GRU(e_t, h_{t-1})

    This allows the metacontroller to remember relevant information
    about the history at test-time when future context isn't available.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gates: reset and update
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_dim]
            h: Hidden state [batch, hidden_dim]
        Returns:
            New hidden state [batch, hidden_dim]
        """
        combined = torch.cat([x, h], dim=-1)

        r = torch.sigmoid(self.W_r(combined))  # Reset gate
        z = torch.sigmoid(self.W_z(combined))  # Update gate

        combined_reset = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_reset))  # Candidate

        h_new = (1 - z) * h + z * h_tilde
        return h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class SequenceEmbedder(nn.Module):
    """
    Internal Sequence Embedder.

    Creates an acausal embedding s(e_{1:T}) by processing the ENTIRE sequence
    of residual stream activations. This is crucial for learning abstract actions
    because goals only materialize over complete trajectories.

    Example: An agent heading toward "red" might take the same first steps as
    one heading toward "green". Only by seeing the whole trajectory can we
    identify the true goal.

    Implementation: Simple bidirectional processing or mean pooling.
    The paper uses an SSM; we'll use a simpler attention-based approach.
    """

    def __init__(self, embed_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Project down then process
        self.input_proj = nn.Linear(embed_dim, output_dim)

        # Simple transformer encoder for sequence processing (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=4,
            dim_feedforward=output_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, e_seq: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            e_seq: Sequence of residual activations [batch, seq_len, embed_dim]
            attention_mask: Optional mask for padding [batch, seq_len]
        Returns:
            Sequence embedding [batch, output_dim]
        """
        # Project to working dimension
        x = self.input_proj(e_seq)  # [batch, seq_len, output_dim]

        # Bidirectional processing
        if attention_mask is not None:
            # Convert to transformer mask format (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.output_proj(x)  # [batch, output_dim]


class ControllerEncoder(nn.Module):
    """
    Controller Encoder.

    Produces parameters (μ_t, Σ_t) for the latent code proposal distribution:
        z̃_t ~ N(μ_t, Σ_t)

    The encoder is conditioned on:
    1. Current residual activation e_t (what's happening now)
    2. History state h_{t-1} (what happened before)
    3. Sequence embedding s(e_{1:T}) (ACAUSAL - the whole trajectory)

    The acausal conditioning is KEY: it allows the encoder to know what
    abstract action is needed BEFORE it's fully executed.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        gru_dim: int,
        seq_embed_dim: int,
        latent_dim: int
    ):
        super().__init__()

        input_dim = embed_dim + gru_dim + seq_embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate heads for mean and log-variance
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        s_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            e_t: Current residual activation [batch, embed_dim]
            h_t: GRU hidden state [batch, gru_dim]
            s_embed: Sequence embedding [batch, seq_embed_dim]
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log-variance (diagonal) [batch, latent_dim]
        """
        x = torch.cat([e_t, h_t, s_embed], dim=-1)
        hidden = self.encoder(x)

        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)

        return mu, logvar

    def sample(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Reparameterization trick for sampling.

        z̃ = μ + σ * ε, where ε ~ N(0, I)

        This allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mu + std * eps


class SwitchingUnit(nn.Module):
    """
    Switching Unit.

    Produces the temporal integration rate β_t ∈ [0, 1]:
    - β_t ≈ 1: Switch to a new abstract action
    - β_t ≈ 0: Continue with the current abstract action

    IMPORTANT: The switching unit is CAUSAL (no future information).
    This allows it to work at test time when we don't know the future.

    The paper shows that despite no explicit regularization, the switching
    gate learns to behave in a quasi-binary, sparsely-switching fashion
    that aligns with actual subgoal boundaries!
    """

    def __init__(
        self,
        embed_dim: int,
        gru_dim: int,
        latent_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()

        input_dim = embed_dim + gru_dim + latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single scalar output
        )

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        z_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            e_t: Current residual activation [batch, embed_dim]
            h_t: GRU hidden state [batch, gru_dim]
            z_prev: Previous latent code [batch, latent_dim]
        Returns:
            beta: Switching probability [batch, 1]
        """
        x = torch.cat([e_t, h_t, z_prev], dim=-1)
        logit = self.net(x)
        beta = torch.sigmoid(logit)
        return beta


class ControllerDecoder(nn.Module):
    """
    Controller Decoder (Hypernetwork).

    Maps latent codes z_t to controller matrices U_t:
        U_t = f_hyp(z_t)

    The controller U_t modifies the residual stream:
        ê_{t,l} = e_{t,l} + U_t @ e_{t,l}

    For efficiency, we use LOW-RANK controllers (like LoRA):
        U_t = A_t @ B_t where A_t ∈ R^{d×r}, B_t ∈ R^{r×d}

    This is a hypernetwork because it outputs the WEIGHTS of another network.
    """

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        rank: int,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.rank = rank

        # Network that produces low-rank factors
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Heads for A and B matrices
        self.A_head = nn.Linear(hidden_dim, embed_dim * rank)
        self.B_head = nn.Linear(hidden_dim, rank * embed_dim)

        # Scaling factor (like LoRA)
        self.scale = 1.0 / rank

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Latent code [batch, latent_dim]
        Returns:
            A: First factor [batch, embed_dim, rank]
            B: Second factor [batch, rank, embed_dim]
        """
        hidden = self.net(z)

        A = self.A_head(hidden).view(-1, self.embed_dim, self.rank)
        B = self.B_head(hidden).view(-1, self.rank, self.embed_dim)

        return A * self.scale, B

    def apply_controller(
        self,
        e: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the low-rank controller to residual activations.

        ê = e + A @ B @ e = e + U @ e

        Args:
            e: Residual activations [batch, embed_dim]
            A: First factor [batch, embed_dim, rank]
            B: Second factor [batch, rank, embed_dim]
        Returns:
            Modified residual [batch, embed_dim]
        """
        # e: [batch, embed_dim] -> [batch, embed_dim, 1]
        e_col = e.unsqueeze(-1)

        # U @ e = A @ (B @ e)
        # B @ e: [batch, rank, 1]
        # A @ (B @ e): [batch, embed_dim, 1]
        delta = torch.bmm(A, torch.bmm(B, e_col)).squeeze(-1)

        return e + delta


class Metacontroller(nn.Module):
    """
    Full Metacontroller Module.

    Combines all components to:
    1. Process sequence to get global embedding (training only)
    2. Maintain history with GRU
    3. Encode latent proposals
    4. Decide switching
    5. Temporally integrate latent codes
    6. Decode to controller matrices

    Training Mode (acausal):
    - Has access to full sequence embedding s(e_{1:T})
    - Can learn what abstract action is needed

    Inference Mode (causal):
    - No future information
    - Uses learned switching and prior sampling
    """

    def __init__(
        self,
        embed_dim: int,          # Dimension of residual stream (e.g., 896 for Qwen-0.5B)
        latent_dim: int = 16,    # Dimension of latent code z
        gru_dim: int = 64,       # GRU hidden dimension
        seq_embed_dim: int = 64, # Sequence embedding dimension
        encoder_hidden: int = 64,
        decoder_hidden: int = 64,
        switch_hidden: int = 64,
        controller_rank: int = 16,
        seq_embed_layers: int = 2
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.gru_dim = gru_dim
        self.seq_embed_dim = seq_embed_dim

        # Components
        self.gru = GRUCell(embed_dim, gru_dim)
        self.sequence_embedder = SequenceEmbedder(embed_dim, seq_embed_dim, seq_embed_layers)
        self.encoder = ControllerEncoder(embed_dim, encoder_hidden, gru_dim, seq_embed_dim, latent_dim)
        self.switching_unit = SwitchingUnit(embed_dim, gru_dim, latent_dim, switch_hidden)
        self.decoder = ControllerDecoder(latent_dim, embed_dim, controller_rank, decoder_hidden)

        # Initial latent code (learnable)
        self.z_init = nn.Parameter(torch.zeros(latent_dim))

    def forward_training(
        self,
        residual_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass during metacontroller training (acausal mode).

        Args:
            residual_sequence: [batch, seq_len, embed_dim]
            attention_mask: [batch, seq_len] (1 = valid, 0 = padding)
            temperature: Sampling temperature for z

        Returns:
            Dictionary containing:
            - z_sequence: Latent codes [batch, seq_len, latent_dim]
            - mu_sequence: Encoder means [batch, seq_len, latent_dim]
            - logvar_sequence: Encoder log-variances [batch, seq_len, latent_dim]
            - beta_sequence: Switching gates [batch, seq_len, 1]
            - A_sequence: Controller A factors [batch, seq_len, embed_dim, rank]
            - B_sequence: Controller B factors [batch, seq_len, rank, embed_dim]
            - controlled_sequence: Modified residuals [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = residual_sequence.shape
        device = residual_sequence.device

        # Get acausal sequence embedding (sees full sequence)
        s_embed = self.sequence_embedder(residual_sequence, attention_mask)  # [batch, seq_embed_dim]

        # Initialize states
        h = self.gru.init_hidden(batch_size, device)
        z = self.z_init.unsqueeze(0).expand(batch_size, -1)  # [batch, latent_dim]

        # Storage
        z_list, mu_list, logvar_list, beta_list = [], [], [], []
        A_list, B_list, controlled_list = [], [], []

        for t in range(seq_len):
            e_t = residual_sequence[:, t, :]  # [batch, embed_dim]

            # Update GRU (causal history)
            h = self.gru(e_t, h)

            # Encode latent proposal (acausal - uses s_embed)
            mu, logvar = self.encoder(e_t, h, s_embed)
            z_proposal = self.encoder.sample(mu, logvar, temperature)

            # Get switching probability (causal)
            beta = self.switching_unit(e_t, h, z)

            # Temporal integration
            # z_t = β_t * z̃_t + (1 - β_t) * z_{t-1}
            z = beta * z_proposal + (1 - beta) * z

            # Decode to controller
            A, B = self.decoder(z)

            # Apply controller to residual
            e_controlled = self.decoder.apply_controller(e_t, A, B)

            # Store
            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)
            beta_list.append(beta)
            A_list.append(A)
            B_list.append(B)
            controlled_list.append(e_controlled)

        return {
            'z_sequence': torch.stack(z_list, dim=1),
            'mu_sequence': torch.stack(mu_list, dim=1),
            'logvar_sequence': torch.stack(logvar_list, dim=1),
            'beta_sequence': torch.stack(beta_list, dim=1),
            'A_sequence': torch.stack(A_list, dim=1),
            'B_sequence': torch.stack(B_list, dim=1),
            'controlled_sequence': torch.stack(controlled_list, dim=1)
        }

    def forward_step(
        self,
        e_t: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
        sample_z: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Single step forward pass for inference/RL (causal mode).

        In internal RL, we don't have the sequence embedding, so we:
        1. Use the prior p(z) = N(0, I) for proposals (or a learned policy)
        2. Still use learned switching

        Args:
            e_t: Current residual [batch, embed_dim]
            h: GRU state [batch, gru_dim]
            z: Previous latent [batch, latent_dim]
            sample_z: Whether to sample new z from prior

        Returns:
            Dictionary with updated states and controller
        """
        batch_size = e_t.shape[0]
        device = e_t.device

        # Update GRU
        h_new = self.gru(e_t, h)

        # Get switching probability
        beta = self.switching_unit(e_t, h_new, z)

        if sample_z:
            # Sample from prior for proposal
            z_proposal = torch.randn(batch_size, self.latent_dim, device=device)
        else:
            z_proposal = z  # Keep current

        # Temporal integration
        z_new = beta * z_proposal + (1 - beta) * z

        # Decode to controller
        A, B = self.decoder(z_new)

        # Apply controller
        e_controlled = self.decoder.apply_controller(e_t, A, B)

        return {
            'h': h_new,
            'z': z_new,
            'beta': beta,
            'A': A,
            'B': B,
            'e_controlled': e_controlled
        }

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize GRU state and latent code."""
        h = self.gru.init_hidden(batch_size, device)
        z = self.z_init.unsqueeze(0).expand(batch_size, -1)
        return h, z

    def compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from prior N(0, I).

        D_KL(N(μ, σ²) || N(0, I)) = 0.5 * (μ² + σ² - log(σ²) - 1)

        This regularizes the latent space to be structured and
        ensures we can sample from the prior during RL.
        """
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        return kl.sum(dim=-1).mean()


class AbstractActionPolicy(nn.Module):
    """
    Abstract Action Policy for Internal RL.

    During internal RL, we replace the acausal encoder with a CAUSAL policy
    that outputs abstract actions z given the residual stream history.

    This policy is what gets trained by RL. Everything else is frozen.

    Architecture: Simple recurrent policy (like the paper's 1-layer SSM)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 16
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Simple GRU-based policy
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Output heads
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(
        self,
        e: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            e: Residual activation [batch, embed_dim] or [batch, seq, embed_dim]
            h: Hidden state [1, batch, hidden_dim]
        Returns:
            mu: Mean [batch, latent_dim]
            logvar: Log-variance [batch, latent_dim]
            h_new: New hidden state
        """
        if e.dim() == 2:
            e = e.unsqueeze(1)  # [batch, 1, embed_dim]

        output, h_new = self.gru(e, h)
        output = output[:, -1, :]  # [batch, hidden_dim]

        mu = self.mu_head(output)
        logvar = self.logvar_head(output)

        return mu, logvar, h_new

    def sample(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Sample z from the policy distribution."""
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def log_prob(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of z under the policy.

        log N(z; μ, σ²) = -0.5 * (log(2π) + log(σ²) + (z-μ)²/σ²)
        """
        var = logvar.exp()
        log_prob = -0.5 * (
            math.log(2 * math.pi) +
            logvar +
            (z - mu).pow(2) / var
        )
        return log_prob.sum(dim=-1)  # Sum over latent dimensions

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


# ============================================================================
# ELBO Loss for Metacontroller Training
# ============================================================================

def compute_elbo_loss(
    action_logits: torch.Tensor,
    target_actions: torch.Tensor,
    mu_sequence: torch.Tensor,
    logvar_sequence: torch.Tensor,
    kl_weight: float = 0.1,
    attention_mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute the ELBO loss for metacontroller training.

    ELBO = Σ_t [ log p(a_t | z_t, e_{1:t}) - α * D_KL(q(z_t) || p(z_t)) ]

    We maximize ELBO, so we minimize -ELBO.

    Args:
        action_logits: Predicted action logits [batch, seq_len, vocab_size]
        target_actions: Ground truth actions [batch, seq_len]
        mu_sequence: Encoder means [batch, seq_len, latent_dim]
        logvar_sequence: Encoder log-variances [batch, seq_len, latent_dim]
        kl_weight: Weight α for KL term
        attention_mask: [batch, seq_len]

    Returns:
        Dictionary with loss components
    """
    batch_size, seq_len, vocab_size = action_logits.shape

    # Reconstruction loss (negative log-likelihood)
    nll = F.cross_entropy(
        action_logits.view(-1, vocab_size),
        target_actions.view(-1),
        reduction='none'
    ).view(batch_size, seq_len)

    # KL divergence per timestep
    kl = 0.5 * (mu_sequence.pow(2) + logvar_sequence.exp() - logvar_sequence - 1)
    kl = kl.sum(dim=-1)  # [batch, seq_len]

    # Apply mask if provided
    if attention_mask is not None:
        mask = attention_mask.float()
        nll = (nll * mask).sum() / mask.sum()
        kl = (kl * mask).sum() / mask.sum()
    else:
        nll = nll.mean()
        kl = kl.mean()

    # Total loss (negative ELBO)
    loss = nll + kl_weight * kl

    return {
        'loss': loss,
        'nll': nll,
        'kl': kl
    }


# ============================================================================
# Utility Functions
# ============================================================================

def extract_residuals_at_layer(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int
) -> torch.Tensor:
    """
    Extract residual stream activations at a specific layer.

    This requires hooking into the model's forward pass.
    Implementation depends on the specific model architecture.
    """
    residuals = []

    def hook_fn(module, input, output):
        # output is typically (hidden_states, ...) tuple
        if isinstance(output, tuple):
            residuals.append(output[0].detach())
        else:
            residuals.append(output.detach())

    # Register hook on the target layer
    # For Qwen: model.model.layers[layer_idx]
    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    return residuals[0] if residuals else None


if __name__ == "__main__":
    # Quick test
    print("Testing Metacontroller components...")

    batch_size = 4
    seq_len = 50
    embed_dim = 896  # Qwen-0.5B hidden dim
    latent_dim = 16

    # Create metacontroller
    mc = Metacontroller(
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        gru_dim=64,
        seq_embed_dim=64
    )

    # Test forward pass
    residuals = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len)

    outputs = mc.forward_training(residuals, mask)

    print(f"z_sequence shape: {outputs['z_sequence'].shape}")
    print(f"beta_sequence shape: {outputs['beta_sequence'].shape}")
    print(f"controlled_sequence shape: {outputs['controlled_sequence'].shape}")

    # Check beta values
    beta_mean = outputs['beta_sequence'].mean().item()
    print(f"Mean beta (switching rate): {beta_mean:.3f}")

    print("\nAll tests passed!")
