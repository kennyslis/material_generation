import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dims, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        device = timesteps.device
        emb = torch.exp(torch.arange(half_dim, device=device) * -factor)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = build_mlp([hidden_dim * 2 + 1, hidden_dim, hidden_dim])
        self.node_mlp = build_mlp([hidden_dim * 2, hidden_dim, hidden_dim])

    def forward(
        self,
        node_state: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = node_state.shape
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        relative = pos_i - pos_j
        dist = torch.norm(relative, dim=-1, keepdim=True)
        node_i = node_state.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = node_state.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_input = torch.cat([node_i, node_j, dist], dim=-1)
        edge_msg = self.edge_mlp(edge_input)

        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        edge_mask = edge_mask.unsqueeze(-1)
        edge_msg = edge_msg * edge_mask
        aggregated = edge_msg.sum(dim=2) / edge_mask.sum(dim=2).clamp_min(1.0)
        updated = self.node_mlp(torch.cat([node_state, aggregated], dim=-1))
        return node_state + updated * mask.unsqueeze(-1)


class PropertyHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = build_mlp([hidden_dim, hidden_dim, hidden_dim // 2, out_dim], dropout=0.05)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(graph_embedding)


@dataclass
class DiffusionOutput:
    noise_loss: torch.Tensor
    property_loss: torch.Tensor
    total_loss: torch.Tensor
    metrics: Dict[str, float]


class CrystalDiffusionModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        diffusion_steps: int = 100,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = build_mlp([hidden_dim, hidden_dim, hidden_dim])
        self.cond_proj = build_mlp([condition_dim, hidden_dim, hidden_dim])
        self.node_proj = build_mlp([node_dim + 2, hidden_dim, hidden_dim])
        self.layers = nn.ModuleList([GraphMessageLayer(hidden_dim) for _ in range(num_layers)])

        self.feature_noise_head = build_mlp([hidden_dim, hidden_dim, node_dim])
        self.position_noise_head = build_mlp([hidden_dim, hidden_dim, 2])
        self.property_head = PropertyHead(hidden_dim, 4)

    def q_sample(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        timesteps: torch.Tensor,
        feature_noise: torch.Tensor,
        position_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        noisy_features = alpha_bar.sqrt() * node_features + (1 - alpha_bar).sqrt() * feature_noise
        noisy_positions = alpha_bar.sqrt() * positions + (1 - alpha_bar).sqrt() * position_noise
        return noisy_features, noisy_positions

    def encode(
        self,
        noisy_features: torch.Tensor,
        noisy_positions: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        node_state = self.node_proj(torch.cat([noisy_features, noisy_positions], dim=-1))
        time_emb = self.time_proj(self.time_embedding(timesteps)).unsqueeze(1)
        cond_emb = self.cond_proj(condition).unsqueeze(1)
        node_state = node_state + time_emb + cond_emb
        for layer in self.layers:
            node_state = layer(node_state, noisy_positions, mask)
        return node_state

    def graph_embedding(self, node_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_state = node_state * mask.unsqueeze(-1)
        return masked_state.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        targets: torch.Tensor,
        property_weights: torch.Tensor,
    ) -> DiffusionOutput:
        batch_size = node_features.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=node_features.device)
        feature_noise = torch.randn_like(node_features)
        position_noise = torch.randn_like(positions)
        noisy_features, noisy_positions = self.q_sample(
            node_features, positions, timesteps, feature_noise, position_noise
        )
        node_state = self.encode(noisy_features, noisy_positions, mask, condition, timesteps)

        pred_feature_noise = self.feature_noise_head(node_state)
        pred_position_noise = self.position_noise_head(node_state)
        graph_emb = self.graph_embedding(node_state, mask)
        pred_properties = self.property_head(graph_emb)

        feature_loss = ((pred_feature_noise - feature_noise) ** 2 * mask.unsqueeze(-1)).sum()
        feature_loss = feature_loss / mask.sum().clamp_min(1.0)
        position_loss = ((pred_position_noise - position_noise) ** 2 * mask.unsqueeze(-1)).sum()
        position_loss = position_loss / mask.sum().clamp_min(1.0)
        noise_loss = feature_loss + position_loss

        property_mse = (pred_properties - targets) ** 2
        property_loss = (property_mse * property_weights).mean()
        total_loss = noise_loss + property_loss

        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "noise_loss": float(noise_loss.detach().cpu()),
            "property_loss": float(property_loss.detach().cpu()),
            "delta_g_mae": float((pred_properties[:, 0] - targets[:, 0]).abs().mean().detach().cpu()),
            "stability_mae": float((pred_properties[:, 1:3] - targets[:, 1:3]).abs().mean().detach().cpu()),
            "synth_mae": float((pred_properties[:, 3] - targets[:, 3]).abs().mean().detach().cpu()),
        }
        return DiffusionOutput(
            noise_loss=noise_loss,
            property_loss=property_loss,
            total_loss=total_loss,
            metrics=metrics,
        )

    @torch.no_grad()
    def predict_properties(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = node_features.shape[0]
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=node_features.device)
        node_state = self.encode(node_features, positions, mask, condition, timesteps)
        return self.property_head(self.graph_embedding(node_state, mask))

    @torch.no_grad()
    def sample(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = node_features.shape[0]
        current_features = torch.randn_like(node_features)
        current_positions = torch.randn_like(positions)
        null_condition = torch.zeros_like(condition)

        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((batch_size,), step, device=node_features.device, dtype=torch.long)
            cond_state = self.encode(current_features, current_positions, mask, condition, timesteps)
            uncond_state = self.encode(current_features, current_positions, mask, null_condition, timesteps)

            cond_feature_noise = self.feature_noise_head(cond_state)
            cond_position_noise = self.position_noise_head(cond_state)
            uncond_feature_noise = self.feature_noise_head(uncond_state)
            uncond_position_noise = self.position_noise_head(uncond_state)

            pred_feature_noise = uncond_feature_noise + guidance_scale * (
                cond_feature_noise - uncond_feature_noise
            )
            pred_position_noise = uncond_position_noise + guidance_scale * (
                cond_position_noise - uncond_position_noise
            )

            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]

            current_features = (current_features - beta / (1 - alpha_bar).sqrt() * pred_feature_noise) / alpha.sqrt()
            current_positions = (current_positions - beta / (1 - alpha_bar).sqrt() * pred_position_noise) / alpha.sqrt()

            if step > 0:
                current_features = current_features + beta.sqrt() * torch.randn_like(current_features)
                current_positions = current_positions + beta.sqrt() * torch.randn_like(current_positions)

            current_features = current_features * mask.unsqueeze(-1)
            current_positions = current_positions * mask.unsqueeze(-1)

        return current_features, current_positions
