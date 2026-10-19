from __future__ import annotations

"""
中文注释说明：marl_models/attention.py

文件作用：
    实现注意力网络模块，用于增强多智能体状态交互建模能力。

整体流程：
    1. 读取全局配置、命令行参数或上游传入对象，准备实验所需的环境、模型与数据。
    2. 按本文件职责执行仿真、训练、评估、绘图或结果汇总等核心步骤。
    3. 将关键指标、模型参数或报告写入统一结果目录，便于论文实验复现和对比。

关键变量与对象：
    - CrossAttentionExtractor: 核心类，封装本模块中的主要状态和行为。
    - AttentionActorBase: 核心类，封装本模块中的主要状态和行为。
    - AttentionCriticBase: 核心类，封装本模块中的主要状态和行为。

主要依赖：
    config, marl_models, torch

注意事项：
    本文件新增的是解释性中文注释，不改变原有算法、参数默认值或文件读写路径。
"""

import config
from marl_models.buffer_and_helpers import layer_init
import torch
import torch.nn as nn
from torch.nn import functional as F


# 类 CrossAttentionExtractor，继承自 ：核心类，封装本模块中的主要状态和行为。
class CrossAttentionExtractor(nn.Module):
    """
    Cross-Attention Module used in actors and critics of attention-based models
    Standard Scaled Dot-Product Attention.
    Inputs:
        - self_embedding: The 'Query' (Agent's own state)
        - target_embeddings: The 'Keys/Values' (Neighbors or UEs)
    """

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：self_dim, target_dim。
    def __init__(self, self_dim: int, target_dim: int) -> None:
        super().__init__()
        self.head_dim: int = config.ATTN_HIDDEN_DIM // config.ATTN_NUM_HEADS
        assert config.ATTN_HIDDEN_DIM % config.ATTN_NUM_HEADS == 0, "hidden_dim must be divisible by num_heads"
        self.query_layer: nn.Linear = layer_init(nn.Linear(self_dim, config.ATTN_HIDDEN_DIM))

        self.key_layer: nn.Linear = layer_init(nn.Linear(target_dim, config.ATTN_HIDDEN_DIM))
        self.value_layer: nn.Linear = layer_init(nn.Linear(target_dim, config.ATTN_HIDDEN_DIM))
        self.scale: float = config.ATTN_HIDDEN_DIM ** (-0.5)  # Scaling factor for dot-product attention (1 / sqrt(d_k))
        self.out_proj: nn.Linear = layer_init(nn.Linear(config.ATTN_HIDDEN_DIM, config.ATTN_HIDDEN_DIM))

    # 函数 forward：定义神经网络前向传播计算，主要参数：self_embedding, target_embeddings, mask。
    def forward(self, self_embedding: torch.Tensor, target_embeddings: torch.Tensor, mask: torch.Tensor | None = None):
        # self_embedding: (batch, self_dim)
        # target_embeddings: (batch, max_targets, target_dim)
        batch_size: int = self_embedding.shape[0]

        # Linear Projections & Split Heads
        # Q: (batch, 1, hidden) -> (batch, 1, num_heads, head_dim) -> (batch, num_heads, 1, head_dim)
        Q: torch.Tensor = self.query_layer(self_embedding).unsqueeze(1).view(batch_size, 1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)

        # K, V: (batch, num_targets, hidden) -> (batch, num_targets, num_heads, head_dim) -> (batch, num_heads, num_targets, head_dim)
        K: torch.Tensor = self.key_layer(target_embeddings).view(batch_size, -1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)
        # 关键变量 V：全局常量或配置项，会影响环境规模、训练过程或实验输出。
        V: torch.Tensor = self.value_layer(target_embeddings).view(batch_size, -1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)

        # -- Original Manual Attention Implementation --

        # # Attention Scores
        # # (batch, 1, hidden) @ (batch, hidden, max_targets) -> (batch, 1, max_targets)
        # scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # if mask is not None:
        #     # Mask padding positions (set score to -infinity so Softmax becomes 0)
        #     # Mask: (batch, targets) -> (batch, 1, 1, targets)
        #     mask_expanded: torch.Tensor = mask.unsqueeze(1).unsqueeze(1)
        #     scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        # attn_weights: torch.Tensor = F.softmax(scores, dim=-1)

        # # Handle case where all targets are padding (e.g., no neighbors) -> nan check
        # attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # # Weighted Sum
        # context: torch.Tensor = torch.matmul(attn_weights, V)  # (batch, 1, hidden)

        # -- Refactored to use Pytorch's in-built Attention --

        attn_mask: torch.Tensor | None = None
        # 条件分支：根据当前配置、状态或评估结果选择不同处理路径。
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1).bool()

        context: torch.Tensor = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)

        context = torch.nan_to_num(context, nan=0.0)
        # -- Change over --

        # (batch, heads, 1, head_dim) -> (batch, 1, heads, head_dim) -> (batch, 1, hidden)
        context = context.transpose(1, 2).reshape(batch_size, 1, -1)

        # Final Projection
        output = self.out_proj(context)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return output.squeeze(1)

    def get_attention_weights(self, self_embedding: torch.Tensor, target_embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = self_embedding.shape[0]
        Q = self.query_layer(self_embedding).unsqueeze(1).view(batch_size, 1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)
        K = self.key_layer(target_embeddings).view(batch_size, -1, config.ATTN_NUM_HEADS, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask_expanded <= 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return weights.squeeze(2)


# 类 AttentionActorBase，继承自 ：核心类，封装本模块中的主要状态和行为。
class AttentionActorBase(nn.Module):
    """
    Base class for Attention-based Actors (Shared by MADDPG, MATD3, MASAC, MAPPO).
    Handles encoding, attention, and feature fusion.
    """

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim。
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.num_neighbors: int = config.MAX_UAV_NEIGHBORS
        self.neighbor_obs_dim: int = config.NEIGHBOR_OBS_DIM
        self.num_ues: int = config.MAX_ASSOCIATED_UES
        self.ue_obs_dim: int = config.UE_OBS_DIM
        self.hidden_dim: int = config.ATTN_HIDDEN_DIM
        self.mlp_dim: int = config.MLP_HIDDEN_DIM

        # Slicing flattened input
        self.neighbor_block_size: int = self.num_neighbors * self.neighbor_obs_dim
        self.ue_block_size: int = self.num_ues * self.ue_obs_dim
        self.own_dim: int = obs_dim - self.neighbor_block_size - self.ue_block_size

        # Feature Encoders (The "Embedding" Layers)
        # These project raw inputs (x, y, battery) into the shared hidden_dim
        # nn.Linear instead of nn.Embedding since inputs are continuous
        self.self_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.own_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())
        self.neighbor_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.neighbor_obs_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())
        self.ue_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(self.ue_obs_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())

        # Cross-Attention Modules
        self.neighbor_attn: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=self.hidden_dim, target_dim=self.hidden_dim)
        self.ue_attn: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=self.hidden_dim, target_dim=self.hidden_dim)

        # Fusion Layer
        self.fusion_dim: int = self.hidden_dim * 3
        self.fc1: nn.Linear = layer_init(nn.Linear(self.fusion_dim, self.mlp_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(self.mlp_dim)
        self.fc2: nn.Linear = layer_init(nn.Linear(self.mlp_dim, self.hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(self.hidden_dim)

    # 函数 get_feature_embedding：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_flat。
    def get_feature_embedding(self, obs_flat: torch.Tensor) -> torch.Tensor:
        batch_size: int = obs_flat.shape[0]
        own_state: torch.Tensor = obs_flat[:, : self.own_dim]
        neighbor_part: torch.Tensor = obs_flat[:, self.own_dim : self.own_dim + self.neighbor_block_size]
        neighbor_states: torch.Tensor = neighbor_part.reshape(batch_size, self.num_neighbors, self.neighbor_obs_dim)
        ue_part: torch.Tensor = obs_flat[:, self.own_dim + self.neighbor_block_size :]
        ue_states: torch.Tensor = ue_part.reshape(batch_size, self.num_ues, self.ue_obs_dim)

        # Generate Masks (0 for padding, 1 for real)
        # If absolute sum of features is 0 (or close), it's padding.
        neighbor_mask: torch.Tensor = (torch.abs(neighbor_states).sum(dim=-1) > 1e-5).float()
        ue_mask: torch.Tensor = (torch.abs(ue_states).sum(dim=-1) > 1e-5).float()

        # Encoding (Creating Embeddings)
        self_emb: torch.Tensor = self.self_encoder(own_state)
        neighbor_embs: torch.Tensor = self.neighbor_encoder(neighbor_states)
        ue_embs: torch.Tensor = self.ue_encoder(ue_states)

        # Attention
        neighbor_context: torch.Tensor = self.neighbor_attn(self_emb, neighbor_embs, mask=neighbor_mask)
        ue_context: torch.Tensor = self.ue_attn(self_emb, ue_embs, mask=ue_mask)

        # Fusion
        combined: torch.Tensor = torch.cat([self_emb, neighbor_context, ue_context], dim=1)
        fusion: torch.Tensor = F.relu(self.ln1(self.fc1(combined)))
        fusion = F.relu(self.ln2(self.fc2(fusion)))
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return fusion

    def extract_attention_weights(self, obs_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = obs_flat.shape[0]
        own_state = obs_flat[:, : self.own_dim]
        neighbor_part = obs_flat[:, self.own_dim : self.own_dim + self.neighbor_block_size]
        neighbor_states = neighbor_part.reshape(batch_size, self.num_neighbors, self.neighbor_obs_dim)
        ue_part = obs_flat[:, self.own_dim + self.neighbor_block_size :]
        ue_states = ue_part.reshape(batch_size, self.num_ues, self.ue_obs_dim)
        neighbor_mask = (torch.abs(neighbor_states).sum(dim=-1) > 1e-5).float()
        ue_mask = (torch.abs(ue_states).sum(dim=-1) > 1e-5).float()
        self_emb = self.self_encoder(own_state)
        neighbor_embs = self.neighbor_encoder(neighbor_states)
        ue_embs = self.ue_encoder(ue_states)
        return {
            "neighbor_weights": self.neighbor_attn.get_attention_weights(self_emb, neighbor_embs, mask=neighbor_mask),
            "ue_weights": self.ue_attn.get_attention_weights(self_emb, ue_embs, mask=ue_mask),
            "neighbor_mask": neighbor_mask,
            "ue_mask": ue_mask,
            "neighbor_states": neighbor_states,
            "ue_states": ue_states,
            "own_state": own_state,
        }


# 类 AttentionCriticBase，继承自 ：核心类，封装本模块中的主要状态和行为。
class AttentionCriticBase(nn.Module):
    """Base class for Attention-based Critics (Inspired from MAAC)"""

    # 函数 __init__：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_dim, action_dim。
    def __init__(self, obs_dim: int, action_dim: int = 0) -> None:
        super().__init__()
        self.hidden_dim: int = config.ATTN_HIDDEN_DIM
        self.num_heads: int = config.ATTN_NUM_HEADS
        self.mlp_dim: int = config.MLP_HIDDEN_DIM

        # Feature Extraction
        # Input: [Obs + Action] for Q-Critics (MADDPG/MATD3/MASAC)
        # Input: [Obs] for V-Critics (MAPPO)
        input_dim: int = obs_dim + action_dim

        self.state_encoder: nn.Sequential = nn.Sequential(layer_init(nn.Linear(input_dim, self.mlp_dim)), nn.LayerNorm(self.mlp_dim), nn.ReLU(), layer_init(nn.Linear(self.mlp_dim, self.hidden_dim)), nn.LayerNorm(self.hidden_dim), nn.ReLU())

        self.attention: CrossAttentionExtractor = CrossAttentionExtractor(self_dim=self.hidden_dim, target_dim=self.hidden_dim)

        self.fusion_dim: int = self.hidden_dim * 2

    # 函数 get_all_embeddings：关键函数，承载本模块的一段可复用实验逻辑，主要参数：inputs。
    def get_all_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return self.state_encoder(inputs)

    # 函数 attend_to_others：关键函数，承载本模块的一段可复用实验逻辑，主要参数：embeddings, num_agents, agent_index。
    def attend_to_others(self, embeddings: torch.Tensor, num_agents: int, agent_index: int) -> torch.Tensor:
        """Performs attention for agent i over all other agents."""
        # Extract "Me"
        me_embedding: torch.Tensor = embeddings[:, agent_index, :]

        # Extract "Others" (Masking logic)
        # We need to exclude 'agent_index' from the attention targets
        other_indices: list[int] = [j for j in range(num_agents) if j != agent_index]
        others_embeddings: torch.Tensor = embeddings[:, other_indices, :]

        # Attention
        # "Me" asks "How do the others affect my value?"
        context: torch.Tensor = self.attention(me_embedding, others_embeddings)

        # Fusion:
        combined: torch.Tensor = torch.cat([me_embedding, context], dim=1)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return combined

    # 函数 vectorized_attend_to_others：关键函数，承载本模块的一段可复用实验逻辑，主要参数：embeddings。
    def vectorized_attend_to_others(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes attention for all agents simultaneously, masking out self-attention.
        Args:
            embeddings: (Batch, Num_Agents, Hidden_Dim)
        Returns:
            combined: (Batch, Num_Agents, Fusion_Dim)
        """
        # Code repetition to support N-to-N attention compared to 1-to-N in self.attention.forward() and maintain clarity.
        batch_size, num_agents, _ = embeddings.shape

        # Pass through the linear layers of the existing attention module
        Q: torch.Tensor = self.attention.query_layer(embeddings)
        # 关键变量 K：全局常量或配置项，会影响环境规模、训练过程或实验输出。
        K: torch.Tensor = self.attention.key_layer(embeddings)
        # 关键变量 V：全局常量或配置项，会影响环境规模、训练过程或实验输出。
        V: torch.Tensor = self.attention.value_layer(embeddings)

        # Reshape for multi-head attention
        # (Batch, Num_Agents, Heads, Head_Dim) -> (Batch, Heads, Num_Agents, Head_Dim)
        Q = Q.view(batch_size, num_agents, self.num_heads, -1).transpose(1, 2)
        # 关键变量 K：全局常量或配置项，会影响环境规模、训练过程或实验输出。
        K = K.view(batch_size, num_agents, self.num_heads, -1).transpose(1, 2)
        # 关键变量 V：全局常量或配置项，会影响环境规模、训练过程或实验输出。
        V = V.view(batch_size, num_agents, self.num_heads, -1).transpose(1, 2)

        # Mask to exclude self-attention (agent i does not attend to agent i)
        # torch.eye puts 1s on the diagonal. `~` flips it so the diagonal is False.
        mask: torch.Tensor = ~torch.eye(num_agents, dtype=torch.bool, device=embeddings.device)

        # Compute for all agents at once
        context: torch.Tensor = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        # Reshape back to (Batch, Num_Agents, Hidden_Dim)
        context = context.transpose(1, 2).reshape(batch_size, num_agents, -1)
        context = self.attention.out_proj(context)

        # Fusion: concatenate original embeddings with the attended context
        combined: torch.Tensor = torch.cat([embeddings, context], dim=-1)
        # 返回结果：把本阶段计算出的指标、状态或对象交给上层流程继续使用。
        return combined

    # 函数 get_q_embedding：关键函数，承载本模块的一段可复用实验逻辑，主要参数：obs_tensor, action_tensor, agent_index。
    def get_q_embedding(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor, agent_index: int) -> torch.Tensor:
        """
        Calculates the embedding for agent i (for Q-value) by attending to all other agents.
        Args:
            obs_tensor: (Batch, Num_Agents, Obs_Dim)
            action_tensor: (Batch, Num_Agents, Action_Dim)
            agent_index: The index of the agent we are critiquing
        Returns:
            output_embedding: (Batch, Fusion_Dim)
        """
        num_agents: int = obs_tensor.shape[1]
        inputs: torch.Tensor = torch.cat([obs_tensor, action_tensor], dim=2)

        # Encode everyone
        embeddings: torch.Tensor = self.get_all_embeddings(inputs)

        # Attend to others
        return self.attend_to_others(embeddings, num_agents, agent_index)
