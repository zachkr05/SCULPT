"""Scene-conditioned trajectory diffusion transformer. Requires PyTorch >= 2.0.

Architecture only: 12 bidirectional transformer blocks by default, with
diffusion-step and optional preference conditioning through adaLN-Zero.

Input contract (B = batch size, N = padded obstacle count, H = horizon):
    noisy_trajectory:             float [B, H, 2]; H defaults to 300
    diffusion_step:               scalar or [B]; the diffusion noise level
    scene['occupancy_grid']:     [B, 128, 128], values 0/1
    scene['esdf']:               [B, 128, 128], distances in grid cells
    scene['semantic_map']:       long [B, 128, 128], class IDs 0..3
    scene['obstacle_metadata']:  float [B, N, 6], columns:
                                 x, y, orientation_radians, class_id, width, height
    scene['robot_pos']:          float [B, 2], coordinates in grid cells
    scene['goal_pos']:           float [B, 2], coordinates in grid cells
    conditioning:                optional float [B, conditioning_dim]
    returns:                     predicted noise [B, H, 2]

Class IDs: 0 = empty/padding, 1 = table, 2 = chair, 3 = person.
Use pack_obstacles() for the lists of dictionaries returned by gen_sample().
All other scene inputs must already be batched tensors on the model's device.
Maps preserve the generator's [x, y] axis order; do not transpose them.

Normalize CLEAN trajectory coordinates with 2 * xy / 127 - 1 BEFORE applying
your diffusion process. Pass the resulting noisy trajectory directly here.
Scene coordinates and ESDF remain in raw grid units and are normalized inside.
The diffusion scheduler, loss, sampling, and endpoint clamping live outside
this file. The model predicts epsilon; it does not directly output clean xy.

Pretraining: conditioning=None (or an all-zero vector) means no preference;
the preference branch contributes exactly zero, leaving the base model intact.
Fine-tuning: load the pretrained state_dict, provide a preference vector, and
use the USER-EDITED trajectory as the clean diffusion target. The vector must
also be available at inference (e.g. a user embedding learned during fine-tuning).
No network layers need to be replaced, and every parameter remains trainable.
The architecture itself does not infer a preference vector from an edit.

The clean original_trajectory is a training target, not a scene input.
Optional repulsive_potential and original_control_points are not consumed.

Example:
    model = TrajectoryDiffusionTransformer()
    # scene contains the batched tensors described above.
    noise_prediction = model(noisy_trajectory, diffusion_step, scene)
    noise_prediction = model(noisy_trajectory, diffusion_step, scene,
                             conditioning=preference_vector)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


CLASS_IDS = {"table": 1, "chair": 2, "person": 3}


def pack_obstacles(
    metadata_batch: Sequence[Sequence[Mapping]],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Pack a batch of gen_sample()['obstacle_metadata'] lists as [B, N, 6].

    Example: obstacles = pack_obstacles([sample['obstacle_metadata'], ...])
    Zero-padded rows have class ID 0 and are automatically masked by the model.
    """
    if not metadata_batch:
        raise ValueError("metadata_batch must contain at least one scene")
    max_objects = max(len(objects) for objects in metadata_batch)
    packed = torch.zeros(len(metadata_batch), max_objects, 6, dtype=torch.float32)
    for batch_index, objects in enumerate(metadata_batch):
        for object_index, obj in enumerate(objects):
            xy = torch.as_tensor(obj["position"], dtype=torch.float32).reshape(2)
            wh = torch.as_tensor(obj["dimensions"], dtype=torch.float32).reshape(2)
            class_id = obj.get("class_id")
            if class_id is None:
                class_id = CLASS_IDS[obj["class"]]
            if int(class_id) not in CLASS_IDS.values():
                raise ValueError("Obstacle class_id must be1, 2, or 3")
            packed[batch_index, object_index] = torch.stack(
                [xy[0], xy[1], torch.tensor(float(obj["orientation"])),
                 torch.tensor(float(class_id)), wh[0], wh[1]]
            )
    return packed.to(device=device)


def sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    """Separate uses encode diffusion noise levels and trajectory positions."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = values.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DiffusionTransformerBlock(nn.Module):
    """GPT-style attention/MLP residual block with full attention and adaLN-Zero."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: Tensor, context: Tensor, padding_mask: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(context).chunk(6, dim=-1)
        )
        qkv = modulate(self.norm_attention(x), shift_a, scale_a)
        attention_output = self.attention(
            qkv, qkv, qkv, key_padding_mask=padding_mask,
            need_weights=False, is_causal=False,
        )[0]
        x = x + gate_a[:, None, :] * attention_output
        mlp_input = modulate(self.norm_mlp(x), shift_m, scale_m)
        return x + gate_m[:, None, :] * self.mlp(mlp_input)


class TrajectoryDiffusionTransformer(nn.Module):
    """Jointly attend over map patches, objects, start/goal, and noisy waypoints.

    Default: 12 layers, width 384, 6 heads, MLP ratio 4, 16x16 map patches.
    The 128x128 map produces 64 patch tokens; each waypoint produces one token.
    `conditioning` is a global preference vector, independent of noise level.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (128, 128),
        patch_size: int = 16,
        max_horizon: int = 300,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        conditioning_dim: int = 32,
        semantic_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(grid_size) != 2 or min(grid_size) <= 1:
            raise ValueError("grid_size must contain two dimensions greater than 1")
        if patch_size <= 0 or any(size % patch_size for size in grid_size):
            raise ValueError("patch_size must divide both grid dimensions")
        if heads < 1 or dim < 2 or dim % heads:
            raise ValueError("dim must be >= 2 and divisible by heads")
        if min(depth, max_horizon, conditioning_dim, semantic_dim) < 1:
            raise ValueError("depth, max_horizon, conditioning_dim, semantic_dim must be positive")
        if mlp_ratio <= 0 or int(dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")

        self.grid_size = tuple(grid_size)
        self.max_horizon = max_horizon
        self.conditioning_dim = conditioning_dim
        num_patches = (grid_size[0] // patch_size) * (grid_size[1] // patch_size)
        self.register_buffer(
            "coordinate_scale", torch.tensor(grid_size, dtype=torch.float32) - 1
        )
        self.register_buffer(
            "trajectory_position",
            sinusoidal_embedding(torch.arange(max_horizon), dim).unsqueeze(0),
        )

        self.semantic_embedding = nn.Embedding(4, semantic_dim)
        self.patchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.map_projection = nn.Linear((2 + semantic_dim) * patch_size**2, dim)
        self.map_position = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.obstacle_class_embedding = nn.Embedding(4, semantic_dim)
        self.obstacle_projection = nn.Linear(6 + semantic_dim, dim)
        self.robot_projection = nn.Linear(2, dim)
        self.goal_projection = nn.Linear(2, dim)
        self.trajectory_projection = nn.Linear(2, dim)
        # Token roles: map, object, robot, goal, noisy trajectory.
        self.token_type = nn.Parameter(torch.zeros(5, dim))

        self.step_embedding = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        # No biases: zeros map to exactly zero throughout training/fine-tuning.
        self.preference_embedding = nn.Sequential(
            nn.Linear(conditioning_dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.output_projection = nn.Linear(dim, 2)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.map_position, std=0.02)
        nn.init.normal_(self.token_type, std=0.02)
        nn.init.normal_(self.semantic_embedding.weight, std=0.02)
        nn.init.normal_(self.obstacle_class_embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _normalize_xy(self, xy: Tensor) -> Tensor:
        return 2.0 * xy / self.coordinate_scale - 1.0

    def forward(
        self,
        noisy_trajectory: Tensor,
        diffusion_step: Tensor | int | float,
        scene: Mapping[str, Tensor],
        conditioning: Tensor | None = None,
    ) -> Tensor:
        """Return epsilon prediction [B, H, 2]; padded objects cannot be attended to.

        None and zero conditioning are identical when dropout is disabled (or
        under the same dropout realization). No causal attention mask is used.
        """
        if noisy_trajectory.ndim != 3 or noisy_trajectory.shape[-1] != 2:
            raise ValueError("noisy_trajectory must have shape [B, H, 2]")
        batch, horizon, _ = noisy_trajectory.shape
        if batch < 1 or not 1 <= horizon <= self.max_horizon:
            raise ValueError("Require B >= 1 and 1 <= H <= max_horizon")
        device = noisy_trajectory.device
        dtype = self.trajectory_projection.weight.dtype
        noisy_trajectory = noisy_trajectory.to(dtype=dtype)

        steps = torch.as_tensor(diffusion_step, device=device)
        if steps.ndim == 0:
            steps = steps.expand(batch)
        if steps.shape != (batch,):
            raise ValueError("diffusion_step must be a scalar or have shape [B]")
        context = self.step_embedding(
            sinusoidal_embedding(steps, self.token_type.shape[-1]).to(dtype=dtype)
        )
        if conditioning is not None:
            if conditioning.shape != (batch, self.conditioning_dim):
                raise ValueError("conditioning must have shape [B, conditioning_dim]")
            context = context + self.preference_embedding(
                conditioning.to(device=device, dtype=dtype)
            )

        occupancy = scene["occupancy_grid"].to(dtype=dtype)
        esdf = scene["esdf"].to(dtype=dtype)
        semantics = scene["semantic_map"].long()
        expected_map_shape = (batch, *self.grid_size)
        if any(x.shape != expected_map_shape for x in (occupancy, esdf, semantics)):
            raise ValueError(f"All scene maps must have shape {expected_map_shape}")
        semantic_features = self.semantic_embedding(semantics).permute(0, 3, 1, 2)
        distance_scale = math.hypot(*self.grid_size)
        map_features = torch.cat([
            occupancy[:, None], (esdf / distance_scale)[:, None], semantic_features
        ], dim=1)
        map_tokens = self.map_projection(self.patchify(map_features).transpose(1, 2))
        map_tokens = map_tokens + self.map_position + self.token_type[0]

        obstacles = scene["obstacle_metadata"].to(dtype=dtype)
        if obstacles.ndim != 3 or obstacles.shape[0] != batch or obstacles.shape[2] != 6:
            raise ValueError("obstacle_metadata must have shape [B, N, 6]")
        classes = obstacles[..., 3].long()
        obstacle_padding = classes.eq(0)
        
        # Ignore all padded feature values, including non-finite padding.
        obstacles = obstacles.masked_fill(obstacle_padding[..., None], 0.0)
        angle = obstacles[..., 2:3]
        obstacle_features = torch.cat([
            self._normalize_xy(obstacles[..., :2]),
            angle.sin(), angle.cos(),
            obstacles[..., 4:6] / self.coordinate_scale,
            self.obstacle_class_embedding(classes),
        ], dim=-1)
        obstacle_tokens = self.obstacle_projection(obstacle_features) + self.token_type[1]
        robot = scene["robot_pos"].to(dtype=dtype)
        goal = scene["goal_pos"].to(dtype=dtype)
        if robot.shape != (batch, 2) or goal.shape != (batch, 2):
            raise ValueError("robot_pos and goal_pos must have shape [B, 2]")
        robot_token = self.robot_projection(self._normalize_xy(robot))[:, None]
        goal_token = self.goal_projection(self._normalize_xy(goal))[:, None]
        trajectory_tokens = (
            self.trajectory_projection(noisy_trajectory)
            + self.trajectory_position[:, :horizon]
            + self.token_type[4]
        )
        tokens = torch.cat([
            map_tokens, obstacle_tokens,
            robot_token + self.token_type[2], goal_token + self.token_type[3],
            trajectory_tokens,
        ], dim=1)
        padding_mask = torch.cat([
            torch.zeros(batch, map_tokens.shape[1], device=device, dtype=torch.bool),
            obstacle_padding,
            torch.zeros(batch, 2 + horizon, device=device, dtype=torch.bool),
        ], dim=1)
        for block in self.blocks:
            tokens = block(tokens, context, padding_mask)

        trajectory_tokens = tokens[:, -horizon:]
        shift, scale = self.output_modulation(context).chunk(2, dim=-1)
        output = modulate(self.output_norm(trajectory_tokens), shift, scale)
        return self.output_projection(output)
 
