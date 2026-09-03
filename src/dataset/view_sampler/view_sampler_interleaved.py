from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerInterleavedCfg:
    name: Literal["interleaved"]
    num_context_views: int
    skip: int
    # Max images per GPU, used by DynamicBatchSampler to size dynamic batches.
    max_img_per_gpu: int = 24


class ViewSamplerInterleaved(ViewSampler[ViewSamplerInterleavedCfg]):
    def sample(
        self,
        scene: str,
        num_context_views: int,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        """Sample a contiguous block of ``num_context_views`` views and take every
        ``(skip + 1)``-th of them as targets. Returns (context, target, overlap)."""
        num_views, _, _ = extrinsics.shape

        # for n context views get n context
        n = self.cfg.num_context_views
        # then get skip value as s from config
        s = self.cfg.skip

        frames_needed = n

        if not self.cameras_are_circular:
             if num_views < frames_needed:
                 raise ValueError(f"Scene {scene} has {num_views} views, but interleaved sampler requires at least {frames_needed} views.")
             
             max_start = num_views - frames_needed
             start_idx = torch.randint(0, max_start + 1, size=(), device=device).item()
        else:
             start_idx = torch.randint(0, num_views, size=(), device=device).item()

        context_indices = []
        for i in range(n):
            idx = start_idx + i
            if self.cameras_are_circular:
                idx %= num_views
            context_indices.append(idx)
            
        # pick every s th frame from the chosen context frames as the target frames
        target_indices = context_indices[s::s+1]
            
        overlap = torch.tensor([0.5], dtype=torch.float32, device=device) # Dummy overlap

        return (
            torch.tensor(context_indices, dtype=torch.int64, device=device),
            torch.tensor(target_indices, dtype=torch.int64, device=device),
            overlap
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return (self.cfg.num_context_views + self.cfg.skip - 1) // self.cfg.skip
