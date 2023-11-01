# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ball_query_forward', 'stack_ball_query_forward'])


class BallQuery(Function):
    """Find nearby points in spherical space."""

    @staticmethod
    def forward(
            ctx,
            min_radius: float,
            max_radius: float,
            sample_num: int,
            xyz: torch.Tensor,
            center_xyz: torch.Tensor,
            xyz_batch_cnt: Optional[torch.Tensor] = None,
            center_xyz_batch_cnt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            min_radius (float): minimum radius of the balls.
            max_radius (float): maximum radius of the balls.
            sample_num (int): maximum number of features in the balls.
            xyz (torch.Tensor): (B, N, 3) xyz coordinates of the features,
                or staked input (N1 + N2 ..., 3).
            center_xyz (torch.Tensor): (B, npoint, 3) centers of the ball
                query, or staked input (M1 + M2 ..., 3).
            xyz_batch_cnt: (batch_size): Stacked input xyz coordinates nums in
                each batch, just like (N1, N2, ...). Defaults to None.
                New in version 1.7.0.
            center_xyz_batch_cnt: (batch_size): Stacked centers coordinates
                nums in each batch, just line (M1, M2, ...). Defaults to None.
                New in version 1.7.0.

        Returns:
            torch.Tensor: (B, npoint, nsample) tensor with the indices of the
            features that form the query balls.
        """
        assert center_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert min_radius < max_radius
        if xyz_batch_cnt is not None and center_xyz_batch_cnt is not None:
            assert xyz_batch_cnt.dtype == torch.int
            assert center_xyz_batch_cnt.dtype == torch.int
            idx = center_xyz.new_zeros((center_xyz.shape[0], sample_num),
                                       dtype=torch.int32)
            ext_module.stack_ball_query_forward(
                center_xyz,
                center_xyz_batch_cnt,
                xyz,
                xyz_batch_cnt,
                idx,
                max_radius=max_radius,
                nsample=sample_num,
            )
        else:
            B, N, _ = xyz.size()
            npoint = center_xyz.size(1)
            idx = xyz.new_zeros(B, npoint, sample_num, dtype=torch.int32)
            ext_module.ball_query_forward(
                center_xyz,
                xyz,
                idx,
                b=B,
                n=N,
                m=npoint,
                min_radius=min_radius,
                max_radius=max_radius,
                nsample=sample_num)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None) -> Tuple[None, None, None, None]:
        return None, None, None, None


ball_query = BallQuery.apply
