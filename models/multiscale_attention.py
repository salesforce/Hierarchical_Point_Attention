from logging import raiseExceptions
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModuleUpSample


class MS_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., sr_ratio=[2.0], base_np=1024):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.base_np = base_np
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.kv1 = nn.Linear(dim, dim, bias=False)

        self.sr_ratio = sr_ratio
        self.ms_transform = nn.ModuleList([])
        self.kv2 = nn.ModuleList([])
        for r in self.sr_ratio:
            self.kv2.append(nn.Linear(dim, dim, bias=False))
            if r > 1:
                self.ms_transform.append(PointnetFPModuleUpSample(mlp=[dim, dim], n=int(self.base_np * r)))
            else:
                self.ms_transform.append(PointnetSAModuleVotes(mlp=[dim, dim], npoint=int(self.base_np * r),
                                                            radius=(0.2/r), nsample=int(16/r), 
                                                            use_xyz=True, normalize_xyz=True))
        self.nscale = len(self.sr_ratio) + 1

    def forward_v0(self, query, key, key_pos, xyz_all, xyz_upsample=None):
        Np, B, C = key.size()
        Nq, _, _ = query.size()
        q = self.q(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #[B, H, L, D]
        q_slice = q[:, :self.num_heads//self.nscale]
        _, nh, _, _ = q_slice.size()

        kv1 = self.kv1(key).reshape(B, -1, 2, nh, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1] #[B, H, L, D]
        attn1 = (q_slice @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, Nq, nh*C//self.num_heads)

        x = [x1]

        for i, m in enumerate(self.ms_transform):
            q_slice = q[:, (self.num_heads // self.nscale)*(i+1):(self.num_heads // self.nscale)*(i+2)] \
                                    if (i+2) < self.nscale else q[:, (self.num_heads // self.nscale)*(i+1):]
            _, nh, _, _ = q_slice.size()
            if isinstance(m, PointnetFPModuleUpSample):
                key2_pos, key2, ke2_ind = m(xyz_all, key_pos, key.permute(1, 2, 0).contiguous(), xyz_upsample)
            else:
                key2_pos, key2, ke2_ind = m(key_pos, key.permute(1, 2, 0).contiguous())
            key2 = key2.permute(2, 0, 1)
            kv2 = self.kv2[i](key2).reshape(B, -1, 2, nh, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k2, v2 = kv2[0], kv2[1]
            
            attn2 = (q_slice @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, Nq, nh*C//self.num_heads)
            x.append(x2)

        x = torch.cat(x, dim=-1)

        x = self.proj(x) # [B, Nq, C]

        return x.permute(1, 0, 2) #[Nq, B, C]

    def with_pos_embed(self, tensor, pos_embed: Optional[torch.Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed
    

    def forward(self, query, key, key_pos, xyz_all, xyz_upsample=None, aux_input=None):
        return self.forward_v0(query, key, key_pos, xyz_all, xyz_upsample)

