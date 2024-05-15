import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2.pointnet2_utils import *


class BoxQueryAndGroup(nn.Module):
    r"""
    Groups with a box query (based on each query's prediction box)

    Parameters
    ---------
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, cuda_ops=True):
        super(BoxQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.cuda_ops = cuda_ops

    def forward_cuda(self, key_xyz: torch.Tensor,
                key_features: torch.Tensor,
                query_xyz: torch.Tensor):
        r"""
        Parameters
        ----------
        key_xyz : torch.Tensor
            (B, N=1024, 3) tensor of the xyz coordinates of the features
        key_features : torch.Tensor
            (B, C, N=1024) tensor of the descriptors of the the features
        query_xyz : torch.Tensor
            (B, npoint=256, 6) query points positions [and sizes (L/H/W)] (the center of each group)
        """   
        local_group_idx = box_query(self.radius, self.nsample, key_xyz, query_xyz)
        local_group_mask = torch.where(local_group_idx==-1, True, False)
        # keep at least one (the first one) sample to not be masked, otherwise cross-attn will have Nan output
        local_group_mask[:, :, 0] = False
        # set negative indexes to 0, so that it can work with the grouping_operation
        local_group_idx = torch.where(local_group_idx==-1, torch.tensor(0).type_as(local_group_idx).cuda(), local_group_idx)

        xyz_trans = key_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, local_group_idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= query_xyz[:, :, :3].transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            raise NotImplementedError

        assert key_features is not None
        grouped_features = grouping_operation(key_features, local_group_idx)
        new_features = grouped_features

        ret = [grouped_xyz, new_features, local_group_mask]
        return tuple(ret)

    def forward_torch(self, key_xyz: torch.Tensor,
                key_features: torch.Tensor,
                query_xyz: torch.Tensor):

        B, nq, _ = query_xyz.size()
        N = key_xyz.size()[1]

        query_centers = query_xyz[:, :, :3].unsqueeze(2)
        box_boundaries = query_xyz[:, :, 3:].unsqueeze(2) #[B, nq, 1, 3]
        
        key_xyz_expand = key_xyz.unsqueeze(1) #[B, 1, N, 3]
        offsets = torch.abs(key_xyz_expand - query_centers) #[B, nq, N, 3]

        local_group_idx = torch.empty([B, nq, self.nsample], dtype=torch.int).to(key_xyz.device)
        local_group_mask = torch.zeros([B, nq, self.nsample], dtype=torch.bool, device=key_xyz.device) # False means not being masked
        zeros = torch.zeros_like(offsets[0, 0, :, 0])
        for b in range(B):
            for q in range(nq):
                idx = torch.nonzero(
                    torch.logical_and(
                        torch.logical_and(
                            torch.le(offsets[b, q, :, 0] - 0.5 * box_boundaries[b, q, :, 0], zeros), 
                            torch.le(offsets[b, q, :, 1] - 0.5 * box_boundaries[b, q, :, 1], zeros)), 
                            torch.le(offsets[b, q, :, 2] - 0.5 * box_boundaries[b, q, :, 2], zeros
                        )
                    )
                ) # [nsamples, 1]
                
                # If the idx is empty, do ball querys to get some samples
                if idx.size()[0] == 0:
                    idx = ball_query(self.radius, self.nsample, 
                                key_xyz_expand[:, 0, :, :], query_xyz[b, q, :3].unsqueeze(0).unsqueeze(0))[0, 0, :]
                    idx = idx.unsqueeze(1)
                # truncate or pad the idx to a fix length
                elif idx.size()[0] > self.nsample:
                    rnd_idx = torch.randperm(idx.size()[0])[:self.nsample]
                    idx = idx[rnd_idx]
                elif idx.size()[0] < self.nsample:
                    idx_length = idx.size()[0]
                    pad_length = self.nsample - idx_length
                    pad = torch.zeros([pad_length, 1]).type_as(idx)
                    idx = torch.cat([idx, pad], dim=0)
                    local_group_mask[b, q, idx_length:] = True # mask values being True means not being allowed to attend
                local_group_idx[b, q, :] = idx.squeeze(1)
        
        xyz_trans = key_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, local_group_idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= query_xyz[:, :, :3].transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            raise NotImplementedError

        assert key_features is not None
        grouped_features = grouping_operation(key_features, local_group_idx)
        new_features = grouped_features

        ret = [grouped_xyz, new_features, local_group_mask]
        return tuple(ret)

    def forward(self, key_xyz: torch.Tensor,
                key_features: torch.Tensor,
                query_xyz: torch.Tensor):
        if self.cuda_ops:
            return self.forward_cuda(key_xyz, key_features, query_xyz)
        else:
            return self.forward_torch(key_xyz, key_features, query_xyz)

class PointnetSampleGroup(nn.Module):
    def __init__(
            self,
            *,
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = False,
            normalize_xyz: bool = True, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            raise NotImplementedError

    def forward(self, key_xyz: torch.Tensor,
                key_features: torch.Tensor,
                query_xyz: torch.Tensor):
        r"""
        Parameters
        ----------
        key_xyz : torch.Tensor
            (B, N=1024, 3) tensor of the xyz coordinates of the features
        key_features : torch.Tensor
            (B, C, N=1024) tensor of the descriptors of the the features
        query_xyz : torch.Tensor
            (B, npoint=256, 6) query points positions [and sizes (L/H/W)] (the center of each group)
        """
        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(
                key_xyz, query_xyz[:, :, :3], key_features
            )  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                key_xyz, query_xyz[:, :, :3], key_features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        if not self.ret_unique_cnt:
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, grouped_features, unique_cnt


class LocalTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, n_point=256, n_sample=16, radius=0.2, 
                    dropout=0.1, use_ffn=True, pos_embed=None, cross_embed=None, prenorm=False,use_cuda=True):
        super().__init__()

        self.use_ffn = use_ffn
        self.n_point = n_point
        self.n_sample = n_sample
        self.radius = radius
        # a group module that samples nearby points around each query points 
        # query from [B, C, nq] to [B, C, nq, n_sample]
        if not use_cuda:
            print("using pytorch implementation, can be inefficient.")
        self.local_grouper = BoxQueryAndGroup(radius=radius, nsample=n_sample, cuda_ops=use_cuda)

        # an attention module that performs cross-attention between each query point 
            #  and its correpsonding the n_sample points, 
            #  Q:[1, B * nq, C], K, V: [n_sample, B * nq, C]  -> output: [1, B * nq, C]
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.query_pos_embed = pos_embed
        self.key_pos_embed = cross_embed
        self.prenorm = prenorm
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_post(self, query, key, query_pos, key_pos, return_weight=False):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        """

        # the grouped_xyz has been normalized around their query centers
        if isinstance(self.local_grouper, BoxQueryAndGroup):
            # mask [B, nq, nsamples]
            grouped_xyz, grouped_features, attn_mask = self.local_grouper(key_pos, key, query_pos) 
        else:
            grouped_xyz, grouped_features = self.local_grouper(key_pos, key, query_pos) # [B, C, nq, n_sample], [B, 3, nq, n_sample]
            attn_mask = None
        # pos embed
        key_pos = grouped_xyz.permute(0, 2, 3, 1).reshape(-1, self.n_sample, 3) # [B * nq, n_sample, 3]
        if self.key_pos_embed is not None:
            key_pos_embed = self.key_pos_embed(key_pos) # [B * nq, C, n_sample]
        else:
            key_pos_embed = None
        if self.query_pos_embed is not None:
            query_pos_embed = self.query_pos_embed(query_pos)
        else:
            query_pos_embed = None

        # reshape the q, k, v from [B, C, nq] to [n_sample, B * nq, C]
        B, C, nq, n_sample = grouped_features.size()
        key = grouped_features.permute(0, 2, 1, 3).reshape(-1, C, n_sample).permute(2, 0, 1) #(n_sample, b * nq, C)
        key_pos_embed = key_pos_embed.permute(2, 0, 1)
        query = query.permute(0, 2, 1).reshape(-1, C).unsqueeze(0) # [1, B*nq, C]
        query_pos_embed = query_pos_embed.permute(0, 2, 1).reshape(-1, C).unsqueeze(0) # [1, B*nq, C]
        if attn_mask is not None:
            # reshape from [B, nq, nsample] to [B*nq*n_attn_heads, 1, nsample]
            attn_mask = attn_mask.reshape(B*nq, 1, -1) # [B*nq, 1, nsample]
            attn_mask = attn_mask.repeat(1, self.nhead, 1).reshape(-1, n_sample) # [B*nq, nhead, nsample]
            attn_mask = attn_mask.unsqueeze(1) # [B*nq*n_attn_heads, 1, nsample]

        query2, attn_weight = self.cross_attn(query=self.with_pos_embed(query, query_pos_embed),
                                key=self.with_pos_embed(key, key_pos_embed),
                                value=self.with_pos_embed(key, key_pos_embed), 
                                attn_mask=attn_mask)
        query = query + self.dropout1(query2)
        # post norm and FFN
        query = self.norm1(query)
        if self.use_ffn:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.norm2(query)  # [1, B * nq, C]

        # restore the query shape
        query = query.reshape(B, nq, C)
        query = query.permute(0, 2, 1)
        if return_weight:
            return query, attn_weight.permute(1, 0, 2)
        return query

    def forward_pre(self, query, key, query_pos, key_pos, return_weight=False):
        # the grouped_xyz has been normalized around their query centers
        if isinstance(self.local_grouper, BoxQueryAndGroup):
            # mask [B, nq, nsamples]
            grouped_xyz, grouped_features, attn_mask = self.local_grouper(key_pos, key, query_pos) 
        else:
            grouped_xyz, grouped_features = self.local_grouper(key_pos, key, query_pos) # [B, C, nq, n_sample], [B, 3, nq, n_sample]
            attn_mask = None
        # pos embed
        key_pos = grouped_xyz.permute(0, 2, 3, 1).reshape(-1, self.n_sample, 3) # [B * nq, n_sample, 3]
        if self.key_pos_embed is not None:
            key_pos_embed = self.key_pos_embed(key_pos) # [B * nq, C, n_sample]
        else:
            key_pos_embed = None
        if self.query_pos_embed is not None:
            query_pos_embed = self.query_pos_embed(query_pos)
        else:
            query_pos_embed = None

        # reshape the q, k, v from [B, C, nq] to [n_sample, B * nq, C]
        B, C, nq, n_sample = grouped_features.size()
        key = grouped_features.permute(0, 2, 1, 3).reshape(-1, C, n_sample).permute(2, 0, 1) #(n_sample, b * nq, C)
        key_pos_embed = key_pos_embed.permute(2, 0, 1)
        query = query.permute(0, 2, 1).reshape(-1, C).unsqueeze(0) # [1, B*nq, C]
        query_pos_embed = query_pos_embed.permute(0, 2, 1).reshape(-1, C).unsqueeze(0) # [1, B*nq, C]

        # attn_mask: making points outside each query box
        if attn_mask is not None:
            # reshape from [B, nq, nsample] to [B*nq*n_attn_heads, 1, nsample]
            attn_mask = attn_mask.reshape(B*nq, 1, -1) # [B*nq, 1, nsample]
            attn_mask = attn_mask.repeat(1, self.nhead, 1).reshape(-1, n_sample) # [B*nq, nhead, nsample]
            attn_mask = attn_mask.unsqueeze(1) # [B*nq*n_attn_heads, 1, nsample]

        query2 = self.norm1(query)
        query2, attn_weight = self.cross_attn(query=self.with_pos_embed(query2, query_pos_embed),
                                key=self.with_pos_embed(key, key_pos_embed),
                                value=self.with_pos_embed(key, key_pos_embed), 
                                attn_mask=attn_mask)
        query = query + self.dropout1(query2)

        if self.use_ffn:
            query2 = self.norm2(query)
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query2))))
            query = query + self.dropout2(query2)

        # restore the query shape
        query = query.reshape(B, nq, C)
        query = query.permute(0, 2, 1)
        if return_weight:
            return query, attn_weight.permute(1, 0, 2)

        return query


    def forward(self, query, key, query_pos, key_pos, point_cloud_dims=None, return_weight=False):
        if self.prenorm:
            return self.forward_pre(query, key, query_pos, key_pos, return_weight=return_weight)
        else:
            return self.forward_post(query, key, query_pos, key_pos, return_weight=return_weight)


