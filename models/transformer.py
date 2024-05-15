import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from models.multiscale_attention import MS_Attention
from multi_head_attention import MultiheadAttention
from utils.pc_util import shift_scale_points


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, prenorm=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed
        self.prenorm = prenorm
        
    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_pre(self, query, key, query_pos, key_pos, point_cloud_dims=None, return_weight=False):
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        query2 = self.norm1(query)
        q = k = v = self.with_pos_embed(query2, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)

        query2 = self.norm2(query)
        query2, attn_weight = self.multihead_attn(query=self.with_pos_embed(query2, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))
        query = query + self.dropout2(query2)

        query2 = self.norm3(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query2))))
        query = query + self.dropout3(query2)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        if return_weight:
            return query, attn_weight
        return query

    def forward_post(self, query, key, query_pos, key_pos, point_cloud_dims=None, return_weight=False):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]

        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        q = k = v = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2, attn_weight = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        if return_weight:
            return query, attn_weight
        return query

    def forward(self, query, key, query_pos, key_pos, point_cloud_dims=None, return_weight=False):
        if self.prenorm:
            return self.forward_pre(query, key, query_pos, key_pos, point_cloud_dims=point_cloud_dims, return_weight=return_weight)
        else:
            return self.forward_post(query, key, query_pos, key_pos, point_cloud_dims=point_cloud_dims, return_weight=return_weight)



class MSTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, prenorm=False, sr_ratio=[2.0], base_np=1024):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MS_Attention(d_model, nhead, attn_drop=dropout, sr_ratio=sr_ratio, base_np=base_np)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed
        self.prenorm = prenorm


    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_pre(self, query, key, query_pos, key_pos, opt_ms_input, xyz_upsample=None):
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        query2 = self.norm1(query)
        q = k = v = self.with_pos_embed(query2, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)

        query2 = self.norm2(query)
        query2 = self.multihead_attn(self.with_pos_embed(query2, query_pos_embed),
                                     self.with_pos_embed(key, key_pos_embed),
                                     key_pos[:, :, :3], opt_ms_input, xyz_upsample
                                     )
        query = query + self.dropout2(query2)

        query2 = self.norm3(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query2))))
        query = query + self.dropout3(query2)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query

    def forward_post(self, query, key, query_pos, key_pos, opt_ms_input, xyz_upsample=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param xyz_all: [B P_all 3]

        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        q = k = v = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(self.with_pos_embed(query2, query_pos_embed),
                                     self.with_pos_embed(key, key_pos_embed),
                                     key_pos[:, :, :3], opt_ms_input, xyz_upsample
                                     )[0]

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query

    def forward(self, query, key, query_pos, key_pos, opt_ms_input, xyz_upsample=None):
        if self.prenorm:
            return self.forward_pre(query, key, query_pos, key_pos, opt_ms_input, xyz_upsample)
        else:
            return self.forward_post(query, key, query_pos, key_pos, opt_ms_input, xyz_upsample)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
