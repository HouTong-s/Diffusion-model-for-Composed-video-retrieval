from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import logging
import torch
from torch import nn
import numpy as np
import math


logger = logging.getLogger(__name__)


class Diffusion(nn.Module):
    def __init__(self, width, dropout=0.1, temp=1):
        super(Diffusion, self).__init__()
        self.width = width
        self.dropout = dropout
        self.temp = temp

        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.proj = nn.Linear(self.width, self.width)

        self.sequence_pos_encoder = PositionalEncoding(self.width, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.width, self.sequence_pos_encoder)

        self.decoder = nn.Sequential(
            nn.Linear(self.width * 2, self.width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.width * 2, 1),
        )

    def forward(self, x, timesteps, query_emb, target_emb):
        # num is the model's num
        # query_emb (batch, feat_size)
        # target_emb (batch, num+1,  feat_size)
        cond_emb = self.embed_timestep(timesteps).squeeze(1)  # batch, feat_size

        q = self.q_proj(query_emb + cond_emb)  # batch, 512
        k = self.k_proj(target_emb + cond_emb.unsqueeze(1))  # batch, num+1, feat_size
        v = self.v_proj(target_emb + cond_emb.unsqueeze(1))  # batch, num+1, feat_size

        weight = torch.einsum('bd,bvd->bv', [q, k])
        weight = weight + x
        weight = torch.softmax(weight, dim=-1)
        new_emb = torch.einsum('bv,bvd->bd', [weight, v])
        new_emb = self.proj(new_emb)

        emb = torch.cat([new_emb.unsqueeze(1).repeat(1, target_emb.size(1), 1), target_emb], dim=-1)  # batch, num+1, feat_size * 2

        p = self.decoder(emb).squeeze(2)  # batch, num+1
        p += weight

        return p


class Diffusion_v(nn.Module):
    def __init__(self, width, dropout=0.1, temp=1):
        super(Diffusion_v, self).__init__()
        self.width = width
        self.dropout = dropout
        self.temp = temp

        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.proj = nn.Linear(self.width, self.width)

        self.sequence_pos_encoder = PositionalEncoding(self.width, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.width, self.sequence_pos_encoder)

        self.decoder = nn.Sequential(
            nn.Linear(self.width * 2, self.width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.width * 2, 1),
        )

    def forward(self, x, timesteps, query_emb, target_emb):
        # num is the model's num
        # query_emb (batch, num+1, feat_size)
        # target_emb (batch, feat_size)
        cond_emb = self.embed_timestep(timesteps).squeeze(1)  # batch, feat_size

        q = self.q_proj(target_emb + cond_emb.unsqueeze(1))  # batch, num+1, feat_size
        k = self.k_proj(query_emb + cond_emb.unsqueeze(1))  # batch, num+1, feat_size
        v = self.v_proj(query_emb + cond_emb.unsqueeze(1))  # batch, num+1, feat_size

        weight = torch.einsum('bad,bad->ba', [q, k])
        weight = weight + x
        weight = torch.softmax(weight, dim=-1)
        new_emb = torch.einsum('ba,bad->bd', [weight, v])
        new_emb = self.proj(new_emb)

        emb = torch.cat([new_emb.unsqueeze(1).repeat(1, query_emb.size(1), 1), query_emb], dim=-1)  # batch, num+1, feat_size * 2

        p = self.decoder(emb).squeeze(2)  # batch, num+1
        p += weight
        return p


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])
