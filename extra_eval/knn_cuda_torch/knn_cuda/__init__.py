"""Exact k-nearest neighbours with the interface of unlimblue/KNN_CUDA 0.2 (whose repository
and wheel are no longer online). Only models/dvae.py (DGCNN, not on LSceneLLM's OpenScene
encoder path) uses it.

KNN(k, transpose_mode)(ref, query) -> (dist, indx)
  transpose_mode=False: ref (B, D, Nr), query (B, D, Nq) -> (B, k, Nq) each
  transpose_mode=True:  ref (B, Nr, D), query (B, Nq, D) -> (B, Nq, k) each
dist is the Euclidean distance, ascending.
"""
import torch
import torch.nn as nn


class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super().__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        with torch.no_grad():
            if not self._t:
                ref, query = ref.transpose(1, 2), query.transpose(1, 2)
            d = torch.cdist(query.float(), ref.float())  # B, Nq, Nr
            dist, indx = d.topk(self.k, dim=-1, largest=False, sorted=True)
            if not self._t:
                dist, indx = dist.transpose(1, 2).contiguous(), indx.transpose(1, 2).contiguous()
            return dist, indx.long()
