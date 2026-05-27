"""Show, Attend and Tell decoder: soft-attention LSTM + beam search."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.feat_proj  = nn.Linear(feat_dim, attn_dim)
        self.hid_proj   = nn.Linear(hidden_dim, attn_dim)
        self.score      = nn.Linear(attn_dim, 1)

    def forward(self, feats: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feats [B, L, D], h [B, H]
        a = F.relu(self.feat_proj(feats) + self.hid_proj(h).unsqueeze(1))
        e = self.score(a).squeeze(-1)                 # [B, L]
        alpha = F.softmax(e, dim=1)                   # [B, L]
        ctx = (feats * alpha.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return ctx, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 2048,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        attn_dim: int = 512,
        dropout: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.feat_dim   = feat_dim
        self.pad_id     = pad_id

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attn  = SoftAttention(feat_dim, hidden_dim, attn_dim)
        self.lstm  = nn.LSTMCell(embed_dim + feat_dim, hidden_dim)
        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)
        self.gate   = nn.Linear(hidden_dim, feat_dim)   # beta gate from S-A-T paper
        self.dropout = nn.Dropout(dropout)
        self.out   = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = feats.mean(dim=1)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def forward(
        self,
        feats: torch.Tensor,           # [B, L, D]
        captions: torch.Tensor,        # [B, T] (teacher forcing input incl. <bos>)
        lengths: torch.Tensor,         # [B]
    ) -> dict:
        B, T = captions.shape
        device = feats.device

        h, c = self.init_state(feats)
        logits  = torch.zeros(B, T - 1, self.vocab_size, device=device)
        alphas  = torch.zeros(B, T - 1, feats.size(1), device=device)

        embed = self.embed(captions)   # [B, T, E]

        for t in range(T - 1):
            ctx, alpha = self.attn(feats, h)
            beta = torch.sigmoid(self.gate(h))
            ctx = beta * ctx
            x = torch.cat([embed[:, t, :], ctx], dim=-1)
            h, c = self.lstm(x, (h, c))
            logits[:, t, :] = self.out(self.dropout(h))
            alphas[:, t, :] = alpha
        return {"logits": logits, "alphas": alphas}

    @torch.no_grad()
    def generate_greedy(
        self,
        feats: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 22,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = feats.size(0)
        device = feats.device
        h, c = self.init_state(feats)
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        alphas: list[torch.Tensor] = []
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(max_len - 1):
            ctx, alpha = self.attn(feats, h)
            beta = torch.sigmoid(self.gate(h))
            ctx = beta * ctx
            x = torch.cat([self.embed(tokens[:, -1]), ctx], dim=-1)
            h, c = self.lstm(x, (h, c))
            logits = self.out(h)
            nxt = logits.argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            alphas.append(alpha)
            done = done | (nxt == eos_id)
            if done.all():
                break
        return tokens, torch.stack(alphas, dim=1)   # [B, T-1], [B, T-1, L]

    @torch.no_grad()
    def generate_beam(
        self,
        feats: torch.Tensor,          # [1, L, D] — beam search runs per-image
        bos_id: int,
        eos_id: int,
        beam: int = 3,
        max_len: int = 22,
        length_penalty: float = 0.7,
    ) -> tuple[list[int], torch.Tensor]:
        assert feats.size(0) == 1
        device = feats.device
        L, D = feats.size(1), feats.size(2)
        h, c = self.init_state(feats)
        # expand to beam
        feats_b = feats.expand(beam, L, D).contiguous()
        h = h.expand(beam, -1).contiguous()
        c = c.expand(beam, -1).contiguous()

        seqs = torch.full((beam, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(beam, device=device)
        scores[1:] = float("-inf")
        finished: list[tuple[float, list[int], torch.Tensor]] = []
        alphas_per_beam: list[list[torch.Tensor]] = [[] for _ in range(beam)]

        for t in range(max_len - 1):
            ctx, alpha = self.attn(feats_b, h)
            beta = torch.sigmoid(self.gate(h))
            ctx = beta * ctx
            x = torch.cat([self.embed(seqs[:, -1]), ctx], dim=-1)
            h, c = self.lstm(x, (h, c))
            logits = self.out(h)
            logp = F.log_softmax(logits, dim=-1)
            cand_scores = scores.unsqueeze(1) + logp  # [beam, V]
            flat = cand_scores.view(-1)
            top_vals, top_idx = flat.topk(beam)
            beam_idx = top_idx // self.vocab_size
            tok_idx  = top_idx %  self.vocab_size

            new_seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            new_alphas: list[list[torch.Tensor]] = []
            for i, bi in enumerate(beam_idx.tolist()):
                new_alphas.append(alphas_per_beam[bi] + [alpha[bi]])

            still_seqs, still_h, still_c, still_scores, still_alphas = [], [], [], [], []
            for i in range(beam):
                tok = tok_idx[i].item()
                if tok == eos_id:
                    sc = top_vals[i].item() / ((new_seqs[i].size(0)) ** length_penalty)
                    finished.append((sc, new_seqs[i].tolist(), torch.stack(new_alphas[i])))
                else:
                    still_seqs.append(new_seqs[i])
                    still_h.append(h[beam_idx[i]])
                    still_c.append(c[beam_idx[i]])
                    still_scores.append(top_vals[i])
                    still_alphas.append(new_alphas[i])
            if not still_seqs:
                break
            seqs   = torch.stack(still_seqs, dim=0)
            h      = torch.stack(still_h,    dim=0)
            c      = torch.stack(still_c,    dim=0)
            scores = torch.stack(still_scores, dim=0)
            alphas_per_beam = still_alphas
            if seqs.size(0) < beam:
                pad_n = beam - seqs.size(0)
                seqs   = torch.cat([seqs,   seqs[-1:].expand(pad_n, -1)],   dim=0)
                h      = torch.cat([h,      h[-1:].expand(pad_n, -1)],     dim=0)
                c      = torch.cat([c,      c[-1:].expand(pad_n, -1)],     dim=0)
                scores = torch.cat([scores, torch.full((pad_n,), float("-inf"), device=device)])
                alphas_per_beam.extend([alphas_per_beam[-1]] * pad_n)

        if not finished:
            # take best running beam
            best_i = scores.argmax().item()
            finished.append((scores[best_i].item(),
                             seqs[best_i].tolist(),
                             torch.stack(alphas_per_beam[best_i]) if alphas_per_beam[best_i] else torch.zeros(1, L, device=device)))
        finished.sort(key=lambda x: x[0], reverse=True)
        _, tokens, alpha_seq = finished[0]
        return tokens, alpha_seq
