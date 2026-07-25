"""
model_cpagrn_multihead.py — CPA-GRN v6 experiment: Multi-Head Neighbor Attention

ΝΕΟ πείραμα αρχιτεκτονικής (21/7), ΔΕΝ έχει ξαναδοκιμαστεί. Διαφέρει από το
ήδη-απορριφθέν v5 "query-aware attention" (§Architecture Search): εκεί
προστέθηκε το h_i (embedding του πλοίου-ερωτήματος) ως επιπλέον όρος στην
είσοδο του attn_mlp. Εδώ ΔΕΝ αλλάζει η είσοδος του attn_mlp — αλλάζει the
μηχανισμός προσοχής ώστε να έχει πολλαπλά, παράλληλα "κεφάλια" (heads),
καθένα από τα οποία μπορεί να μάθει να εστιάζει σε διαφορετική πτυχή της
σχέσης πλοίο-γείτονας (π.χ. ένα head σε κίνδυνο CPA, άλλο σε ευθυγράμμιση
πορείας), όπως στο multi-head attention του Transformer.

Το CPAFeatures, το GRU encoder, ο αποκωδικοποιητής, και το top-k
φιλτράρισμα με βάση απόσταση παραμένουν ΑΚΡΙΒΩΣ όπως στο model_cpagrn.py
(v4) — αλλάζει ΜΟΝΟ το NeighborAggregation module.

Αυτό το αρχείο είναι πλήρως ανεξάρτητο από το model_cpagrn.py — δεν το
αγγίζει, δεν αντικαθιστά τίποτα. Χρησιμοποιείται μόνο από το
train_cpagrn_v5_multihead.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation — ΑΚΡΙΒΩΣ ίδιο με model_cpagrn.py, χωρίς αλλαγή
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        hdg: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i
        v = vel_j - vel_i

        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return torch.stack([
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head Neighbor Aggregation (ΝΕΟ — δεν υπάρχει στο v4)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadNeighborAggregation(nn.Module):
    """
    Ίδιο top-k φιλτράρισμα βάσει απόστασης με το v4 NeighborAggregation, αλλά
    με num_heads παράλληλα attention heads αντί για ένα.

    Σχεδιαστική επιλογή: το attn_mlp παράγει num_heads score ανά ζεύγος
    (αντί για 1), and κάθε head υπολογίζει το δικό του softmax πάνω στους
    γείτονες. Τα μηνύματα (msg_proj) διαχωρίζονται σε num_heads κομμάτια
    d_model/num_heads διαστάσεων το καθένα — τυπικό multi-head pattern.
    Η μάσκα top-k (βάσει απόστασης) είναι ΚΟΙΝΗ σε όλα τα heads, αφού
    αφορά τη φυσική εγγύτητα, όχι κάτι μαθημένο.
    """

    def __init__(self, d_model: int, edge_dim: int = 7, top_k: int = 10, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model πρέπει να διαιρείται ακριβώς με το num_heads"
        self.top_k     = top_k
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.d_model   = d_model

        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_heads),   # ΔΙΑΦΟΡΑ από v4: num_heads score, όχι 1
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,         # [B, N, d_model]
        edges: torch.Tensor,         # [B, N, N, 7]
        mask:  torch.Tensor | None,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = h.shape
        H, Dh   = self.num_heads, self.head_dim

        h_j    = h.unsqueeze(1).expand(B, N, N, D)
        scores = self.attn_mlp(torch.cat([h_j, edges], dim=-1))   # [B, N, N, H]

        if mask is not None:
            mask_j = mask.unsqueeze(1).unsqueeze(-1).expand(B, N, N, H)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        dist = edges[..., 2]  # [B, N, N]
        if mask is not None:
            dist_masked = dist.masked_fill(~mask.unsqueeze(1).expand(B, N, N), float('inf'))
        else:
            dist_masked = dist

        if self.top_k < N:
            k = min(self.top_k, N)
            kth, _ = dist_masked.topk(k, dim=-1, largest=False)
            threshold = kth[..., -1].unsqueeze(-1).unsqueeze(-1)   # [B, N, 1, 1]
            dist_expanded = dist_masked.unsqueeze(-1).expand(B, N, N, H)
            scores = scores.masked_fill(dist_expanded > threshold, float('-inf'))

        weights = F.softmax(scores, dim=2)          # softmax πάνω στους γείτονες (dim=N), ανά head
        weights = torch.nan_to_num(weights, nan=0.0)  # [B, N, N, H]

        msgs = self.msg_proj(h).reshape(B, N, H, Dh)          # [B, N, H, Dh]
        agg  = torch.einsum('bijh,bjhd->bihd', weights, msgs)  # [B, N, H, Dh]
        agg  = agg.reshape(B, N, D)                             # concat heads → [B, N, d_model]

        return self.norm(self.out_proj(agg))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model — ίδια δομή με v4, μόνο το NeighborAggregation άλλαξε
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """
    CPA-Aware Graph Recurrent Network — Πείραμα v6: Multi-Head Attention.

    Ίδια δομή με το v4 (per-step CPA aggregation στο encoder + final spatial
    refinement), μόνο που το NeighborAggregation module αντικαταστάθηκε με
    MultiHeadNeighborAggregation. Ο αριθμός παραμέτρων θα είναι ελαφρώς
    μεγαλύτερος από το v4, λόγω του attn_mlp που τώρα παράγει num_heads
    εξόδους αντί για 1 (αμελητέα διαφορά — λίγες εκατοντάδες παραπάνω
    παράμετροι, όχι διπλασιασμός).
    """

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 10,
        num_heads:    int   = 4,
    ):
        super().__init__()
        self.d_model   = d_model
        self.pred_len  = pred_len
        self.top_k     = top_k
        self.num_heads = num_heads

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cpa_features = CPAFeatures()
        self.neighbor_agg = MultiHeadNeighborAggregation(d_model, edge_dim=7, top_k=top_k, num_heads=num_heads)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.final_spatial = MultiHeadNeighborAggregation(d_model, edge_dim=7, top_k=top_k, num_heads=num_heads)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len * 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs:   torch.Tensor,
        mask:  torch.Tensor | None = None,
        stats: dict | None         = None,   # unused, kept for API compatibility
    ) -> torch.Tensor:
        B, N, T, _ = obs.shape

        x = self.embed(obs)

        fused_steps = []
        for t in range(T):
            pos_t = obs[:, :, t, :2]
            hdg_t = obs[:, :, t, 3]
            vel_t = obs[:, :, t, :2] - obs[:, :, t-1, :2] if t > 0 \
                    else torch.zeros_like(pos_t)

            edges_t = self.cpa_features(pos_t, vel_t, hdg_t)
            x_t     = x[:, :, t, :]
            nbr_t   = self.neighbor_agg(x_t, edges_t, mask)
            fused_steps.append(x_t + nbr_t)

        fused_seq = torch.stack(fused_steps, dim=2)

        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        pos_last   = obs[:, :, -1, :2]
        vel_last   = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                     else torch.zeros_like(pos_last)
        hdg_last   = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
        h = h + self.final_spatial(h, edges_last, mask)

        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss — ίδιο με model_cpagrn.py
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(
    pred_disp:   torch.Tensor,
    target_disp: torch.Tensor,
    mask:        torch.Tensor,
) -> torch.Tensor:
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()
