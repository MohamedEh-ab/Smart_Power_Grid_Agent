"""
smart_grid_gatpo_lib.py
=======================
Production-ready, self-contained library for the GATPO architecture:
    Graph Attention Network (GAT) + Proximal Policy Optimization (PPO)
    for real-time Optimal Power Flow (OPF) dispatch on power grids.

Module layout
-------------
    YBusBuilder          – builds the normalised (G, B) admittance tensor
                           from either a Grid2Op observation or a pandapower net.

    BusClusterer         – K-Means clustering on |Y_bus| rows; produces
                           soft spatial masks that bias actor exploration
                           toward electrically affected zones.

    PositionalEncoding   – fixed sinusoidal PE injected into bus features.

    AdmittanceGATLayer   – dual-channel (G, B) multi-head attention layer.

    ActorCritic          – full GAT backbone with three actor heads (ΔP, V, C)
                           and a critic; supports spatial bias and gating.

    RolloutBuffer        – on-policy trajectory store.

    RunningMeanStd       – Welford online normaliser for critic targets.

    GATPO(BaseEstimator) – sklearn-compatible PPO trainer/predictor.
                           .fit(env)     – curriculum training
                           .predict(obs) – greedy inference → np.ndarray
                           model(obs)    – __call__ alias (PyTorch-style)

    Grid2OpAdapter       – thin wrapper around any Grid2Op environment that
                           exposes all attributes GATPO needs, including
                           automatic Y-bus and cluster-mask computation.

    PandapowerAdapter    – thin wrapper around a pandapower network so you
                           can run the exact usage pattern:

                               net     = pp.networks.case118()
                               adapter = PandapowerAdapter(net)
                               model   = GATPO(n_bus=adapter.n_bus,
                                               n_gen=adapter.n_gen)
                               model.load("model.pth")
                               pp.runpp(net)
                               for t in range(total_steps):
                                   obs    = adapter.get_obs(net)
                                   action = model(obs)           # __call__
                                   adapter.apply_action(net, action)
                                   pp.runpp(net)
                                   adapter.check_stability(net)

Author  : refactored / extended from Smart_Grid_Agent_v2.ipynb
         (original research: Mohamed Ehab Ahmed Mohamed Bishla, SCU)
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


# ============================================================================
# 1.  Y-BUS BUILDER
#     Converts raw power-flow data into a normalised 2-channel admittance
#     tensor [1, 2, N, N] — the edge input for every GAT layer.
# ============================================================================

class YBusBuilder:
    """
    Computes the normalised (G, B) bus-admittance matrix from power-flow data.

    Supports two back-ends:
        * Grid2Op   – pass a Grid2Op observation object.
        * pandapower – pass a pandapower net after pp.runpp().

    The resulting tensor has shape  [1, 2, n_bus, n_bus]:
        channel 0  →  normalised conductance  G̃  (real-power coupling)
        channel 1  →  normalised susceptance  B̃  (reactive-power / elec. distance)

    Both channels are standardised to zero-mean / unit-std, then clipped to ±3.
    Tripped lines are zeroed out.

    Parameters
    ----------
    n_bus : int
        Number of buses / substations.
    clip : float
        Symmetric clip applied after standardisation.  Default 3.0.
    """

    def __init__(self, n_bus: int, clip: float = 3.0) -> None:
        self.n_bus = n_bus
        self.clip  = clip

    # ------------------------------------------------------------------
    def from_grid2op(
        self,
        obs,
        line_or_sub: np.ndarray,
        line_ex_sub: np.ndarray,
        sub_v_nom:   np.ndarray,
    ) -> torch.Tensor:
        """
        Build Y-bus tensor from a Grid2Op observation.

        Parameters
        ----------
        obs         : Grid2Op observation (post env.step / env.reset)
        line_or_sub : int array [n_line]  – origin-end substation per line
        line_ex_sub : int array [n_line]  – extremity-end substation per line
        sub_v_nom   : float array [n_bus] – nominal kV per substation

        Returns
        -------
        Tensor  [1, 2, n_bus, n_bus]
        """
        n     = self.n_bus
        y_bus = np.zeros((n, n), dtype=np.complex64)

        theta_or = np.radians(obs.theta_or)
        theta_ex = np.radians(obs.theta_ex)
        v_or_pu  = obs.v_or / (sub_v_nom[line_or_sub] + 1e-8)
        v_ex_pu  = obs.v_ex / (sub_v_nom[line_ex_sub] + 1e-8)

        v_or_c = v_or_pu * (np.cos(theta_or) + 1j * np.sin(theta_or))
        v_ex_c = v_ex_pu * (np.cos(theta_ex) + 1j * np.sin(theta_ex))
        s_or_c = obs.p_or + 1j * obs.q_or
        I_vec  = np.conj(s_or_c / (v_or_c + 1e-8))
        dV     = v_or_c - v_ex_c
        dV     = np.where(np.abs(dV) < 1e-6, 1e-6 + 0j, dV)
        y_vec  = np.where(np.abs(dV) < 1e-6,
                          complex(100.0, -100.0), I_vec / dV)

        np.add.at(y_bus, (line_or_sub, line_ex_sub), -y_vec)
        np.add.at(y_bus, (line_ex_sub, line_or_sub), -y_vec)
        np.add.at(y_bus, (line_or_sub, line_or_sub),  y_vec)
        np.add.at(y_bus, (line_ex_sub, line_ex_sub),  y_vec)

        tripped = ~np.asarray(obs.line_status, dtype=bool)
        if tripped.any():
            y_bus[line_or_sub[tripped], line_ex_sub[tripped]] = 0
            y_bus[line_ex_sub[tripped], line_or_sub[tripped]] = 0

        return self._normalise(y_bus)

    # ------------------------------------------------------------------
    def from_pandapower(self, net) -> torch.Tensor:
        """
        Build Y-bus tensor from a pandapower network (after pp.runpp()).

        Preferred path: uses net._ppc['internal']['Ybus'] when available
        (exact, sparse, computed by pandapower's own power-flow solver).
        Falls back to a line-parameter reconstruction otherwise.

        Parameters
        ----------
        net : pandapower network object

        Returns
        -------
        Tensor  [1, 2, n_bus, n_bus]
        """
        try:
            ybus_sp = net._ppc["internal"]["Ybus"]
            y_dense = np.asarray(ybus_sp.todense(), dtype=np.complex64)
            # Resize to self.n_bus if ppc uses a larger internal bus numbering
            if y_dense.shape[0] > self.n_bus:
                y_dense = y_dense[: self.n_bus, : self.n_bus]
            return self._normalise(y_dense)
        except (KeyError, AttributeError):
            pass

        # Fallback: build from line parameters
        n     = self.n_bus
        y_bus = np.zeros((n, n), dtype=np.complex64)

        for _, row in net.line.iterrows():
            if not row.get("in_service", True):
                continue
            fb = int(row["from_bus"])
            tb = int(row["to_bus"])
            if fb >= n or tb >= n:
                continue
            try:
                vn_kv  = net.bus.at[fb, "vn_kv"]
                zbase  = vn_kv ** 2 / net.sn_mva
                length = row.get("length_km", 1.0)
                r_pu   = row["r_ohm_per_km"] * length / zbase
                x_pu   = row["x_ohm_per_km"] * length / zbase
                z      = complex(r_pu, x_pu)
                y_ij   = 1.0 / z if abs(z) > 1e-9 else complex(100.0, -100.0)
            except Exception:
                y_ij = complex(1.0, -1.0)

            y_bus[fb, tb] -= y_ij
            y_bus[tb, fb] -= y_ij
            y_bus[fb, fb] += y_ij
            y_bus[tb, tb] += y_ij

        return self._normalise(y_bus)

    # ------------------------------------------------------------------
    def _normalise(self, y_bus: np.ndarray) -> torch.Tensor:
        """Standardise G and B channels; return [1, 2, N, N] tensor."""
        G = np.clip(y_bus.real, -200, 200).astype(np.float32)
        B = np.clip(y_bus.imag, -200, 200).astype(np.float32)

        def _std(m: np.ndarray) -> np.ndarray:
            am = np.abs(m)
            return np.clip(
                (am - am.mean()) / (am.std() + 1e-8), -self.clip, self.clip
            )

        edge = np.stack([_std(G), _std(B)], axis=0)          # [2, N, N]
        return torch.FloatTensor(edge).unsqueeze(0)           # [1, 2, N, N]


# ============================================================================
# 2.  BUS CLUSTERER
#     K-Means on |Y_bus| rows → cluster labels → soft spatial action masks.
# ============================================================================

class BusClusterer:
    """
    Partitions grid buses into electrically coherent zones using K-Means on
    the row-sums of the bus-admittance magnitude matrix.

    After fitting, call  get_gen_mask_for_violations()  at every step to
    obtain a [n_gen] float mask that biases actor exploration toward
    generators inside voltage-violated clusters.

    Parameters
    ----------
    n_clusters          : int   Number of K-Means clusters. Default 8.
    violation_threshold : float |V - 1.0| (pu) to flag a bus. Default 0.05.
    in_zone_bias        : float Mask value for gens in affected clusters. Default 0.8.
    out_zone_bias       : float Mask value for gens in unaffected clusters. Default 0.2.
    random_state        : int   K-Means seed. Default 42.
    """

    def __init__(
        self,
        n_clusters:          int   = 8,
        violation_threshold: float = 0.05,
        in_zone_bias:        float = 0.8,
        out_zone_bias:       float = 0.2,
        random_state:        int   = 42,
    ) -> None:
        self.n_clusters          = n_clusters
        self.violation_threshold = violation_threshold
        self.in_zone_bias        = in_zone_bias
        self.out_zone_bias       = out_zone_bias
        self.random_state        = random_state

        self._kmeans:     Optional[KMeans]    = None
        self.bus_cluster: Optional[np.ndarray] = None   # [n_bus]
        self.gen_cluster: Optional[np.ndarray] = None   # [n_gen]

    # ------------------------------------------------------------------
    def fit(
        self,
        y_bus_tensor: torch.Tensor,
        gen_to_bus:   np.ndarray,
    ) -> "BusClusterer":
        """
        Fit K-Means on the admittance magnitude matrix.

        Parameters
        ----------
        y_bus_tensor : Tensor  [1, 2, N, N]  from YBusBuilder
        gen_to_bus   : int array [n_gen]  – maps generator index → bus index

        Returns
        -------
        self
        """
        G     = y_bus_tensor[0, 0].numpy()
        B     = y_bus_tensor[0, 1].numpy()
        Y_mag = np.sqrt(G ** 2 + B ** 2)
        np.fill_diagonal(Y_mag, 0.0)

        n_clust = min(self.n_clusters, Y_mag.shape[0])
        self._kmeans     = KMeans(n_clusters=n_clust,
                                  random_state=self.random_state, n_init=10)
        self.bus_cluster = self._kmeans.fit_predict(Y_mag)    # [n_bus]
        self.gen_cluster = self.bus_cluster[gen_to_bus]       # [n_gen]
        return self

    # ------------------------------------------------------------------
    def get_gen_mask_for_violations(self, v_pu: np.ndarray) -> np.ndarray:
        """
        Return a soft float mask [n_gen] from current bus voltages.

        Generators in clusters that contain at least one violating bus
        receive  in_zone_bias;  all others receive  out_zone_bias.

        Parameters
        ----------
        v_pu : float array [n_bus]  – per-unit voltage magnitude

        Returns
        -------
        float array [n_gen]
        """
        if self.bus_cluster is None or self.gen_cluster is None:
            raise RuntimeError("Call .fit() before get_gen_mask_for_violations().")

        v_dev = np.abs(np.clip(v_pu, 0.8, 1.2) - 1.0)
        violating = np.where(v_dev > self.violation_threshold)[0]

        if len(violating) == 0:
            return np.ones(len(self.gen_cluster), dtype=np.float32)

        affected = {int(self.bus_cluster[b]) for b in violating}
        return np.array(
            [self.in_zone_bias if int(c) in affected else self.out_zone_bias
             for c in self.gen_cluster],
            dtype=np.float32,
        )

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None


# ============================================================================
# 3.  POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding injected into bus-node features.

    Parameters
    ----------
    d_model : int  – embedding dimension.
    max_len : int  – maximum number of buses supported. Default 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        if n > self.pe.size(0):
            raise ValueError(f"Nodes {n} > max_len {self.pe.size(0)}")
        return x + self.pe[:n].unsqueeze(0)


# ============================================================================
# 4.  ADMITTANCE-CONDITIONED GAT LAYER
# ============================================================================

class AdmittanceGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer conditioned on the complex Y-bus.

    An edge-bias MLP on the (G, B) channel pair lets different heads
    specialise in active-power (G) vs reactive-power (B) flow.

    Parameters
    ----------
    in_features      : int
    out_features     : int   Must be divisible by num_heads.
    num_heads        : int   Default 4.
    edge_dim         : int   Edge channels (2 for G and B). Default 2.
    edge_mlp_hidden  : int   Hidden units per head in edge MLP. Default 32.
    dropout          : float Attention dropout. Default 0.1.
    """

    def __init__(
        self,
        in_features:     int,
        out_features:    int,
        num_heads:       int   = 4,
        edge_dim:        int   = 2,
        edge_mlp_hidden: int   = 32,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        assert out_features % num_heads == 0, \
            "out_features must be divisible by num_heads"
        self.heads    = num_heads
        self.head_dim = out_features // num_heads
        self.dropout  = dropout

        self.W     = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, edge_mlp_hidden * num_heads),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden * num_heads, num_heads),
        )
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x:                torch.Tensor,
        adj:              Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x   : [B, N, in_features]
        adj : [B, 2, N, N]  optional (G and B channels)

        Returns
        -------
        out  : [B, N, out_features]
        attn : [B, H, N, N]  only when return_attention=True
        """
        B, N, _ = x.size()
        h = self.W(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        s_src  = (h * self.a_src).sum(-1, keepdim=True)
        s_dst  = (h * self.a_dst).sum(-1, keepdim=True)
        scores = s_src + s_dst.transpose(-2, -1)

        if adj is not None:
            edge_exists = adj.abs().sum(dim=1, keepdim=True) > 1e-8
            e_bias = self.edge_proj(
                adj.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            scores = (scores + e_bias).masked_fill(
                ~edge_exists.expand(B, self.heads, N, N), -1e9
            )

        scores = self.leaky(scores) / math.sqrt(self.head_dim)
        attn   = F.softmax(scores, dim=-1)
        attn   = F.dropout(attn, p=self.dropout, training=self.training)
        out    = torch.matmul(attn, h).transpose(1, 2).contiguous().view(B, N, -1)
        return (out, attn) if return_attention else out


# ============================================================================
# 5.  ACTOR-CRITIC
# ============================================================================

class ActorCritic(nn.Module):
    """
    Dual-GAT Actor-Critic network for power-grid OPF dispatch.

    Input stack
    -----------
    bus_feat  [B, n_bus, 6]  → Linear(6→32) + PositionalEncoding
    gen_feat  [B, 4*n_gen]   → MLP(4n_gen→128→64), broadcast to all buses
    combined  [B, n_bus, 96] → GAT1(96→128) → GAT2(128→128) with residuals
    pool      [B, 512]       → Linear(512→256) + ReLU  →  fusion h

    Actor heads (all gated / spatially biased)
    ------------------------------------------
    P-head  : Linear(256 → n_gen)        → Sigmoid → Normal(μ_P, σ_P)
    V-head  : Linear(256+n_gen → n_gen)  → Sigmoid → gate-blend → Normal(μ_V, σ_V)
    C-head  : Linear(256+n_gen → n_gen)  → Sigmoid → gate-blend → Normal(μ_C, σ_C)

    Critic
    ------
    Linear(256 → 1)

    Parameters
    ----------
    n_bus, n_gen, input_dim, hidden_dim, num_heads, log_std_min, log_std_max
    """

    _PROJ_DIM = 32

    def __init__(
        self,
        n_bus:       int   = 118,
        n_gen:       int   = 62,
        input_dim:   int   = 6,
        hidden_dim:  int   = 128,
        num_heads:   int   = 4,
        log_std_min: float = 0.01,
        log_std_max: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_bus       = n_bus
        self.n_gen       = n_gen
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        d = self._PROJ_DIM
        self.input_proj  = nn.Linear(input_dim, d)
        self.pos_encoder = PositionalEncoding(d, max_len=n_bus)
        self.gen_encoder = nn.Sequential(
            nn.Linear(4 * n_gen, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        cat_dim       = d + 64
        self.res_proj = nn.Linear(cat_dim, hidden_dim)
        self.gat1     = AdmittanceGATLayer(cat_dim,    hidden_dim, num_heads)
        self.gat2     = AdmittanceGATLayer(hidden_dim, hidden_dim, num_heads)

        self.fusion     = nn.Linear(4 * hidden_dim, 256)
        self.gate_layer = nn.Linear(256, 1)

        self.mu_P      = nn.Linear(256,           n_gen)
        self.mu_V      = nn.Linear(256 + n_gen,   n_gen)
        self.mu_C      = nn.Linear(256 + n_gen,   n_gen)
        self.log_std_P = nn.Parameter(torch.full([n_gen], -1.0))
        self.log_std_V = nn.Parameter(torch.full([n_gen], -1.5))
        self.log_std_C = nn.Parameter(torch.full([n_gen], -1.0))
        self.value     = nn.Linear(256, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x:                torch.Tensor,
        adj:              Optional[torch.Tensor] = None,
        gen_feat:         Optional[torch.Tensor] = None,
        cluster_mask:     Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple:
        """
        Parameters
        ----------
        x            : [B, n_bus, input_dim]
        adj          : [B, 2, N, N]       – Y-bus admittance tensor
        gen_feat     : [B, 4*n_gen]       – generator state features
        cluster_mask : [B, n_gen]         – spatial soft-bias mask in [0,1]
        return_attention : bool

        Returns
        -------
        dist_P, dist_emg, val  (+ all_attns list when return_attention=True)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        std_x = x.std(dim=1, keepdim=True).clamp(min=1e-3)
        x     = (x - x.mean(dim=1, keepdim=True)) / std_x
        x     = F.relu(self.input_proj(x))
        x     = self.pos_encoder(x)

        if gen_feat is not None:
            g = self.gen_encoder(gen_feat)
            x = torch.cat([x, g.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)

        x_proj = self.res_proj(x)

        if return_attention:
            h1_raw, a1 = self.gat1(x, adj, return_attention=True)
        else:
            h1_raw = self.gat1(x, adj)
        h1 = F.relu(F.layer_norm(h1_raw, h1_raw.shape[-1:]) + x_proj)

        if return_attention:
            h2_raw, a2 = self.gat2(h1, adj, return_attention=True)
        else:
            h2_raw = self.gat2(h1, adj)
        h2 = F.relu(F.layer_norm(h2_raw, h2_raw.shape[-1:]) + h1)

        h_cat = torch.cat([h1, h2], dim=-1)
        h = F.relu(self.fusion(
            torch.cat([h_cat.mean(1), h_cat.max(1).values], dim=-1)
        ))
        gate = torch.sigmoid(self.gate_layer(h.detach()))
        bias = 2.0 * (cluster_mask - 0.5) if cluster_mask is not None else 0.0

        mu_P  = torch.sigmoid(self.mu_P(h) + bias)
        std_P = self.log_std_P.exp().clamp(self.log_std_min, self.log_std_max)
        dist_P = Normal(mu_P, std_P.expand_as(mu_P))

        emg  = torch.cat([h, mu_P.detach()], dim=-1)
        mu_V = gate * torch.sigmoid(self.mu_V(emg) + bias) + (1 - gate) * 0.5
        mu_C = gate * torch.sigmoid(self.mu_C(emg) + bias) + (1 - gate) * 0.5
        std_V = self.log_std_V.exp().clamp(self.log_std_min, self.log_std_max)
        std_C = self.log_std_C.exp().clamp(self.log_std_min, self.log_std_max)
        dist_emg = Normal(
            torch.cat([mu_V, mu_C], dim=-1),
            torch.cat([std_V.expand_as(mu_V), std_C.expand_as(mu_C)], dim=-1),
        )
        val = self.value(h)

        if return_attention:
            return dist_P, dist_emg, val, [a1, a2]
        return dist_P, dist_emg, val

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 6.  ROLLOUT BUFFER  &  RUNNING MEAN/STD
# ============================================================================

class RolloutBuffer:
    """On-policy trajectory buffer. Clears itself after each PPO update."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states:       List = []
        self.gen_feats:    List = []
        self.actions:      List = []
        self.log_probs:    List = []
        self.rewards:      List = []
        self.masks:        List = []
        self.values:       List = []
        self.adjs:         List = []
        self.infos:        List = []
        self.cluster_masks:List = []

    def __len__(self) -> int:
        return len(self.rewards)

    def push(
        self, state, gen_feat, action, log_prob,
        reward, mask, value, adj, info, cluster_mask=None,
    ) -> None:
        self.states.append(state)
        self.gen_feats.append(gen_feat)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.values.append(value)
        self.adjs.append(adj)
        self.infos.append(info)
        self.cluster_masks.append(
            cluster_mask if cluster_mask is not None else torch.zeros(1, 1)
        )


class RunningMeanStd:
    """Welford online mean/variance tracker for critic return normalisation."""

    def __init__(self, epsilon: float = 1e-4, clip: float = 3.0) -> None:
        self.mean:  float = 0.0
        self.var:   float = 1.0
        self.count: float = epsilon
        self.clip         = clip

    def update(self, x: np.ndarray) -> None:
        x  = np.asarray(x, dtype=np.float64)
        bn, bm, bv = len(x), x.mean(), x.var()
        delta = bm - self.mean
        tot   = self.count + bn
        self.mean += delta * bn / tot
        M2 = (self.var * self.count + bv * bn
              + delta ** 2 * self.count * bn / tot)
        self.var   = M2 / tot
        self.count = tot

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self.clip, self.clip,
        )


# ============================================================================
# 7.  GRID2OP ADAPTER
# ============================================================================

class Grid2OpAdapter:
    """
    Thin wrapper around a Grid2Op environment.

    Automatically computes at every step:
        * Per-bus voltage features (6-dim)
        * Generator state features (4 × n_gen)
        * Y-bus admittance tensor  [1, 2, n_bus, n_bus]  → ._y_bus_adj
        * Spatial cluster mask     [n_gen]

    The adapter exposes all attributes that GATPO.fit() and predict() need,
    so training and inference require no additional glue code.

    Parameters
    ----------
    g2op_env     : Grid2Op environment object
    n_clusters   : int   K-Means clusters for BusClusterer. Default 8.
    noise_std    : float Gaussian noise on bus features. Default 0.02.
    failure_prob : float Per-step random generator trip probability. Default 0.05.
    suez_shock_pct : float Capacity reduction for fossil generators. Default 0.0.

    Usage (training)
    ----------------
        adapter = Grid2OpAdapter(grid2op.make("l2rpn_wcci_2022"))
        model   = GATPO(n_bus=adapter.n_bus, n_gen=adapter.n_gen)
        model.fit(adapter)

    Usage (inference)
    -----------------
        obs, _ = adapter.reset()
        while not done:
            action             = model(obs)
            obs, rew, done, *_ = adapter.step(action)
    """

    def __init__(
        self,
        g2op_env,
        n_clusters:    int   = 8,
        noise_std:     float = 0.02,
        failure_prob:  float = 0.05,
        suez_shock_pct:float = 0.0,
    ) -> None:
        self.g2op_env       = g2op_env
        self.noise_std      = noise_std
        self.failure_prob   = failure_prob
        self.suez_shock_pct = suez_shock_pct

        self.n_bus = g2op_env.n_sub
        self.n_gen = g2op_env.n_gen

        # Topology look-up tables (fixed for the lifetime of the adapter)
        self._line_or_sub = np.array(g2op_env.line_or_to_subid)
        self._line_ex_sub = np.array(g2op_env.line_ex_to_subid)
        self._gen_sub     = np.array(g2op_env.gen_to_subid)
        self._load_sub    = np.array(g2op_env.load_to_subid)

        # Nominal bus voltages
        self._sub_v_nom = np.ones(self.n_bus, dtype=np.float32)
        try:
            line_v = np.array(g2op_env.backend.lines_or_pu_to_kv, dtype=np.float32)
            for i in range(g2op_env.n_line):
                self._sub_v_nom[self._line_or_sub[i]] = line_v[i]
                self._sub_v_nom[self._line_ex_sub[i]] = line_v[i]
        except Exception:
            pass

        # Incident lines per bus
        self._bus_lines: List[List[int]] = [[] for _ in range(self.n_bus)]
        for lid in range(g2op_env.n_line):
            self._bus_lines[self._line_or_sub[lid]].append(lid)
            self._bus_lines[self._line_ex_sub[lid]].append(lid)
        self._max_degree = max((len(b) for b in self._bus_lines), default=1)

        # Renewable / fossil masks
        renewable_kw = {"solar", "wind", "hydro"}
        fossil_kw    = {"gas", "oil"}
        try:
            sources = g2op_env.gen_type
        except Exception:
            sources = ["unknown"] * self.n_gen
        self._renewable_mask = np.array(
            [any(k in str(s).lower() for k in renewable_kw) for s in sources],
            dtype=bool,
        )
        self._fossil_mask = np.array(
            [any(k in str(s).lower() for k in fossil_kw) for s in sources],
            dtype=bool,
        )
        self._load_bus_flag = np.isin(
            np.arange(self.n_bus), self._load_sub
        ).astype(np.float32)

        # Helpers
        self._ybus_builder  = YBusBuilder(self.n_bus)
        self._clusterer     = BusClusterer(n_clusters=n_clusters)
        self._last_obs      = None
        self._y_bus_adj: Optional[torch.Tensor] = None

        # Training-compatible public attributes (written by GATPO.fit)
        try:
            self.max_steps = g2op_env.chronics_handler.max_episode_duration()
        except Exception:
            self.max_steps = 864
        self.n_steps         = 0
        self.total_train_steps = 0
        self.last_equity     = 100.0
        self.last_v_viol     = 0
        self.last_max_v_dev  = 0.0
        self.last_max_loading= 0.0
        self.last_power_loss = 0.0
        self.last_n_violations = 0
        self.last_g_V = self.last_g_T = self.last_g_G = self.last_g_C = 0.0
        self.last_lambda_V = self.last_lambda_T = 0.0
        self.last_lambda_G = self.last_lambda_C = 0.0

        # Lagrange multipliers — written by GATPO.fit before each episode
        self.lambda_V = self.lambda_T = self.lambda_G = self.lambda_C = 0.0

        # Bootstrap the clusterer from an initial observation
        self._init_clusterer()

    # ------------------------------------------------------------------
    def _init_clusterer(self) -> None:
        obs = self.g2op_env.reset()
        adj = self._ybus_builder.from_grid2op(
            obs, self._line_or_sub, self._line_ex_sub, self._sub_v_nom
        )
        self._clusterer.fit(adj, self._gen_sub)

    # ------------------------------------------------------------------
    def _get_bus_voltages(self, obs) -> np.ndarray:
        v_sum = np.zeros(self.n_bus, dtype=np.float32)
        v_cnt = np.zeros(self.n_bus, dtype=np.float32)
        for i, s in enumerate(self._line_or_sub):
            v_sum[s] += obs.v_or[i]; v_cnt[s] += 1
        for i, s in enumerate(self._line_ex_sub):
            v_sum[s] += obs.v_ex[i]; v_cnt[s] += 1
        return (v_sum / np.maximum(v_cnt, 1.0)) / (self._sub_v_nom + 1e-8)

    # ------------------------------------------------------------------
    def _obs_to_features(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bus_feat [n_bus, 6], gen_feat [4*n_gen])."""
        # Active-power injection per bus
        p = np.zeros(self.n_bus, dtype=np.float32)
        for g in range(self.g2op_env.n_gen):
            p[self._gen_sub[g]] += obs.gen_p[g]
        for l in range(self.g2op_env.n_load):
            p[self._load_sub[l]] -= obs.load_p[l]

        v_pu  = self._get_bus_voltages(obs)
        theta = np.zeros(self.n_bus, dtype=np.float32)
        for lid in range(self.g2op_env.n_line):
            theta[self._line_or_sub[lid]] = obs.theta_or[lid]
            theta[self._line_ex_sub[lid]] = obs.theta_ex[lid]

        line_status  = np.asarray(obs.line_status, dtype=np.float32)
        conn_ratio   = np.zeros(self.n_bus, dtype=np.float32)
        tripped_norm = np.zeros(self.n_bus, dtype=np.float32)
        for b in range(self.n_bus):
            inc = self._bus_lines[b]
            if inc:
                s = line_status[inc]
                conn_ratio[b]   = s.mean()
                tripped_norm[b] = (1.0 - s).sum() / self._max_degree

        elec = np.stack(
            [p / 400.0, v_pu - 1.0, theta / 180.0], axis=-1
        ).astype(np.float32)
        elec += np.random.normal(0, self.noise_std, elec.shape).astype(np.float32)

        bus_feat = np.concatenate([
            elec,
            conn_ratio.reshape(-1, 1),
            tripped_norm.reshape(-1, 1),
            self._load_bus_flag.reshape(-1, 1),
        ], axis=-1)  # [n_bus, 6]

        pmax     = self.g2op_env.gen_pmax + 1e-6
        gen_feat = np.concatenate([
            obs.gen_p / pmax,
            obs.prod_v / 1.04,
            self._renewable_mask.astype(np.float32),
            1.0 - obs.gen_p / pmax,
        ])  # [4 * n_gen]

        return bus_feat, gen_feat

    # ------------------------------------------------------------------
    def _compute_adj(self, obs) -> torch.Tensor:
        return self._ybus_builder.from_grid2op(
            obs, self._line_or_sub, self._line_ex_sub, self._sub_v_nom
        )

    # ------------------------------------------------------------------
    def get_cluster_mask(self) -> np.ndarray:
        """Return soft [n_gen] cluster mask based on current voltages."""
        if self._last_obs is None:
            return np.ones(self.n_gen, dtype=np.float32)
        v_pu = self._get_bus_voltages(self._last_obs)
        return self._clusterer.get_gen_mask_for_violations(v_pu)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs = self.g2op_env.reset()
        self._last_obs  = obs
        self.n_steps    = 0
        self._y_bus_adj = self._compute_adj(obs)
        return self._obs_to_features(obs), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        Apply action and advance one simulation step.

        action : np.ndarray  [3 * n_gen]  values in [0, 1]
                 layout  →  [ΔP setpoints | V setpoints | curtailment fractions]
        """
        self.n_steps         += 1
        self.total_train_steps += 1
        action = np.asarray(action, dtype=np.float32).flatten()
        ng = self.n_gen

        a_P = np.clip(action[:ng],        0.0, 1.0)
        a_V = np.clip(action[ng:2 * ng],  0.0, 1.0)
        a_C = np.clip(action[2 * ng:],    0.0, 1.0)

        if self.suez_shock_pct > 0:
            a_P[self._fossil_mask] *= (1.0 - self.suez_shock_pct)
        if np.random.rand() < self.failure_prob:
            a_P[np.random.randint(ng)] = 0.0

        pmax     = np.array(self.g2op_env.gen_pmax, dtype=np.float32)
        pmin     = np.array(self.g2op_env.gen_pmin, dtype=np.float32)
        V_LO, V_HI = 0.90, 1.10
        target_p = np.clip(a_P * pmax, pmin, pmax)
        target_v = np.clip(V_LO + a_V * (V_HI - V_LO), V_LO, V_HI)

        cur_disp = self._last_obs.actual_dispatch
        redisp   = {i: float(target_p[i] - cur_disp[i])
                    for i in range(ng) if not self._renewable_mask[i]}
        curtail  = {i: float(a_C[i]) for i in range(ng) if self._renewable_mask[i]}

        g2op_act = self.g2op_env.action_space({
            "redispatch": redisp,
            "prod_v":     {i: float(v) for i, v in enumerate(target_v)},
            "curtail":    curtail,
        })
        obs, _, done, info = self.g2op_env.step(g2op_act)
        if not isinstance(info, dict):
            info = {}

        v_pu  = self._get_bus_voltages(obs)
        v_dev = np.abs(v_pu - 1.0)
        g_V   = float(np.mean(np.maximum(0.0, v_dev - 0.06)))
        g_T   = float(np.clip(np.mean(np.maximum(0.0, obs.rho - 1.0)), 0, 2))
        g_G   = float(np.mean(
            np.abs(np.clip(target_p, pmin, pmax) - target_p) / (pmax + 1e-6)
        ))
        g_C   = 0.0
        info.update({"g_V": g_V, "g_T": g_T, "g_G": g_G, "g_C": g_C})

        jain_n = float(np.clip(
            obs.load_p.sum() / (obs.gen_p.sum() + 1e-6), 0.0, 1.0
        ))
        reward = (2.0 + 2.0 * jain_n
                  - self.lambda_V * g_V
                  - self.lambda_T * g_T
                  - self.lambda_G * g_G)
        if done and self.n_steps < self.max_steps:
            reward -= 2.0 * (self.max_steps - self.n_steps) / self.max_steps
        if done and self.n_steps >= self.max_steps:
            reward += 3.0

        self.last_equity        = float(jain_n * 100.0)
        self.last_v_viol        = int(np.sum(v_dev > 0.06))
        self.last_max_v_dev     = float(v_dev.max())
        self.last_max_loading   = float(obs.rho.max())
        self.last_g_V, self.last_g_T = g_V, g_T
        self.last_g_G, self.last_g_C = g_G, g_C
        self.last_lambda_V = self.lambda_V
        self.last_lambda_T = self.lambda_T
        self.last_lambda_G = self.lambda_G
        self.last_lambda_C = self.lambda_C
        self.last_power_loss    = float(max(0.0, obs.gen_p.sum() - obs.load_p.sum()))
        self.last_n_violations  = self.last_v_viol

        self._last_obs  = obs
        self._y_bus_adj = self._compute_adj(obs)
        return self._obs_to_features(obs), reward, done, False, info


# ============================================================================
# 8.  PANDAPOWER ADAPTER
# ============================================================================

class PandapowerAdapter:
    """
    Thin interface between GATPO and a pandapower network.

    Automatically:
        * Builds the Y-bus tensor from net._ppc (exact) after pp.runpp()
        * Extracts 6-dim bus features and 4×n_gen generator features
        * Fits a BusClusterer from the initial Y-bus
        * Applies model actions back to net.gen / net.ext_grid setpoints
        * Runs a stability check (voltage bands + line loading)

    Parameters
    ----------
    net        : pandapower network (any case — case14, case118, …)
    n_clusters : int   Spatial clustering K. Default 8.
    noise_std  : float Gaussian noise added to bus features. Default 0.0.

    Usage
    -----
        import pandapower as pp
        from smart_grid_gatpo_lib import GATPO, PandapowerAdapter

        net     = pp.networks.case118()
        pp.runpp(net)
        adapter = PandapowerAdapter(net)

        model   = GATPO(n_bus=adapter.n_bus, n_gen=adapter.n_gen)
        model.load("model.pth")          # or model.fit(some_g2op_adapter)

        for t in range(total_steps):
            obs    = adapter.get_obs(net)      # returns (bus_feat, gen_feat)
            action = model(obs)                # __call__ → np.ndarray [3*n_gen]
            adapter.apply_action(net, action)
            pp.runpp(net)
            adapter.check_stability(net)
    """

    def __init__(
        self,
        net,
        n_clusters: int   = 8,
        noise_std:  float = 0.0,
    ) -> None:
        try:
            import pandapower as pp
            pp.runpp(net, verbose=False)
        except ImportError as e:
            raise ImportError("pandapower required: pip install pandapower") from e
        except Exception:
            pass  # net may already have results

        self.noise_std  = noise_std
        self.n_bus      = len(net.bus)
        # Controllable sources = generators + external grids
        self.n_gen      = len(net.gen) + len(net.ext_grid)

        self._ybus_builder = YBusBuilder(self.n_bus)
        self._clusterer    = BusClusterer(n_clusters=n_clusters)

        # Voltage limits from net (fall back to ±10 %)
        self._v_min = net.bus["min_vm_pu"].fillna(0.90).values.astype(np.float32)
        self._v_max = net.bus["max_vm_pu"].fillna(1.10).values.astype(np.float32)

        # Map each generator slot to a bus index
        self._gen_to_bus = self._get_gen_bus_indices(net)

        # Build initial Y-bus and fit clusterer
        adj = self._ybus_builder.from_pandapower(net)
        self._clusterer.fit(adj, self._gen_to_bus)

    # ------------------------------------------------------------------
    def _get_gen_bus_indices(self, net) -> np.ndarray:
        buses = []
        for _, row in net.gen.iterrows():
            buses.append(int(row["bus"]))
        for _, row in net.ext_grid.iterrows():
            buses.append(int(row["bus"]))
        buses = buses[: self.n_gen]
        while len(buses) < self.n_gen:
            buses.append(buses[-1] if buses else 0)
        return np.array(buses, dtype=int)

    # ------------------------------------------------------------------
    def get_obs(self, net) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (bus_features [n_bus, 6], gen_features [4*n_gen]) from net.

        Call after pp.runpp(net).

        Feature layout per bus
        ----------------------
        [0] p_mw / 400          – normalised active-power injection
        [1] vm_pu − 1.0         – voltage deviation from 1.0 pu
        [2] va_degree / 180     – normalised voltage angle
        [3] connectivity ratio  – fraction of in-service lines at bus
        [4] tripped norm        – fraction of tripped lines at bus
        [5] load_flag           – 1 if load is connected, else 0

        Returns
        -------
        obs : tuple  ready to pass directly to model(obs)
        """
        n = self.n_bus

        # Bus electrical quantities
        vm_pu  = net.res_bus["vm_pu"].fillna(1.0).values.astype(np.float32)
        va_deg = net.res_bus["va_degree"].fillna(0.0).values.astype(np.float32)
        p_mw   = net.res_bus["p_mw"].fillna(0.0).values.astype(np.float32)

        # Connectivity: decrease per out-of-service line
        conn = np.ones(n, dtype=np.float32)
        for _, row in net.line.iterrows():
            if not row.get("in_service", True):
                fb, tb = int(row["from_bus"]), int(row["to_bus"])
                if fb < n:
                    conn[fb] = max(0.0, conn[fb] - 0.1)
                if tb < n:
                    conn[tb] = max(0.0, conn[tb] - 0.1)

        tripped = 1.0 - conn

        load_flag = np.zeros(n, dtype=np.float32)
        for _, row in net.load.iterrows():
            b = int(row["bus"])
            if b < n:
                load_flag[b] = 1.0

        bus_feat = np.stack([
            p_mw  / 400.0,
            vm_pu - 1.0,
            va_deg / 180.0,
            conn,
            tripped,
            load_flag,
        ], axis=-1).astype(np.float32)

        bus_feat += np.random.normal(
            0, self.noise_std, bus_feat.shape
        ).astype(np.float32)

        # Generator features  [4 * n_gen]
        p_norm, v_norm, ren, head = [], [], [], []

        for idx, row in net.gen.iterrows():
            pmax = float(row.get("max_p_mw", 1.0)) + 1e-6
            p    = float(
                net.res_gen.at[idx, "p_mw"]
            ) if idx in net.res_gen.index else 0.0
            p_norm.append(p / pmax)
            v_norm.append(float(row.get("vm_pu", 1.0)) / 1.04)
            ren.append(0.0)
            head.append(max(0.0, 1.0 - p / pmax))

        for idx, row in net.ext_grid.iterrows():
            p = float(
                net.res_ext_grid.at[idx, "p_mw"]
            ) if idx in net.res_ext_grid.index else 0.0
            p_norm.append(p / 9999.0)
            v_norm.append(float(row.get("vm_pu", 1.0)) / 1.04)
            ren.append(0.0)
            head.append(max(0.0, 1.0 - p / 9999.0))

        # Trim / pad to n_gen slots
        def _pad(lst, fill=0.0):
            lst = lst[: self.n_gen]
            lst += [fill] * max(0, self.n_gen - len(lst))
            return lst

        gen_feat = np.array(
            _pad(p_norm) + _pad(v_norm) + _pad(ren) + _pad(head),
            dtype=np.float32,
        )
        return bus_feat, gen_feat

    # ------------------------------------------------------------------
    def get_adj(self, net) -> torch.Tensor:
        """Compute and return the Y-bus admittance tensor [1, 2, n_bus, n_bus]."""
        return self._ybus_builder.from_pandapower(net)

    # ------------------------------------------------------------------
    def get_cluster_mask(self, net) -> np.ndarray:
        """Return a soft [n_gen] cluster mask from current bus voltages."""
        v_pu = net.res_bus["vm_pu"].fillna(1.0).values.astype(np.float32)
        return self._clusterer.get_gen_mask_for_violations(v_pu)

    # ------------------------------------------------------------------
    def apply_action(self, net, action: np.ndarray) -> None:
        """
        Write GATPO action back into the pandapower network setpoints.

        action : np.ndarray  [3 * n_gen]  values in [0, 1]
                 layout  →  [ΔP fractions | V setpoints | curtailment fractions]

        The curtailment slice is stored but not applied for non-renewable gens.
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        ng     = self.n_gen
        a_P    = np.clip(action[:ng],       0.0, 1.0)
        a_V    = np.clip(action[ng:2 * ng], 0.0, 1.0)
        V_LO, V_HI = 0.90, 1.10
        gen_idx = 0

        for idx, row in net.gen.iterrows():
            if gen_idx >= ng:
                break
            pmax = float(row.get("max_p_mw", 1.0))
            pmin = float(row.get("min_p_mw", 0.0))
            net.gen.at[idx, "p_mw"]  = float(np.clip(a_P[gen_idx] * pmax, pmin, pmax))
            net.gen.at[idx, "vm_pu"] = float(V_LO + a_V[gen_idx] * (V_HI - V_LO))
            gen_idx += 1

        for idx, _ in net.ext_grid.iterrows():
            if gen_idx >= ng:
                break
            net.ext_grid.at[idx, "vm_pu"] = float(
                V_LO + a_V[gen_idx] * (V_HI - V_LO)
            )
            gen_idx += 1

    # ------------------------------------------------------------------
    def check_stability(
        self,
        net,
        v_tol:       float = 0.10,
        loading_tol: float = 1.00,
    ) -> Dict:
        """
        Basic post-runpp stability check.

        Parameters
        ----------
        v_tol       : float  Max allowed |vm_pu − 1.0|. Default 0.10 (±10 %).
        loading_tol : float  Max allowed line loading fraction. Default 1.0 (100 %).

        Returns
        -------
        dict with keys:
            stable        bool
            v_violations  list of (bus_idx, vm_pu)
            overloaded    list of (line_idx, loading_pct)
            max_v_dev     float
            max_loading   float
        """
        vm      = net.res_bus["vm_pu"].fillna(1.0).values
        v_viol  = [(int(i), float(v))
                   for i, v in enumerate(vm) if abs(v - 1.0) > v_tol]

        loading = net.res_line["loading_percent"].fillna(0.0).values / 100.0
        ovld    = [(int(i), float(l * 100))
                   for i, l in enumerate(loading) if l > loading_tol]

        stable = len(v_viol) == 0 and len(ovld) == 0
        result = {
            "stable":       stable,
            "v_violations": v_viol,
            "overloaded":   ovld,
            "max_v_dev":    float(np.max(np.abs(vm - 1.0))),
            "max_loading":  float(np.max(loading)),
        }
        if not stable:
            print(
                f"  [STABILITY]  V-viol={len(v_viol)}  "
                f"overloads={len(ovld)}  "
                f"max_v_dev={result['max_v_dev']:.4f}  "
                f"max_load={result['max_loading']:.4f}"
            )
        return result


# ============================================================================
# 9.  GATPO  —  SKLEARN-COMPATIBLE PPO WRAPPER
# ============================================================================

class GATPO(BaseEstimator):
    """
    GATPO: Graph Attention Network + Proximal Policy Optimization.

    sklearn BaseEstimator — all hyperparameters in __init__ for GridSearchCV
    and Pipeline compatibility.

    Calling the model directly (``model(obs)``) runs greedy inference and
    returns a numpy array, matching PyTorch / pandapower usage:

        action = model(obs)                  # same as model.predict(obs)
        action = model(obs, adj=adj, cluster_mask=cm)

    Parameters
    ----------
    n_bus, n_gen, input_dim, hidden_dim, num_heads, log_std_min, log_std_max
        Network architecture (must match the trained checkpoint).
    lr, gamma, gae_lambda, clip_eps, ppo_epochs, batch_size, grad_clip, horizon
        PPO hyperparameters.
    lr_lambda, lambda_max, lambda_decay, rho_v, rho_t, rho_g, rho_c
        Augmented Lagrangian parameters.
    lr_milestones, lr_gamma
        MultiStepLR schedule.
    curriculum
        List of (n_episodes, shock_lo, shock_hi, label) tuples.
    ckpt_every, output_dir, ckpt_dir, device, verbose
        Training meta-parameters.
    """

    _DEFAULT_CURRICULUM: List[Tuple] = [
        (400, 0.00, 0.00, "I   — Baseline"),
        (300, 0.10, 0.30, "II  — Adaptation"),
        (300, 0.40, 0.50, "IIb — Stability Edge"),
        (400, 0.30, 0.60, "III — Resiliency"),
        (100, 0.00, 0.70, "IV  — Generalization"),
    ]

    def __init__(
        self,
        n_bus:        int   = 118,
        n_gen:        int   = 62,
        input_dim:    int   = 6,
        hidden_dim:   int   = 128,
        num_heads:    int   = 4,
        log_std_min:  float = 0.01,
        log_std_max:  float = 0.35,
        lr:           float = 1e-4,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_eps:     float = 0.2,
        ppo_epochs:   int   = 5,
        batch_size:   int   = 64,
        grad_clip:    float = 0.5,
        horizon:      int   = 512,
        lr_lambda:    float = 0.001,
        lambda_max:   float = 50.0,
        lambda_decay: float = 0.995,
        rho_v:        float = 5.0,
        rho_t:        float = 5.0,
        rho_g:        float = 5.0,
        rho_c:        float = 5.0,
        lr_milestones: Optional[List[int]] = None,
        lr_gamma:     float = 0.5,
        curriculum:   Optional[List[Tuple]] = None,
        ckpt_every:   int   = 250,
        output_dir:   str   = "./output",
        ckpt_dir:     str   = "./checkpoints",
        device:       str   = "cpu",
        verbose:      int   = 50,
    ) -> None:
        self.n_bus         = n_bus
        self.n_gen         = n_gen
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.num_heads     = num_heads
        self.log_std_min   = log_std_min
        self.log_std_max   = log_std_max
        self.lr            = lr
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.ppo_epochs    = ppo_epochs
        self.batch_size    = batch_size
        self.grad_clip     = grad_clip
        self.horizon       = horizon
        self.lr_lambda     = lr_lambda
        self.lambda_max    = lambda_max
        self.lambda_decay  = lambda_decay
        self.rho_v         = rho_v
        self.rho_t         = rho_t
        self.rho_g         = rho_g
        self.rho_c         = rho_c
        self.lr_milestones = lr_milestones if lr_milestones is not None else [100, 160, 220]
        self.lr_gamma      = lr_gamma
        self.curriculum    = curriculum if curriculum is not None else self._DEFAULT_CURRICULUM
        self.ckpt_every    = ckpt_every
        self.output_dir    = output_dir
        self.ckpt_dir      = ckpt_dir
        self.device        = device
        self.verbose       = verbose

        # Runtime state (not sklearn params)
        self._model:     Optional[ActorCritic]          = None
        self._optimizer: Optional[torch.optim.Adam]     = None
        self._scheduler                                  = None
        self._ret_rms    = RunningMeanStd()
        self._buffer     = RolloutBuffer()
        self._ppo_step   = 0
        self._is_fitted  = False
        self._lambda_V   = 0.0
        self._lambda_T   = 0.0
        self._lambda_G   = 0.0
        self._lambda_C   = 0.0

    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        """Instantiate ActorCritic, Adam optimiser, and LR scheduler (idempotent)."""
        if self._model is None:
            self._model = ActorCritic(
                n_bus=self.n_bus, n_gen=self.n_gen,
                input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                log_std_min=self.log_std_min, log_std_max=self.log_std_max,
            ).to(self.device)
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.lr
            )
            self._scheduler = MultiStepLR(
                self._optimizer,
                milestones=self.lr_milestones,
                gamma=self.lr_gamma,
            )

    # ------------------------------------------------------------------
    def _t(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def _compute_gae(self, rewards, values, masks) -> torch.Tensor:
        vals = values.detach().cpu().numpy().flatten()
        vals = np.append(vals, 0.0)
        gae, returns = 0.0, []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * vals[i + 1] * masks[i] - vals[i]
            gae   = delta + self.gamma * self.gae_lambda * masks[i] * gae
            returns.insert(0, gae + vals[i])
        ret = np.array(returns, dtype=np.float64)
        self._ret_rms.update(ret)
        return torch.tensor(
            self._ret_rms.normalise(ret).astype(np.float32), device=self.device
        )

    # ------------------------------------------------------------------
    def _ppo_update(
        self, states, gen_feats, actions, old_lp,
        returns, advantages, adjs, cluster_masks,
    ) -> Tuple[float, float]:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_ep   = max(sum(c[0] for c in self.curriculum), 1)
        ta = tc = n = 0

        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(states.size(0))
            for s in range(0, len(idx), self.batch_size):
                mb   = idx[s: s + self.batch_size]
                mS   = states[mb];  mG  = gen_feats[mb]
                mA   = actions[mb]; mOld= old_lp[mb]
                mR   = returns[mb]; mAdv= advantages[mb]
                mAdj = adjs[mb]
                mCM  = cluster_masks[mb] if cluster_masks is not None else None

                dP, dE, val = self._model(
                    mS, mAdj, gen_feat=mG, cluster_mask=mCM
                )
                lp_P = dP.log_prob(mA[..., : self.n_gen]).sum(-1)
                lp_E = dE.log_prob(
                    torch.cat([mA[..., self.n_gen: 2 * self.n_gen],
                               mA[..., 2 * self.n_gen:]], dim=-1)
                ).sum(-1)
                new_lp = lp_P + lp_E

                if (mOld - new_lp).mean() > 0.05:
                    break

                ent   = (dP.entropy().mean() + dE.entropy().mean()) * 0.5
                ratio = torch.exp(new_lp - mOld)
                al    = -torch.min(
                    ratio * mAdv,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mAdv,
                ).mean()
                cl    = 0.5 * F.mse_loss(val.squeeze(-1), mR)
                ec    = max(0.001, 0.01 * (1 - self._ppo_step / total_ep))
                loss  = al + cl - ec * ent

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
                self._optimizer.step()
                ta += al.item(); tc += cl.item(); n += 1

        self._ppo_step += 1
        self._scheduler.step()
        return ta / max(n, 1), tc / max(n, 1)

    # ------------------------------------------------------------------
    def _update_multipliers(self, gV, gT, gG, gC) -> None:
        def _asc(lam, g):
            return min(
                self.lambda_max,
                max(0.0, (lam + self.lr_lambda * g) * self.lambda_decay),
            )
        self._lambda_V = _asc(self._lambda_V, gV)
        self._lambda_T = _asc(self._lambda_T, gT)
        self._lambda_G = _asc(self._lambda_G, gG)
        self._lambda_C = _asc(self._lambda_C, gC)

    # ------------------------------------------------------------------
    def _flush_buffer(self, env, clusterer) -> None:
        """Run one PPO update from the accumulated buffer, then clear it."""
        s_b   = torch.stack(self._buffer.states).to(self.device)
        g_b   = torch.cat(self._buffer.gen_feats).to(self.device)
        a_b   = torch.cat(self._buffer.actions).to(self.device)
        lp_b  = torch.cat(self._buffer.log_probs).detach().to(self.device)
        v_b   = torch.cat(self._buffer.values).squeeze(-1)
        adj_b = torch.stack(self._buffer.adjs).squeeze(1).to(self.device)
        cm_b  = (
            torch.stack(self._buffer.cluster_masks).squeeze(1).to(self.device)
            if clusterer is not None else None
        )

        bgV = float(np.mean([i.get("g_V", 0.0) for i in self._buffer.infos]))
        bgT = float(np.mean([i.get("g_T", 0.0) for i in self._buffer.infos]))
        bgG = float(np.mean([i.get("g_G", 0.0) for i in self._buffer.infos]))
        bgC = float(np.mean([i.get("g_C", 0.0) for i in self._buffer.infos]))
        self._update_multipliers(bgV, bgT, bgG, bgC)

        rets = self._compute_gae(
            self._buffer.rewards, v_b.cpu(), self._buffer.masks
        ).to(self.device)
        adv = rets - v_b.detach().to(self.device)
        self._ppo_update(s_b, g_b, a_b, lp_b, rets, adv, adj_b, cm_b)
        self._buffer.clear()

    # ------------------------------------------------------------------
    def fit(self, env, clusterer=None) -> "GATPO":
        """
        Train GATPO on a Grid2OpAdapter or any gymnasium-compatible environment.

        Parameters
        ----------
        env : Grid2OpAdapter  (preferred), or any object exposing:
              .reset() → ((bus_feat, gen_feat), info)
              .step(action) → ((bus_feat, gen_feat), reward, done, trunc, info)
              ._y_bus_adj      Tensor  [1, 2, N, N]
              .lambda_V/T/G/C  float (writeable)
              .suez_shock_pct  float (writeable)
              .last_g_V/T/G/C  float
        clusterer : BusClusterer | None
            If None and env is Grid2OpAdapter, env._clusterer is used automatically.

        Returns
        -------
        self
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir,   exist_ok=True)
        self._build_model()
        self._model.train()

        # Auto-resolve clusterer
        _clusterer = clusterer
        if _clusterer is None and isinstance(env, Grid2OpAdapter):
            _clusterer = env._clusterer

        total_eps       = sum(c[0] for c in self.curriculum)
        best_score      = -np.inf
        patience_ctr    = 0
        patience        = 200
        global_ep       = 0
        wave_idx        = 0

        print(f"\n{'='*65}")
        print(f"  GATPO Curriculum Training — {total_eps} episodes | {self.device}")
        print(f"{'='*65}")

        while global_ep < total_eps:
            n_eps, shock_lo, shock_hi, wave_label = self.curriculum[wave_idx]
            shock = (
                float(np.random.uniform(shock_lo, shock_hi))
                if shock_hi > shock_lo else shock_lo
            )
            if hasattr(env, "suez_shock_pct"):
                env.suez_shock_pct = shock
            for attr in ("lambda_V", "lambda_T", "lambda_G", "lambda_C"):
                if hasattr(env, attr):
                    setattr(env, attr, getattr(self, f"_{attr}"))

            state, _ = env.reset()
            ep_rew, ep_eq, ep_vv = 0.0, [], []

            while True:
                bus, gen = state
                st_bus = self._t(bus).unsqueeze(0)
                st_gen = self._t(gen).unsqueeze(0)

                cmask: Optional[torch.Tensor] = None
                if _clusterer is not None and _clusterer.is_fitted:
                    cm_np = (
                        env.get_cluster_mask()
                        if isinstance(env, Grid2OpAdapter)
                        else np.ones(self.n_gen, dtype=np.float32)
                    )
                    cmask = self._t(cm_np).unsqueeze(0)

                adj = getattr(env, "_y_bus_adj", None)
                if adj is not None:
                    adj = adj.to(self.device)

                with torch.no_grad():
                    dP, dE, val = self._model(
                        st_bus, adj=adj, gen_feat=st_gen, cluster_mask=cmask
                    )
                    aP  = dP.sample().clamp(0, 1)
                    aE  = dE.sample()
                    aV  = aE[..., : self.n_gen].clamp(0, 1)
                    aC  = aE[..., self.n_gen:].clamp(0, 1)
                    act = torch.cat([aP, aV, aC], dim=-1)
                    lp  = (
                        dP.log_prob(aP).sum(-1)
                        + dE.log_prob(torch.cat([aV, aC], dim=-1)).sum(-1)
                    )

                next_state, reward, done, _, info = env.step(act.cpu().numpy())

                self._buffer.push(
                    st_bus.squeeze(0), st_gen, act, lp,
                    reward, int(not done), val,
                    adj.clone() if adj is not None
                        else torch.zeros(1, 2, self.n_bus, self.n_bus),
                    info, cmask,
                )

                ep_rew += reward
                ep_eq.append(getattr(env, "last_equity", 0.0))
                ep_vv.append(getattr(env, "last_v_viol",  0))
                state = next_state

                if len(self._buffer) >= self.horizon:
                    self._flush_buffer(env, _clusterer)
                if done:
                    break

            # ── episode bookkeeping ──────────────────────────────────
            avg_eq = float(np.mean(ep_eq)) if ep_eq else 0.0
            steps  = getattr(env, "n_steps", 1)
            ms     = getattr(env, "max_steps", max(steps, 1))
            score  = (0.30 * avg_eq / 100.0
                      + 0.15 * steps / ms
                      - 0.05 * np.mean(ep_vv) / self.n_bus)

            if global_ep % self.verbose == 0:
                lr_now = self._optimizer.param_groups[0]["lr"]
                print(
                    f"  ep {global_ep:>5} | wave {wave_idx+1} | "
                    f"rew {ep_rew:>9.2f} | equity {avg_eq:>6.2f}% | "
                    f"shock {shock:.0%} | lr {lr_now:.2e}"
                )

            if global_ep >= 100 and wave_idx >= 3:
                if score > best_score + 0.01:
                    best_score  = score
                    patience_ctr = 0
                    torch.save(
                        self._model.state_dict(),
                        os.path.join(self.output_dir, "best_gatpo.pth"),
                    )
                else:
                    patience_ctr += 1

            if patience_ctr >= patience:
                print(f"\n[EARLY STOP] ep {global_ep} | best={best_score:.4f}")
                break

            if (global_ep + 1) % self.ckpt_every == 0:
                cp = os.path.join(self.ckpt_dir, f"ckpt_{self._ppo_step:06d}.pth")
                torch.save(self._model.state_dict(), cp)
                print(f"  [CKPT] → {cp}")

            global_ep += 1
            if (global_ep >= sum(c[0] for c in self.curriculum[: wave_idx + 1])
                    and wave_idx < len(self.curriculum) - 1):
                wave_idx += 1

        torch.save(
            self._model.state_dict(),
            os.path.join(self.output_dir, "last_gatpo.pth"),
        )
        print("\n  TRAINING COMPLETE")
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        obs:          Tuple[np.ndarray, np.ndarray],
        adj:          Optional[torch.Tensor] = None,
        cluster_mask: Optional[np.ndarray]  = None,
    ) -> np.ndarray:
        """
        Greedy inference — returns the action with highest probability.

        Parameters
        ----------
        obs          : (bus_feat [n_bus, 6], gen_feat [4*n_gen])
        adj          : Tensor [1, 2, N, N]  or None  – Y-bus admittance.
        cluster_mask : array [n_gen]         or None  – spatial bias mask.

        Returns
        -------
        np.ndarray  [3 * n_gen]  in [0, 1]
        layout  →  [ΔP setpoints | V setpoints | curtailment fractions]
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call .fit() or .load() first.")
        self._model.eval()
        bus, gen = obs
        x  = self._t(np.asarray(bus)).unsqueeze(0)
        g  = self._t(np.asarray(gen)).unsqueeze(0)
        cm = self._t(np.asarray(cluster_mask)).unsqueeze(0) \
             if cluster_mask is not None else None
        adj_d = adj.to(self.device) if adj is not None else None

        with torch.no_grad():
            dP, dE, _ = self._model(x, adj=adj_d, gen_feat=g, cluster_mask=cm)
            action = torch.cat(
                [dP.mean.clamp(0, 1), dE.mean.clamp(0, 1)], dim=-1
            )

        self._model.train()
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    def __call__(
        self,
        obs:          Tuple[np.ndarray, np.ndarray],
        adj:          Optional[torch.Tensor] = None,
        cluster_mask: Optional[np.ndarray]  = None,
    ) -> np.ndarray:
        """
        PyTorch-style callable alias for predict().

        Enables:
            action = model(obs)
            action = model(obs, adj=adj, cluster_mask=cm)
        """
        return self.predict(obs, adj=adj, cluster_mask=cluster_mask)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        if self._model is None:
            raise RuntimeError("No model to save. Call .fit() first.")
        torch.save(self._model.state_dict(), path)
        print(f"  Saved → {path}")

    def load(self, path: str) -> "GATPO":
        """Load weights from *path*. Builds ActorCritic if needed."""
        self._build_model()
        self._model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._is_fitted = True
        print(f"  Loaded ← {path}")
        return self

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict:
        return dict(
            n_bus=self.n_bus, n_gen=self.n_gen,
            input_dim=self.input_dim, hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            log_std_min=self.log_std_min, log_std_max=self.log_std_max,
            lr=self.lr, gamma=self.gamma, gae_lambda=self.gae_lambda,
            clip_eps=self.clip_eps, ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size, grad_clip=self.grad_clip,
            horizon=self.horizon,
            lr_lambda=self.lr_lambda, lambda_max=self.lambda_max,
            lambda_decay=self.lambda_decay,
            rho_v=self.rho_v, rho_t=self.rho_t,
            rho_g=self.rho_g, rho_c=self.rho_c,
            lr_milestones=self.lr_milestones, lr_gamma=self.lr_gamma,
            curriculum=self.curriculum, ckpt_every=self.ckpt_every,
            output_dir=self.output_dir, ckpt_dir=self.ckpt_dir,
            device=self.device, verbose=self.verbose,
        )

    def set_params(self, **params) -> "GATPO":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid GATPO parameter: '{k}'")
            setattr(self, k, v)
        if any(k in params for k in
               ("n_bus", "n_gen", "input_dim", "hidden_dim", "num_heads")):
            self._model = self._optimizer = self._scheduler = None
        return self

    def score(self, env, n_episodes: int = 20) -> float:
        """Composite evaluation score for GridSearchCV / cross-validation."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .score().")
        self._model.eval()
        scores = []
        with torch.no_grad():
            for _ in range(n_episodes):
                state, _ = env.reset()
                eq, vv, steps = [], [], 0
                while True:
                    adj    = getattr(env, "_y_bus_adj", None)
                    action = self.predict(state, adj=adj)
                    state, _, done, _, _ = env.step(action)
                    eq.append(getattr(env, "last_equity", 0.0))
                    vv.append(getattr(env, "last_v_viol",  0))
                    steps += 1
                    if done:
                        break
                ms = getattr(env, "max_steps", max(steps, 1))
                scores.append(
                    0.50 * np.mean(eq) / 100.0
                    + 0.30 * steps / ms
                    - 0.20 * np.mean(vv) / self.n_bus
                )
        self._model.train()
        return float(np.mean(scores))


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  smart_grid_gatpo_lib — end-to-end usage examples")
    print("=" * 68)

    # ------------------------------------------------------------------
    # EXAMPLE A : pandapower   ← your exact requested usage pattern
    # ------------------------------------------------------------------
    print("\n── Example A: pandapower ──────────────────────────────────────")
    try:
        import pandapower as pp

        net     = pp.networks.case118()
        pp.runpp(net, verbose=False)

        adapter = PandapowerAdapter(net, n_clusters=4)
        print(f"  Grid    : case118  |  n_bus={adapter.n_bus}  n_gen={adapter.n_gen}")

        model = GATPO(n_bus=adapter.n_bus, n_gen=adapter.n_gen)
        model._build_model()
        print(f"  Params  : {model._model.count_parameters():,} trainable")

        # In real usage, replace the line above with:
        #   model.load("model.pth")
        # Here we skip the checkpoint and just demo the inference loop.

        total_steps = 5
        for t in range(total_steps):
            # 1. Get observation
            obs = adapter.get_obs(net)

            # 2. (Optional) Y-bus tensor for this timestep
            adj = adapter.get_adj(net)

            # 3. (Optional) spatial cluster mask
            cm  = adapter.get_cluster_mask(net)

            # 4. Inference  — action = model(obs)
            action = model(obs, adj=adj, cluster_mask=cm)

            # 5. Write setpoints back into pandapower
            adapter.apply_action(net, action)

            # 6. Solve power flow
            pp.runpp(net, verbose=False)

            # 7. Check stability
            status = adapter.check_stability(net)
            print(f"  t={t+1}  stable={status['stable']}  "
                  f"max_v_dev={status['max_v_dev']:.4f}  "
                  f"max_load={status['max_loading']:.4f}")

        print("  Example A complete ✓")

    except ImportError:
        print("  pandapower not installed — skipping.")
        print("  Install: pip install pandapower")

    # ------------------------------------------------------------------
    # EXAMPLE B : Grid2Op inference loop
    # ------------------------------------------------------------------
    print("\n── Example B: Grid2Op ─────────────────────────────────────────")
    try:
        import grid2op
        from lightsim2grid.lightSimBackend import LightSimBackend

        g2op_env = grid2op.make("l2rpn_wcci_2022", backend=LightSimBackend())
        adapter  = Grid2OpAdapter(g2op_env, n_clusters=8)
        print(f"  Grid    : l2rpn_wcci_2022  |  n_bus={adapter.n_bus}  n_gen={adapter.n_gen}")

        model = GATPO(n_bus=adapter.n_bus, n_gen=adapter.n_gen)
        # model.load("best_gatpo.pth")    # ← uncomment to load a trained ckpt

        obs, _ = adapter.reset()
        done   = False
        steps  = 0
        while not done and steps < 10:
            adj    = adapter._y_bus_adj
            cm     = adapter.get_cluster_mask()
            action = model(obs, adj=adj, cluster_mask=cm)
            obs, rew, done, _, info = adapter.step(action)
            steps += 1
            print(f"  t={steps}  reward={rew:.3f}  "
                  f"equity={adapter.last_equity:.1f}%  "
                  f"v_viol={adapter.last_v_viol}")

        print("  Example B complete ✓")

    except ImportError:
        print("  Grid2Op / LightSim2Grid not installed — skipping.")
        print("  Install: pip install grid2op lightsim2grid")

    # ------------------------------------------------------------------
    # EXAMPLE C : Dummy tensors — no simulator required
    # ------------------------------------------------------------------
    print("\n── Example C: Dummy tensors (no simulator needed) ─────────────")

    n_bus, n_gen = 14, 5

    # Y-bus builder
    ybb = YBusBuilder(n_bus)
    dummy_adj = torch.randn(1, 2, n_bus, n_bus)

    # Bus clusterer
    bc         = BusClusterer(n_clusters=3)
    gen_to_bus = np.random.randint(0, n_bus, n_gen)
    bc.fit(dummy_adj, gen_to_bus)
    v_pu = np.ones(n_bus, dtype=np.float32) + np.random.randn(n_bus) * 0.05
    mask = bc.get_gen_mask_for_violations(v_pu)
    print(f"  Cluster mask : {mask.round(2)}")

    # GATPO model
    model  = GATPO(n_bus=n_bus, n_gen=n_gen)
    model._build_model()
    obs    = (
        np.random.randn(n_bus, 6).astype(np.float32),
        np.random.randn(4 * n_gen).astype(np.float32),
    )
    action = model(obs, adj=dummy_adj, cluster_mask=mask)
    print(f"  Action shape : {action.shape}   "
          f"range [{action.min():.3f}, {action.max():.3f}]")

    os.makedirs("./output", exist_ok=True)
    model.save("./output/demo_gatpo.pth")
    model.load("./output/demo_gatpo.pth")
    print("  Save / load round-trip OK ✓")

    # sklearn interface
    model.set_params(lr=5e-5, gamma=0.98)
    assert model.lr == 5e-5 and model.gamma == 0.98
    print("  set_params / get_params OK ✓")

    print("\n  All examples complete.")
