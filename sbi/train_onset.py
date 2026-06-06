import itertools
import math
import os

import h5py
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import CircleLoss
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary
from tqdm import tqdm

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

import wandb

from data import TraceDataset, TraceOnsetDataset, relabel_by_parameter_bins
from models import TrajectoryTransformer


class BucketCurriculumSampler(Sampler):
    def __init__(self, scores, batch_size, n_buckets=5, shuffle_within_bucket=True):
        self.scores = torch.as_tensor(scores)
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.shuffle_within_bucket = shuffle_within_bucket

        order = torch.argsort(self.scores, descending=True)
        self.buckets = torch.chunk(order, n_buckets)

    def __iter__(self):
        ordered = []
        for bucket in self.buckets:
            bucket = bucket.clone()
            if self.shuffle_within_bucket and len(bucket) > 1:
                perm = torch.randperm(len(bucket))
                bucket = bucket[perm]
            ordered.append(bucket)

        ordered = torch.cat(ordered, dim=0)
        return iter(ordered.tolist())

    def __len__(self):
        return len(self.scores)


class WeightedSupConLoss(nn.Module):
    def __init__(self, tau=0.2, sigma=0.1, k=12, eps=1e-8, smooth=1e-3):
        super().__init__()
        self.tau, self.sigma, self.k, self.eps, self.smooth = tau, sigma, k, eps, smooth

    def forward(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=1).to(torch.float32)
        S = (z @ z.t()) / self.tau
        n = S.size(0)

        I = torch.eye(n, dtype=torch.bool, device=S.device)
        S = S.masked_fill(I, -1e4)

        logprob = S - torch.logsumexp(S, dim=1, keepdim=True)
        logprob = logprob.masked_fill(I, 0.0)

        with torch.no_grad():
            th = theta.to(torch.float32)
            D = torch.cdist(th, th)
            W = torch.exp(-(D * D) / (2 * self.sigma * self.sigma + 1e-12))
            W = W.masked_fill(I, 0.0)

            if self.k:
                k_eff = min(self.k, n - 1)
                if k_eff > 0:
                    idx = torch.topk(W, k_eff, dim=1).indices
                    M = torch.zeros_like(W).scatter(1, idx, 1.0)
                    W = W * M

            rowsum = W.sum(dim=1, keepdim=True)
            offdiag = (~I).float()
            uniform = offdiag / offdiag.sum(dim=1, keepdim=True).clamp_min(1.0)
            W = torch.where(rowsum > 0, W / (rowsum + 1e-12), uniform)

        return -(W * logprob).sum(dim=1).mean()


class CKALoss(nn.Module):
    def __init__(
        self,
        kind: str = "linear",
        sigma_z: float | None = None,
        sigma_theta: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.kind = kind
        self.sigma_z = sigma_z
        self.sigma_theta = sigma_theta
        self.eps = eps

    @staticmethod
    def _gram_linear(x: torch.Tensor) -> torch.Tensor:
        return x @ x.t()

    @staticmethod
    def _gram_rbf(x: torch.Tensor, sigma: float | None) -> torch.Tensor:
        d2 = torch.cdist(x, x).pow(2)
        if sigma is None:
            v = d2[d2 > 0]
            sigma = torch.sqrt(v.median() + 1e-12)
        return torch.exp(-d2 / (2 * sigma * sigma + 1e-12))

    @staticmethod
    def _center(K: torch.Tensor) -> torch.Tensor:
        n = K.size(0)
        H = torch.eye(n, device=K.device) - K.new_full((n, n), 1.0 / n)
        return H @ K @ H

    def forward(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        z = z.to(torch.float32)
        th = theta.to(torch.float32)
        th = (th - th.mean(dim=0, keepdim=True)) / (
            th.std(dim=0, keepdim=True) + self.eps
        )

        if self.kind == "linear":
            K = self._gram_linear(z)
            L = self._gram_linear(th)
        else:
            K = self._gram_rbf(z, self.sigma_z)
            L = self._gram_rbf(th, self.sigma_theta)

        Kc = self._center(K)
        Lc = self._center(L)

        num = (Kc * Lc).sum()
        den = (Kc.square().sum().sqrt() * Lc.square().sum().sqrt()) + self.eps
        cka = num / den
        return -cka


def read_onset_theta(f, state_index: int = -1):
    gamma0 = torch.tensor(f["gamma1"][...]).float()
    gamma1 = torch.tensor(f["gamma2"][...]).float()
    sd1 = torch.tensor(f["sigma_r"][...]).float()

    if gamma0.ndim == 2:
        gamma0 = gamma0[:, state_index]
    if gamma1.ndim == 2:
        gamma1 = gamma1[:, state_index]
    if sd1.ndim == 2:
        sd1 = sd1[:, state_index]

    return torch.stack([gamma0, gamma1, torch.log(sd1)], dim=1), [
        "gamma0",
        "gamma1",
        "log_sd1",
    ]


def evaluate_embedding_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 10_000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compute several complementary metrics that quantify how well the
    learned embedding separates the labelled classes.

    Returns
    -------
    {
      "silhouette"       : float,   # ↑ better
      "davies_bouldin"   : float,   # ↓ better
      "intra_class_dist" : float,
      "inter_class_dist" : float,
      "contrastive_ratio": float    # intra / inter  (<< 1 is good)
    }
    """
    rng = np.random.default_rng(random_state)
    if sample_size is not None and embeddings.shape[0] > sample_size:
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        X, y = embeddings[idx], labels[idx]
    else:
        X, y = embeddings, labels

    # (1) label‑aware clustering scores --------------------------------
    sil = float(silhouette_score(X, y, metric="euclidean"))
    db = float(davies_bouldin_score(X, y))

    # (2) mean intra‑class distance ------------------------------------
    intra_sum, intra_cnt = 0.0, 0
    for cid in np.unique(y):
        Xi = torch.tensor(X[y == cid])
        if Xi.shape[0] < 2:
            continue
        d = torch.cdist(Xi, Xi)  # pairwise
        intra_sum += (d.sum() - d.diag().sum()).item()
        intra_cnt += d.numel() - Xi.shape[0]
    intra_mean = intra_sum / intra_cnt if intra_cnt else float("nan")

    # (3) mean inter‑class distance (between centroids) ----------------
    centroids = np.vstack([X[y == cid].mean(axis=0) for cid in np.unique(y)])
    d_cent = torch.cdist(torch.tensor(centroids), torch.tensor(centroids))
    inter_mean = d_cent[d_cent > 0].mean().item()

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "intra_class_dist": intra_mean,
        "inter_class_dist": inter_mean,
        "contrastive_ratio": intra_mean / inter_mean
        if inter_mean > 0
        else float("nan"),
    }


def train_model(
    *,
    dataset_path: str = "coupled_ou_tracks_onset.h5",
    n_head: int = 2,
    num_layers: int = 2,
    latent_dim: int = 4,
    d_model: int = 16,
    batch_size: int = 512,
    epochs: int = 10_000,
    lr: float = 1e-3,
    lamb: float = 0.5,
    lambda_theta: float = 0.5,
    lambda_onset: float = 0.25,
    lambda_hsic: float = 0.25,
    clip_norm: float = 5.0,
    device: str | None = None,
    use_amp: bool = False,
    out_model: str = "models/onset_embedding.pth",
    plot_dir: str = "plots/training",
    wandb_mode: str = "online",
    train_full_tracks: bool = True,
) -> dict[str, float]:
    """
    End‑to‑end training routine.

    Returns
    -------
    dict
        Final embedding‑quality metrics computed on the held‑out test set.
    """
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(dataset_path, "r") as f:
        if train_full_tracks and "tracks_obs_full" in f:
            tracks = torch.tensor(f["tracks_obs_full"][...]).float()
            mask = torch.zeros(tracks.shape[:2], dtype=torch.bool)
        else:
            tracks = torch.tensor(f["tracks"][...]).float()
            mask = torch.tensor(f["mask"][...]).bool()
        labels = torch.tensor(f["labels"][...]).long()
        onset_idx = torch.tensor(f["onset_idx"][...]).long()

        theta, theta_names = read_onset_theta(f, state_index=-1)

    onset_idx_raw = onset_idx.float()

    theta_mean = theta.mean(dim=0, keepdim=True)
    theta_std = theta.std(dim=0, keepdim=True).clamp_min(1e-6)
    theta = (theta - theta_mean) / theta_std

    time_idx = torch.arange(tracks.shape[1])[None, :]
    visible_post_len = ((time_idx >= onset_idx_raw[:, None]) & (~mask)).sum(dim=1)

    run = wandb.init(
        project="NK",
        entity="dpohanlon",
        mode=wandb_mode,
        config=dict(
            n_head=n_head,
            num_layers=num_layers,
            latent_dim=latent_dim,
            d_model=d_model,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            n_tracks=len(tracks),
            n_labels=len(np.unique(labels)),
            lamb=lamb,
            lambda_onset=lambda_onset,
            train_full_tracks=train_full_tracks,
            theta_names=theta_names,
        ),
    )

    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=0.10,
        random_state=42,
        stratify=labels.numpy(),
    )

    labels, label_info = relabel_by_parameter_bins(
        theta, n_bins=[3, 3, 3], strategy="quantile"
    )

    tracks_train_xy, tracks_test_xy = tracks[idx_train], tracks[idx_test]
    mask_train, mask_test = mask[idx_train], mask[idx_test]
    labels_train, labels_test = labels[idx_train], labels[idx_test]
    theta_train, theta_test = theta[idx_train], theta[idx_test]
    visible_post_len_train = visible_post_len[idx_train]
    onset_idx_train_raw, onset_idx_test_raw = (
        onset_idx_raw[idx_train],
        onset_idx_raw[idx_test],
    )

    mean, std = fit_standard_scaler(tracks_train_xy, mask_train)
    tracks_train_xy = apply_standard_scaler(tracks_train_xy, mean, std)
    tracks_test_xy = apply_standard_scaler(tracks_test_xy, mean, std)
    tracks_train = apply_mask_to_features(
        add_post_onset_feature(tracks_train_xy, onset_idx_train_raw), mask_train
    )
    tracks_test = apply_mask_to_features(
        add_post_onset_feature(tracks_test_xy, onset_idx_test_raw), mask_test
    )

    train_ds = TraceOnsetDataset(
        tracks_train, mask_train, labels_train, theta_train, onset_idx_train_raw
    )
    test_ds = TraceOnsetDataset(
        tracks_test, mask_test, labels_test, theta_test, onset_idx_test_raw
    )

    train_sampler = BucketCurriculumSampler(
        scores=visible_post_len_train,
        batch_size=batch_size,
        n_buckets=5,
        shuffle_within_bucket=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    # ---------- MODEL -------------------------------------------------
    model = TrajectoryTransformer(
        t_max=tracks.shape[1],
        d_model=d_model,
        nhead=n_head,
        num_layers=num_layers,
        embed_dim=latent_dim,
        num_classes=int(labels.max().item()) + 1,
        input_dim=3,
    ).to(DEVICE)

    summary(
        model,
        input_data={
            "x": tracks_train[:1].to(DEVICE),
            "mask": mask_train[:1].to(DEVICE),
        },
    )

    reg_head = nn.Linear(latent_dim, theta.shape[1]).to(DEVICE)

    hsic_criterion = CKALoss(kind="linear").to(DEVICE)
    scaler = GradScaler("cuda", enabled=use_amp and (DEVICE.startswith("cuda")))

    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), reg_head.parameters())
    )

    # circle_criterion = CircleLoss()
    supcon_criterion = WeightedSupConLoss(tau=0.2, sigma=1.0, k=16).to(DEVICE)
    class_criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 5,
        total_steps=epochs * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    # ---------- TRAIN -------------------------------------------------
    model.train()
    for epoch in tqdm(range(1, epochs + 1)):
        for xb, mb, yb, tb, ob in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            ob = ob.to(DEVICE, non_blocking=True)
            with autocast("cuda", enabled=use_amp and (DEVICE.startswith("cuda"))):
                emb, logits = model(xb, mask=mb)
                theta_pred = reg_head(emb)
                loss_wsup = supcon_criterion(emb, tb)
                loss_class = class_criterion(logits, yb)
                loss_theta = F.smooth_l1_loss(theta_pred, tb)
                loss_hsic = hsic_criterion(emb, tb)
                loss = (
                    loss_wsup
                    + lamb * loss_class
                    + lambda_theta * loss_theta
                    + lambda_hsic * loss_hsic
                )

            wandb.log(
                {
                    "training/loss_total": loss.detach().item(),
                    "training/loss_supcon": loss_wsup.detach().item(),
                    "training/loss_class": loss_class.detach().item(),
                    "training/loss_theta": loss_theta.detach().item(),
                    "training/loss_hsic": loss_hsic.detach().item(),
                }
            )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(model.parameters(), reg_head.parameters()), clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    # ---------- VALIDATE ---------------------------------------------
    model.eval()
    emb_list, lbl_list = [], []
    with torch.no_grad():
        for xb, mb, yb, tb, ob in test_loader:
            emb, _ = model(xb.to(DEVICE), mask=mb.to(DEVICE))
            emb_list.append(emb.cpu())
            lbl_list.append(yb)
    metrics = evaluate_embedding_quality(
        torch.cat(emb_list).numpy(),
        torch.cat(lbl_list).numpy(),
    )

    emb_arr = torch.cat(emb_list).numpy()  # [N_test,l_dim]
    lbl_arr = torch.cat(lbl_list).numpy()  # [N_test]

    metrics = evaluate_embedding_quality(emb_arr, lbl_arr)
    print(metrics)

    x0, x1 = emb_arr[:, 0], emb_arr[:, 1]

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)

    # 2D plot

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))

    for cls in np.unique(lbl_arr):
        m = lbl_arr == cls
        plt.scatter(
            x0[m],
            x1[m],
            s=12,
            alpha=0.35,
            color=cmap(int(cls) % 10),
            label=f"class {cls}",
        )

    plt.xlabel("latent dim 0")
    plt.ylabel("latent dim 1")
    plt.legend(markerscale=2, fontsize="x-small", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "embeddings.pdf"))
    plt.clf()

    # 1D plots, x0

    bins = np.linspace(np.percentile(x0, 1), np.percentile(x0, 99), 50)

    plt.figure(figsize=(6, 6))

    for cls in np.unique(lbl_arr):
        m = lbl_arr == cls
        plt.hist(
            x0[m],
            bins=bins,
            histtype="step",
            color=cmap(int(cls) % 10),
            label=f"class {cls}",
        )

    plt.xlabel("latent dim 0")
    plt.legend(markerscale=2, fontsize="x-small", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "embeddings_0.pdf"))
    plt.clf()

    # 1D plots, x1

    bins = np.linspace(np.percentile(x1, 1), np.percentile(x1, 99), 50)

    plt.figure(figsize=(6, 6))

    for cls in np.unique(lbl_arr):
        m = lbl_arr == cls
        plt.hist(
            x1[m],
            bins=bins,
            histtype="step",
            color=cmap(int(cls) % 10),
            label=f"class {cls}",
        )

    plt.xlabel("latent dim 1")
    plt.legend(markerscale=2, fontsize="x-small", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "embeddings_1.pdf"))
    plt.clf()

    with h5py.File(dataset_path, "r") as f:
        pvals = {
            "gamma0": np.asarray(f["gamma1"][...]).reshape(-1)[idx_test],
            "gamma1": np.asarray(f["gamma2"][...]).reshape(-1)[idx_test],
            "sd1": np.asarray(f["sigma_r"][...]).reshape(-1)[idx_test],
        }

    for name, v in pvals.items():
        v = np.asarray(v).reshape(-1)

        if len(v) != len(x0):
            raise ValueError(
                f"{name} has length {len(v)}, but embedding has length {len(x0)}"
            )

        vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)

        plt.figure(figsize=(6, 6))
        sc = plt.scatter(
            x0,
            x1,
            c=v,
            s=12,
            alpha=0.6,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel("latent dim 0")
        plt.ylabel("latent dim 1")
        cbar = plt.colorbar(sc)
        cbar.set_label(name)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"embeddings_color_{name}.pdf"))
        plt.clf()

    phi = np.arctan2(x1, x0)

    for name, v in pvals.items():
        v = np.asarray(v).reshape(-1)

        if len(v) != len(phi):
            raise ValueError(
                f"{name} has length {len(v)}, but phi has length {len(phi)}"
            )

        plt.figure(figsize=(6, 4))
        plt.scatter(v, phi, s=8, alpha=0.5)
        plt.xlabel(name)
        plt.ylabel("angle (rad)")
        plt.ylim(-np.pi, np.pi)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"embedding_angle_vs_{name}.pdf"))
        plt.clf()

    artifact = wandb.Artifact("embedding_plots", type="analysis")
    for fname in [
        os.path.join(plot_dir, f"embeddings.pdf"),
        os.path.join(plot_dir, f"embeddings_0.pdf"),
        os.path.join(plot_dir, f"embeddings_1.pdf"),
        os.path.join(plot_dir, f"embeddings_color_gamma0.pdf"),
        os.path.join(plot_dir, f"embeddings_color_gamma1.pdf"),
        os.path.join(plot_dir, f"embeddings_color_sd1.pdf"),
        os.path.join(plot_dir, f"embedding_angle_vs_gamma0.pdf"),
        os.path.join(plot_dir, f"embedding_angle_vs_gamma1.pdf"),
        os.path.join(plot_dir, f"embedding_angle_vs_sd1.pdf"),
    ]:
        artifact.add_file(fname)

    wandb.log_artifact(artifact)
    wandb.log(metrics)  # push silhouette, DB‑index, etc.

    # ---------- SAVE CHECKPOINT --------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "reg_head_state_dict": reg_head.state_dict(),
            "scaler_mean": mean,
            "scaler_std": std,
            "d_model": d_model,
            "n_head": n_head,
            "num_layers": num_layers,
            "latent_dim": latent_dim,
            "input_dim": 3,
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "theta_names": theta_names,
            "target_schema": "onset_post_drift_shared_diffusion_v1",
            "sd0_fixed": 1.0,
            "diffusion_shared_pre_post": True,
            "pre_drift_model": "derived_decay_zero_coupling",
            "train_full_tracks": train_full_tracks,
        },
        out_model,
    )

    run.finish()  # close the run cleanly

    return metrics


def add_post_onset_feature(
    tracks: torch.Tensor, onset_idx_raw: torch.Tensor
) -> torch.Tensor:
    n, t_max, _ = tracks.shape
    time_idx = torch.arange(t_max, device=tracks.device)[None, :]
    post = (time_idx >= onset_idx_raw.long()[:, None]).to(tracks.dtype)
    return torch.cat([tracks, post.unsqueeze(-1)], dim=-1)


def apply_mask_to_features(tracks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tracks.masked_fill(mask.unsqueeze(-1), 0.0)


def fit_standard_scaler(
    tracks: torch.Tensor,  # [N, T, 2]
    mask: torch.Tensor,  # [N, T], True = padded/invalid
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‐coordinate mean & std so that (x - mean)/std ~ N(0,1),
    only over the positions where mask == True.
    Returns
    -------
    mean : Tensor[2]
    std  : Tensor[2]
    """
    N, T, D = tracks.shape
    assert D == 2
    X = tracks.view(-1, D)
    M = mask.view(-1)  # [N*T]
    X_valid = X[~M]  # [#valid, 2]
    mean = X_valid.mean(dim=0)  # [2]
    std = X_valid.std(dim=0, unbiased=False)  # [2]
    return mean, std


def apply_standard_scaler(
    tracks: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Apply the fitted scaler:
      x' = (x - mean) / (std + eps)
    on a tensor of shape [N,T,2].
    """
    return (tracks - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Trajectory‑Transformer training script"
    )
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument(
        "--dataset_path", type=str, default="coupled_ou_tracks_onset.h5"
    )
    parser.add_argument("--out_model", type=str, default="models/onset_embedding.pth")
    parser.add_argument("--plot_dir", type=str, default="plots/training")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--train_full_tracks", action="store_true", default=False)
    args = parser.parse_args()

    metrics = train_model(**vars(args))
    print(json.dumps(metrics, indent=2))
