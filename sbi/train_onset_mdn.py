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
from torchsummary import summary
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

from data import TraceOnsetDataset, relabel_by_parameter_bins
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
                bucket = bucket[torch.randperm(len(bucket))]
            ordered.append(bucket)
        return iter(torch.cat(ordered, dim=0).tolist())

    def __len__(self):
        return len(self.scores)


class OnsetMDNDataset(Dataset):
    def __init__(self, tracks, mask, class_labels, bin_labels, theta, onset_idx):
        self.tracks = tracks
        self.mask = mask
        self.class_labels = class_labels
        self.bin_labels = bin_labels
        self.theta = theta
        self.onset_idx = onset_idx

    def __len__(self):
        return self.tracks.shape[0]

    def __getitem__(self, idx):
        return (
            self.tracks[idx],
            self.mask[idx],
            self.class_labels[idx],
            self.bin_labels[idx],
            self.theta[idx],
            self.onset_idx[idx],
        )


class WeightedSupConLoss(nn.Module):
    def __init__(self, tau=0.2, sigma=0.1, k=12, eps=1e-8):
        super().__init__()
        self.tau, self.sigma, self.k, self.eps = tau, sigma, k, eps

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
    def __init__(self, kind="linear", eps=1e-8):
        super().__init__()
        self.kind = kind
        self.eps = eps

    @staticmethod
    def _center(K):
        n = K.size(0)
        H = torch.eye(n, device=K.device) - K.new_full((n, n), 1.0 / n)
        return H @ K @ H

    def forward(self, z, theta):
        z = z.to(torch.float32)
        th = theta.to(torch.float32)
        th = (th - th.mean(dim=0, keepdim=True)) / (th.std(dim=0, keepdim=True) + self.eps)
        K = z @ z.t()
        L = th @ th.t()
        Kc = self._center(K)
        Lc = self._center(L)
        num = (Kc * Lc).sum()
        den = (Kc.square().sum().sqrt() * Lc.square().sum().sqrt()) + self.eps
        return -(num / den)


class GaussianMixtureDensityHead(nn.Module):
    def __init__(self, context_dim: int, param_dim: int, n_components: int = 6, hidden_dim: int | None = None):
        super().__init__()
        self.K = n_components
        self.D = param_dim
        self.n_chol = param_dim * (param_dim + 1) // 2
        h = hidden_dim or 4 * context_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, n_components * (1 + param_dim + self.n_chol)),
        )

    def _unpack(self, out):
        K, D, n_chol = self.K, self.D, self.n_chol
        logits, means_flat, chols_flat = out.split([K, K * D, K * n_chol], dim=-1)
        return logits, means_flat.view(-1, K, D), chols_flat.view(-1, K, n_chol)

    def _make_chol(self, chols_flat):
        B, K, _ = chols_flat.shape
        D = self.D
        L = chols_flat.new_zeros(B, K, D, D)
        rows, cols = torch.tril_indices(D, D, device=chols_flat.device)
        L[:, :, rows, cols] = chols_flat
        diag = torch.arange(D, device=chols_flat.device)
        L[:, :, diag, diag] = F.softplus(L[:, :, diag, diag]) + 1e-4
        return L

    def log_prob(self, theta, context):
        out = self.net(context.to(torch.float32))
        logits, means, chols_flat = self._unpack(out)
        L = self._make_chol(chols_flat)
        log_w = F.log_softmax(logits, dim=-1)
        diff = theta.unsqueeze(1) - means
        sol = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
        log_det = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_p_comp = -0.5 * sol.pow(2).sum(-1) - log_det - 0.5 * self.D * math.log(2.0 * math.pi)
        return torch.logsumexp(log_w + log_p_comp, dim=-1)

    @torch.no_grad()
    def sample(self, context, n=1000):
        context = context.view(1, -1).to(torch.float32)
        out = self.net(context)
        logits, means, chols_flat = self._unpack(out)
        L = self._make_chol(chols_flat)
        w = F.softmax(logits.squeeze(0), dim=-1)
        comp_idx = torch.multinomial(w, n, replacement=True)
        sel_means = means[0, comp_idx]
        sel_L = L[0, comp_idx]
        eps = torch.randn(n, self.D, device=context.device)
        return sel_means + torch.einsum("nij,nj->ni", sel_L, eps)


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
    theta = torch.stack([gamma0, gamma1, torch.log(sd1)], dim=1)
    return theta, ["gamma0", "gamma1", "log_sd1"]


def add_post_onset_feature(tracks, onset_idx_raw):
    n, t_max, _ = tracks.shape
    time_idx = torch.arange(t_max, device=tracks.device)[None, :]
    post = (time_idx >= onset_idx_raw.long()[:, None]).to(tracks.dtype)
    return torch.cat([tracks, post.unsqueeze(-1)], dim=-1)


def apply_mask_to_features(tracks, mask):
    return tracks.masked_fill(mask.unsqueeze(-1), 0.0)


def fit_standard_scaler(tracks, mask):
    X = tracks.view(-1, tracks.shape[-1])
    M = mask.view(-1)
    X_valid = X[~M]
    return X_valid.mean(dim=0), X_valid.std(dim=0, unbiased=False).clamp_min(1e-6)


def apply_standard_scaler(tracks, mean, std, eps=1e-8):
    return (tracks - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)


def evaluate_embedding_quality(embeddings, labels, sample_size=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    if sample_size is not None and embeddings.shape[0] > sample_size:
        idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        X, y = embeddings[idx], labels[idx]
    else:
        X, y = embeddings, labels
    out = {}
    try:
        out["silhouette"] = float(silhouette_score(X, y, metric="euclidean"))
    except Exception:
        out["silhouette"] = float("nan")
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, y))
    except Exception:
        out["davies_bouldin"] = float("nan")
    return out


def make_class_context(emb, class_labels, theta=None, min_tracks=2):
    contexts = []
    targets = []
    counts = []
    for c in torch.unique(class_labels):
        idx = torch.nonzero(class_labels == c, as_tuple=False).flatten()
        if idx.numel() < min_tracks:
            continue
        z = emb[idx]
        mean = z.mean(dim=0)
        std = z.std(dim=0, unbiased=False)
        log_n = torch.log(torch.tensor([float(idx.numel())], device=emb.device, dtype=emb.dtype))
        contexts.append(torch.cat([mean, std, log_n], dim=0))
        counts.append(idx.numel())
        if theta is not None:
            targets.append(theta[idx].mean(dim=0))
    if not contexts:
        if theta is None:
            return None, None, None
        return None, None, None
    C = torch.stack(contexts, dim=0)
    T = torch.stack(targets, dim=0) if theta is not None else None
    return C, T, counts


def train_model(
    *,
    dataset_path="coupled_ou_tracks_onset.h5",
    n_head=2,
    num_layers=2,
    latent_dim=4,
    d_model=16,
    batch_size=512,
    epochs=1000,
    lr=1e-3,
    lamb=0.1,
    lambda_theta=0.5,
    lambda_hsic=0.25,
    lambda_npe=1.0,
    n_mdn_components=6,
    mdn_hidden_dim=None,
    mdn_min_tracks_per_class=2,
    clip_norm=5.0,
    device=None,
    use_amp=False,
    out_model="models/onset_mdn.pth",
    plot_dir="plots/training_mdn",
    wandb_mode="offline",
    train_full_tracks=True,
):
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(dataset_path, "r") as f:
        if train_full_tracks and "tracks_obs_full" in f:
            tracks = torch.tensor(f["tracks_obs_full"][...]).float()
            mask = torch.zeros(tracks.shape[:2], dtype=torch.bool)
        else:
            tracks = torch.tensor(f["tracks"][...]).float()
            mask = torch.tensor(f["mask"][...]).bool()
        class_labels = torch.tensor(f["labels"][...]).long()
        onset_idx = torch.tensor(f["onset_idx"][...]).long()
        theta_raw, theta_names = read_onset_theta(f, state_index=-1)

    onset_idx_raw = onset_idx.float()
    theta_mean = theta_raw.mean(dim=0, keepdim=True)
    theta_std = theta_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    theta = (theta_raw - theta_mean) / theta_std
    bin_labels, _ = relabel_by_parameter_bins(theta, n_bins=[3, 3, 3], strategy="quantile")

    idx_train, idx_test = train_test_split(
        np.arange(len(class_labels)),
        test_size=0.10,
        random_state=42,
        stratify=class_labels.numpy(),
    )

    tracks_train_xy, tracks_test_xy = tracks[idx_train], tracks[idx_test]
    mask_train, mask_test = mask[idx_train], mask[idx_test]
    class_train, class_test = class_labels[idx_train], class_labels[idx_test]
    bin_train, bin_test = bin_labels[idx_train], bin_labels[idx_test]
    theta_train, theta_test = theta[idx_train], theta[idx_test]
    onset_train, onset_test = onset_idx_raw[idx_train], onset_idx_raw[idx_test]
    time_idx = torch.arange(tracks.shape[1])[None, :]
    visible_post_len_train = ((time_idx[:, :tracks.shape[1]] >= onset_train[:, None]) & (~mask_train)).sum(dim=1)

    mean, std = fit_standard_scaler(tracks_train_xy, mask_train)
    tracks_train_xy = apply_standard_scaler(tracks_train_xy, mean, std)
    tracks_test_xy = apply_standard_scaler(tracks_test_xy, mean, std)
    tracks_train = apply_mask_to_features(add_post_onset_feature(tracks_train_xy, onset_train), mask_train)
    tracks_test = apply_mask_to_features(add_post_onset_feature(tracks_test_xy, onset_test), mask_test)

    train_ds = OnsetMDNDataset(tracks_train, mask_train, class_train, bin_train, theta_train, onset_train)
    test_ds = OnsetMDNDataset(tracks_test, mask_test, class_test, bin_test, theta_test, onset_test)
    train_sampler = BucketCurriculumSampler(visible_post_len_train, batch_size=batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=8, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

    model = TrajectoryTransformer(
        t_max=tracks.shape[1],
        d_model=d_model,
        nhead=n_head,
        num_layers=num_layers,
        embed_dim=latent_dim,
        num_classes=int(bin_labels.max().item()) + 1,
        input_dim=3,
    ).to(DEVICE)
    try:
        summary(model, input_data={"x": tracks_train[:1].to(DEVICE), "mask": mask_train[:1].to(DEVICE)})
    except Exception:
        pass

    reg_head = nn.Linear(latent_dim, theta.shape[1]).to(DEVICE)
    mdn_head = GaussianMixtureDensityHead(
        context_dim=2 * latent_dim + 1,
        param_dim=theta.shape[1],
        n_components=n_mdn_components,
        hidden_dim=mdn_hidden_dim,
    ).to(DEVICE)

    scaler = GradScaler("cuda", enabled=use_amp and str(DEVICE).startswith("cuda"))
    supcon_criterion = WeightedSupConLoss(tau=0.2, sigma=1.0, k=16).to(DEVICE)
    class_criterion = nn.CrossEntropyLoss()
    hsic_criterion = CKALoss(kind="linear").to(DEVICE)
    optimizer = torch.optim.AdamW(itertools.chain(model.parameters(), reg_head.parameters(), mdn_head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 5,
        total_steps=epochs * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

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
            lambda_npe=lambda_npe,
            n_mdn_components=n_mdn_components,
            mdn_context="mean_std_log_n",
            train_full_tracks=train_full_tracks,
            theta_names=theta_names,
        ),
    )

    model.train(); reg_head.train(); mdn_head.train()
    for epoch in tqdm(range(1, epochs + 1)):
        for xb, mb, cb, yb, tb, ob in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)
            cb = cb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            tb = tb.to(DEVICE, non_blocking=True)
            with autocast("cuda", enabled=use_amp and str(DEVICE).startswith("cuda")):
                emb, logits = model(xb, mask=mb)
                theta_pred = reg_head(emb)
                loss_wsup = supcon_criterion(emb, tb)
                loss_class = class_criterion(logits, yb)
                loss_theta = F.smooth_l1_loss(theta_pred, tb)
                loss_hsic = hsic_criterion(emb, tb)
                ctx, target, _ = make_class_context(emb, cb, tb, min_tracks=mdn_min_tracks_per_class)
                if ctx is None:
                    loss_npe = emb.sum() * 0.0
                else:
                    loss_npe = -mdn_head.log_prob(target, ctx).mean()
                loss = loss_wsup + lamb * loss_class + lambda_theta * loss_theta + lambda_hsic * loss_hsic + lambda_npe * loss_npe

            wandb.log({
                "training/loss_total": float(loss.detach().cpu()),
                "training/loss_supcon": float(loss_wsup.detach().cpu()),
                "training/loss_class": float(loss_class.detach().cpu()),
                "training/loss_theta": float(loss_theta.detach().cpu()),
                "training/loss_hsic": float(loss_hsic.detach().cpu()),
                "training/loss_npe": float(loss_npe.detach().cpu()),
            })
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(model.parameters(), reg_head.parameters(), mdn_head.parameters()), clip_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)

    model.eval(); reg_head.eval(); mdn_head.eval()
    emb_list, bin_list = [], []
    val_nll = []
    with torch.no_grad():
        for xb, mb, cb, yb, tb, ob in test_loader:
            xb = xb.to(DEVICE); mb = mb.to(DEVICE); cb = cb.to(DEVICE); tb = tb.to(DEVICE)
            emb, _ = model(xb, mask=mb)
            emb_list.append(emb.cpu()); bin_list.append(yb)
            ctx, target, _ = make_class_context(emb, cb, tb, min_tracks=mdn_min_tracks_per_class)
            if ctx is not None:
                val_nll.append((-mdn_head.log_prob(target, ctx)).mean().cpu())
    emb_arr = torch.cat(emb_list).numpy()
    lbl_arr = torch.cat(bin_list).numpy()
    metrics = evaluate_embedding_quality(emb_arr, lbl_arr)
    metrics["val_mdn_nll"] = float(torch.stack(val_nll).mean()) if val_nll else float("nan")
    print(metrics)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    if emb_arr.shape[1] >= 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(emb_arr[:, 0], emb_arr[:, 1], c=lbl_arr, s=8, alpha=0.35)
        plt.xlabel("latent dim 0"); plt.ylabel("latent dim 1")
        plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "mdn_embeddings.pdf")); plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "reg_head_state_dict": reg_head.state_dict(),
            "mdn_head_state_dict": mdn_head.state_dict(),
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
            "mdn_n_components": n_mdn_components,
            "mdn_context_dim": 2 * latent_dim + 1,
            "mdn_context_mode": "mean_std_log_n",
            "mdn_min_tracks_per_class": mdn_min_tracks_per_class,
            "mdn_hidden_dim": mdn_hidden_dim if mdn_hidden_dim is not None else 4 * (2 * latent_dim + 1),
        },
        out_model,
    )
    run.finish()
    return metrics


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Set-level Trajectory Transformer + MDN training script")
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--lambda_theta", type=float, default=0.5)
    parser.add_argument("--lambda_hsic", type=float, default=0.25)
    parser.add_argument("--lambda_npe", type=float, default=1.0)
    parser.add_argument("--n_mdn_components", type=int, default=6)
    parser.add_argument("--mdn_hidden_dim", type=int, default=None)
    parser.add_argument("--mdn_min_tracks_per_class", type=int, default=2)
    parser.add_argument("--dataset_path", type=str, default="coupled_ou_tracks_onset.h5")
    parser.add_argument("--out_model", type=str, default="models/onset_mdn.pth")
    parser.add_argument("--plot_dir", type=str, default="plots/training_mdn")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--train_full_tracks", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=False)
    args = parser.parse_args()
    metrics = train_model(**vars(args))
    print(json.dumps(metrics, indent=2))
