import json
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TrajectoryTransformer


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


def _infer_num_layers(state):
    layers = set()
    prefix = "encoder.layers."
    for k in state.keys():
        if k.startswith(prefix):
            idx = k[len(prefix):].split(".", 1)[0]
            if idx.isdigit():
                layers.add(int(idx))
    if not layers:
        raise KeyError("Could not infer num_layers from checkpoint state_dict.")
    return max(layers) + 1


def load_model_and_mdn(ckpt_path, *, t_max, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "mdn_head_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain mdn_head_state_dict.")
    state = ckpt["model_state_dict"]
    input_dim = int(state["coord_proj.weight"].shape[1])
    d_model = int(state["coord_proj.weight"].shape[0])
    embed_dim = int(state["proj.3.weight"].shape[0])
    num_classes = int(state["classifier.weight"].shape[0])
    dim_ff = int(state["encoder.layers.0.linear1.weight"].shape[0])
    num_layers = int(ckpt.get("num_layers", _infer_num_layers(state)))
    nhead = int(ckpt.get("n_head", ckpt.get("nhead", 2)))
    dropout = float(ckpt.get("dropout", 0.1))
    model = TrajectoryTransformer(
        t_max=t_max,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        embed_dim=embed_dim,
        num_classes=num_classes,
        input_dim=input_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    theta_mean = torch.as_tensor(ckpt["theta_mean"], device=device, dtype=torch.float32)
    theta_std = torch.as_tensor(ckpt["theta_std"], device=device, dtype=torch.float32)
    param_dim = theta_mean.shape[-1]
    n_components = int(ckpt.get("mdn_n_components", 6))
    context_dim = int(ckpt.get("mdn_context_dim", 2 * embed_dim + 1))
    hidden_dim = int(ckpt.get("mdn_hidden_dim", 4 * context_dim))
    mdn_head = GaussianMixtureDensityHead(
        context_dim=context_dim,
        param_dim=param_dim,
        n_components=n_components,
        hidden_dim=hidden_dim,
    ).to(device)
    mdn_head.load_state_dict(ckpt["mdn_head_state_dict"], strict=True)
    mdn_head.eval()

    mean = torch.as_tensor(ckpt["scaler_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(ckpt["scaler_std"], device=device, dtype=torch.float32)
    theta_names = ckpt.get("theta_names", ["gamma0", "gamma1", "log_sd1"])
    return model, mdn_head, mean, std, theta_mean, theta_std, theta_names, ckpt


def apply_standard_scaler(tracks, mean, std, eps=1e-8):
    return (tracks - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)


def add_post_onset_feature(tracks, onset_idx_raw):
    n, t_max, _ = tracks.shape
    time_idx = torch.arange(t_max, device=tracks.device)[None, :]
    post = (time_idx >= onset_idx_raw.long()[:, None]).to(tracks.dtype)
    return torch.cat([tracks, post.unsqueeze(-1)], dim=-1)


def apply_mask_to_features(tracks, mask):
    return tracks.masked_fill(mask.unsqueeze(-1), 0.0)


def select_subset(n, max_tracks=None, mode="first", seed=0):
    if max_tracks is None or max_tracks <= 0 or max_tracks >= n:
        return np.arange(n)
    if mode == "first":
        return np.arange(max_tracks)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(n)[:max_tracks]
    raise ValueError(f"Unknown subset mode: {mode}")


def read_selected_tracks(data_file, track_class=None, track_id=None, max_tracks=None, subset_mode="first", subset_seed=0, train_full_tracks=False):
    with h5py.File(data_file, "r") as f:
        labels = np.asarray(f["labels"][:])
        onset_idx_all = np.asarray(f["onset_idx"][:])
        if track_id is not None:
            idx = np.array([int(track_id)])
        elif track_class is not None:
            idx = np.where(labels == int(track_class))[0]
        else:
            idx = np.arange(labels.shape[0])
        sub = select_subset(len(idx), max_tracks=max_tracks, mode=subset_mode, seed=subset_seed)
        idx = idx[sub]
        if train_full_tracks and "tracks_obs_full" in f:
            tracks = np.asarray(f["tracks_obs_full"][idx], dtype=np.float32)
            mask = np.zeros(tracks.shape[:2], dtype=bool)
        else:
            tracks = np.asarray(f["tracks"][idx], dtype=np.float32)
            mask = np.asarray(f["mask"][idx], dtype=bool)
        onset_idx = onset_idx_all[idx].astype(np.int64)
    return torch.tensor(tracks).float(), torch.tensor(mask).bool(), torch.tensor(onset_idx).long(), idx


@torch.no_grad()
def embed_tracks(model, tracks, mask, onset_idx, mean, std, *, chunk_size=512, device="cpu"):
    tracks = tracks.to(device)
    mask = mask.to(device)
    onset_idx = onset_idx.to(device).float()
    x = apply_standard_scaler(tracks, mean, std)
    x = apply_mask_to_features(add_post_onset_feature(x, onset_idx), mask)
    embs = []
    for start in range(0, x.shape[0], chunk_size):
        emb, _ = model(x[start:start + chunk_size], mask=mask[start:start + chunk_size])
        embs.append(emb.detach())
    return torch.cat(embs, dim=0)


def make_class_context(emb):
    mean = emb.mean(dim=0)
    std = emb.std(dim=0, unbiased=False)
    log_n = torch.log(torch.tensor([float(emb.shape[0])], device=emb.device, dtype=emb.dtype))
    return torch.cat([mean, std, log_n], dim=0).view(1, -1)


def normalised_to_physical(theta_norm, theta_mean, theta_std, theta_names):
    theta = theta_norm * theta_std.cpu() + theta_mean.cpu()
    theta = theta.numpy()
    out = {}
    for j, name in enumerate(theta_names):
        if name == "log_sd1":
            out["sd1"] = np.exp(theta[:, j])
        elif name == "log_sd0":
            out["sd0"] = np.exp(theta[:, j])
        else:
            out[name] = theta[:, j]
    return theta, out


def save_posterior_h5(path, theta_raw, samples, attrs=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("theta_raw", data=theta_raw.astype(np.float32), compression="gzip")
        g = f.create_group("samples")
        for k, v in samples.items():
            g.create_dataset(k, data=np.asarray(v, dtype=np.float32), compression="gzip")
        if "sd0" not in g:
            g.create_dataset("sd0", data=np.ones(theta_raw.shape[0], dtype=np.float32), compression="gzip")
        f.attrs["posterior_schema"] = "samples_named_physical_v1"
        f.attrs["method"] = "set_mdn"
        f.attrs["sd0_fixed"] = 1.0
        f.attrs["diffusion_shared_pre_post"] = True
        f.attrs["pre_drift_model"] = "derived_decay_zero_coupling"
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v


def main():
    import argparse
    p = argparse.ArgumentParser(description="Set-level MDN posterior inference")
    p.add_argument("--data_file", "--data_h5", dest="data_file", required=True)
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--track_class", "--class_id", dest="track_class", type=int, default=None)
    p.add_argument("--track_id", type=int, default=None)
    p.add_argument("--max_tracks", type=int, default=None)
    p.add_argument("--subset_mode", choices=["first", "random"], default="first")
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--embed_chunk_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_h5", type=str, default=None)
    p.add_argument("--out_prefix", type=str, default="mdn_posterior")
    p.add_argument("--train_full_tracks", action="store_true", default=False)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, mask, onset_idx, selected_idx = read_selected_tracks(
        args.data_file,
        track_class=args.track_class,
        track_id=args.track_id,
        max_tracks=args.max_tracks,
        subset_mode=args.subset_mode,
        subset_seed=args.subset_seed,
        train_full_tracks=args.train_full_tracks,
    )
    model, mdn_head, mean, std, theta_mean, theta_std, theta_names, ckpt = load_model_and_mdn(
        args.model_ckpt,
        t_max=tracks.shape[1],
        device=device,
    )
    emb = embed_tracks(model, tracks, mask, onset_idx, mean, std, chunk_size=args.embed_chunk_size, device=device)
    context = make_class_context(emb)
    theta_norm = mdn_head.sample(context, n=args.n_samples).cpu()
    theta_raw, samples = normalised_to_physical(theta_norm, theta_mean, theta_std, theta_names)
    if args.out_h5 is not None:
        out_path = args.out_h5
    else:
        label = "all" if args.track_class is None else str(args.track_class)
        out_path = f"{args.out_prefix}_{label}.h5"
    save_posterior_h5(
        out_path,
        theta_raw,
        samples,
        attrs={
            "track_class": -1 if args.track_class is None else int(args.track_class),
            "track_id": -1 if args.track_id is None else int(args.track_id),
            "n_tracks": int(tracks.shape[0]),
            "n_samples": int(args.n_samples),
            "model_ckpt": args.model_ckpt,
            "data_file": args.data_file,
            "mdn_context_mode": str(ckpt.get("mdn_context_mode", "mean_std_log_n")),
        },
    )
    print(json.dumps({"out_h5": out_path, "n_tracks": int(tracks.shape[0]), "posterior_mean": {k: float(np.mean(v)) for k, v in samples.items() if k in {"gamma0", "gamma1", "sd1"}}}, indent=2))


if __name__ == "__main__":
    main()
