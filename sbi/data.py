import torch
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    def __init__(
        self,
        traces: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        theta: torch.Tensor,
    ):
        self.x = traces.float()
        self.m = mask.bool()
        self.labels = labels.long()
        self.theta = theta.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.m[idx], self.labels[idx], self.theta[idx]


class TraceOnsetDataset(Dataset):
    def __init__(
        self,
        traces: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        theta: torch.Tensor,
        onset_idx: torch.Tensor,
    ):
        self.x = traces.float()
        self.m = mask.bool()
        self.labels = labels.long()
        self.theta = theta.float()
        self.onset_idx = onset_idx.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.m[idx],
            self.labels[idx],
            self.theta[idx],
            self.onset_idx[idx],
        )


class TraceSwitchingDataset(Dataset):
    """
    Dataset for the N-state switching extension.

    Parameters
    ----------
    traces       : [N, T, F]   features (xy + per-switch sigmoid gate channels)
    mask         : [N, T]      padding mask
    labels       : [N]         class labels for the classifier head
    theta        : [N, D]      flat per-cell theta vector,
                               D = 3*n_states + (n_states - 1)
    switch_times : [N, K]      per-cell switch times, K = n_states - 1.
                               When n_states == 1 (single regime, no switches),
                               this can be passed as a tensor of shape [N, 0].
    """

    def __init__(
        self,
        traces: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        theta: torch.Tensor,
        switch_times: torch.Tensor,
    ):
        self.x = traces.float()
        self.m = mask.bool()
        self.labels = labels.long()
        self.theta = theta.float()
        self.switch_times = switch_times.float()

        if self.x.shape[0] != self.switch_times.shape[0]:
            raise ValueError(
                f"traces and switch_times must agree on N; got "
                f"{self.x.shape[0]} vs {self.switch_times.shape[0]}"
            )
        if self.switch_times.ndim != 2:
            raise ValueError(
                f"switch_times must be 2D [N, n_states-1]; got "
                f"shape {tuple(self.switch_times.shape)}"
            )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.m[idx],
            self.labels[idx],
            self.theta[idx],
            self.switch_times[idx],
        )


def _compute_bin_edges_1d(
    x: torch.Tensor,
    n_bins: int,
    strategy: str = "quantile",
    eps: float = 1e-6,
) -> torch.Tensor:
    x = x.float().flatten()

    if strategy == "quantile":
        q = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device)
        edges = torch.quantile(x, q)

        edges[0] = edges[0] - eps
        edges[-1] = edges[-1] + eps

        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps

        return edges

    if strategy == "uniform":
        xmin = x.min()
        xmax = x.max()
        if xmax <= xmin:
            xmax = xmin + eps
        edges = torch.linspace(xmin, xmax, n_bins + 1, device=x.device)
        edges[0] = edges[0] - eps
        edges[-1] = edges[-1] + eps
        return edges

    raise ValueError("strategy must be 'quantile' or 'uniform'")


def relabel_by_parameter_bins(
    theta: torch.Tensor,
    n_bins: int | list[int] | tuple[int, ...] = 3,
    strategy: str = "quantile",
) -> tuple[torch.Tensor, dict]:
    """
    Recompute categorical labels from continuous parameter values.

    Parameters
    ----------
    theta : torch.Tensor
        Shape [N, D], e.g. columns [gamma0, gamma1, sd1].
    n_bins : int or sequence of ints
        Number of bins per parameter dimension. If int, the same number is used
        for every dimension.
    strategy : str
        'quantile' or 'uniform'.

    Returns
    -------
    labels : torch.Tensor
        Shape [N], contiguous integer labels in [0, K_used).
    info : dict
        Metadata containing edges, per-dimension bin assignments, etc.
    """
    theta = theta.float()
    if theta.ndim != 2:
        raise ValueError("theta must have shape [N, D]")

    N, D = theta.shape

    if isinstance(n_bins, int):
        n_bins_per_dim = [n_bins] * D
    else:
        n_bins_per_dim = list(n_bins)
        if len(n_bins_per_dim) != D:
            raise ValueError(
                "If n_bins is a sequence, it must have length equal to theta.shape[1]"
            )

    edges_per_dim = []
    bin_idx_per_dim = []

    for d in range(D):
        edges_d = _compute_bin_edges_1d(
            theta[:, d], n_bins_per_dim[d], strategy=strategy
        )
        idx_d = torch.bucketize(theta[:, d], edges_d[1:-1], right=False)
        edges_per_dim.append(edges_d)
        bin_idx_per_dim.append(idx_d)

    bin_idx = torch.stack(bin_idx_per_dim, dim=1)

    multipliers = []
    m = 1
    for b in reversed(n_bins_per_dim):
        multipliers.append(m)
        m *= b
    multipliers = torch.tensor(
        list(reversed(multipliers)), device=theta.device, dtype=torch.long
    )

    raw_labels = (bin_idx.long() * multipliers[None, :]).sum(dim=1)

    unique_raw, labels = torch.unique(raw_labels, sorted=True, return_inverse=True)

    info = {
        "edges_per_dim": edges_per_dim,
        "bin_idx_per_dim": bin_idx,
        "raw_labels": raw_labels,
        "unique_raw_labels": unique_raw,
        "n_bins_per_dim": n_bins_per_dim,
        "n_used_labels": int(unique_raw.numel()),
    }
    return labels.long(), info
