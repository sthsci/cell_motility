from . import config as _config

import h5py
import jax.numpy as jnp
import numpy as np


def select_state_param(x, state_index=-1):
    x = np.asarray(x)
    if x.ndim == 0:
        return x
    if x.ndim == 1:
        return x
    return x[..., state_index]

def _decode_h5_strings(x):
    arr = np.asarray(x)
    if arr.dtype.kind in {"S", "O"}:
        return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr])
    return arr.astype(str)

def _choose_h5_group(f, h5_group=None, channel="all", real_data_defaults=False):
    if h5_group is not None:
        return f[h5_group]

    if real_data_defaults:
        if "onset_motifs" in f:
            if channel != "all" and channel in f["onset_motifs"]:
                return f[f"onset_motifs/{channel}"]
            if "all" in f["onset_motifs"]:
                return f["onset_motifs/all"]
        if channel != "all" and channel in f and "tracks" in f[channel]:
            return f[channel]

    if "tracks" in f and "mask" in f and "onset_idx" in f:
        return f
    if "onset_motifs" in f and "all" in f["onset_motifs"]:
        return f["onset_motifs/all"]
    if "nk" in f and "tracks" in f["nk"]:
        return f["nk"]

    raise KeyError(
        "Could not find a usable dataset. Expected top-level tracks/mask/onset_idx "
        "or an exported group such as onset_motifs/all, onset_motifs/nk, nk, or tumour."
    )

def _load_dataset_one_file(
    path,
    *,
    h5_group=None,
    channel="all",
    split_channel_labels=False,
    real_data_defaults=False,
    file_label=None,
):
    with h5py.File(path, "r") as f:
        g = _choose_h5_group(
            f,
            h5_group=h5_group,
            channel=channel,
            real_data_defaults=real_data_defaults,
        )

        tracks_np = np.asarray(g["tracks"][:], dtype=np.float32)
        pad_mask_np = np.asarray(g["mask"][:], dtype=bool)
        onset_idx_np = np.asarray(g["onset_idx"][:], dtype=np.int32)

        labels_np = None
        if file_label is not None:
            labels_np = np.full(tracks_np.shape[0], int(file_label), dtype=np.int32)
        elif "labels" in g:
            labels_np = np.asarray(g["labels"][:], dtype=np.int32)
        elif "labels" in f and g is f:
            labels_np = np.asarray(f["labels"][:], dtype=np.int32)
        elif split_channel_labels and "channel" in g:
            channels = _decode_h5_strings(g["channel"][:])
            unique = {name: i for i, name in enumerate(sorted(set(channels)))}
            labels_np = np.asarray([unique[c] for c in channels], dtype=np.int32)

        if channel != "all":
            if "channel" in g:
                channels = _decode_h5_strings(g["channel"][:])
                keep = channels == channel
                tracks_np = tracks_np[keep]
                pad_mask_np = pad_mask_np[keep]
                onset_idx_np = onset_idx_np[keep]
                if labels_np is not None:
                    labels_np = labels_np[keep]
            else:
                group_name = getattr(g, "name", "")
                if not group_name.endswith("/" + channel) and group_name.split("/")[-1] != channel:
                    raise ValueError(
                        f"channel={channel!r} was requested, but group {group_name!r} has no channel dataset to filter."
                    )

        if tracks_np.shape[0] == 0:
            raise ValueError(
                f"No tracks remain after applying path={path!r}, "
                f"h5_group={getattr(g, 'name', h5_group)!r}, channel={channel!r}."
            )

    return tracks_np, pad_mask_np, onset_idx_np, labels_np

def load_dataset(
    path="ou_tracks.h5",
    h5_group=None,
    channel="all",
    split_channel_labels=False,
    data_paths=None,
    real_data_defaults=False,
):
    paths = list(data_paths) if data_paths is not None else [path]
    if len(paths) == 0:
        raise ValueError("No input data paths were provided.")

    tracks_all = []
    masks_all = []
    onset_all = []
    labels_all = []
    any_labels = False

    for file_i, pth in enumerate(paths):
        file_label = file_i if real_data_defaults and len(paths) > 1 else None
        tracks_np, pad_mask_np, onset_idx_np, labels_np = _load_dataset_one_file(
            pth,
            h5_group=h5_group,
            channel=channel,
            split_channel_labels=split_channel_labels,
            real_data_defaults=real_data_defaults,
            file_label=file_label,
        )
        tracks_all.append(tracks_np)
        masks_all.append(pad_mask_np)
        onset_all.append(onset_idx_np)
        if labels_np is not None:
            any_labels = True
            labels_all.append(labels_np)
        else:
            labels_all.append(np.zeros(tracks_np.shape[0], dtype=np.int32))

    tracks_np = np.concatenate(tracks_all, axis=0)
    pad_mask_np = np.concatenate(masks_all, axis=0)
    onset_idx_np = np.concatenate(onset_all, axis=0)

    if real_data_defaults and len(paths) == 1 and not any_labels:
        labels_np = np.zeros(tracks_np.shape[0], dtype=np.int32)
    elif any_labels or (real_data_defaults and len(paths) > 1):
        labels_np = np.concatenate(labels_all, axis=0)
    else:
        labels_np = None

    tracks = jnp.asarray(tracks_np, dtype=jnp.float32)
    pad_mask = jnp.asarray(pad_mask_np, dtype=bool)
    labels = None if labels_np is None else jnp.asarray(labels_np, dtype=jnp.int32)
    onset_idx = jnp.asarray(onset_idx_np, dtype=jnp.int32)
    return tracks, pad_mask, labels, onset_idx
