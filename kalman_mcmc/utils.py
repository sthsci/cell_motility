from . import config as _config

import numpy as np


def descale_samples(samples, s):
    s = float(s)
    out = {}

    multiply_by_s = {
        "sd0",
        "sd1",
        # "m0_loc_pol",
        # "m0_scale_pol",
        # "m0_loc_pos",
        # "m0_scale_pos",
        "P0_scale_pol",
        "P0_scale_pos",
        # "m0_pol",
    }

    for name, arr in samples.items():
        arr_np = np.asarray(arr)

        if name in multiply_by_s:
            out[name] = arr_np * s
        elif name == "raw_sd0":
            out[name] = arr_np
            out["sd0_from_raw"] = np.exp(arr_np) * s
        elif name == "raw_sd1":
            out[name] = arr_np
            out["sd1_from_raw"] = np.exp(arr_np) * s
        else:
            out[name] = arr_np

    return out
