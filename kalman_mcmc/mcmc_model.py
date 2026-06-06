from . import config as _config

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from .kalman_analytical import kalman_loglik_all_tracks


def ou_pol_model(
    tracks,
    pad_mask,
    labels,
    onset_idx,
    K,
    sigma_obs,
    dt,
    *,
    u=None,
    scale=1.0,
    P0_pol_scale=1.0,
    P0_pos_scale=1.0,
    ramp_width=2.0,
    lambda_pre_mult=4.0,
    pre_coupling=0.0,
    prior_log_rate_loc=-0.5,
    prior_log_rate_scale=0.75,
    prior_b_loc=0.0,
    prior_b_scale=0.75,
    prior_log_sd1_loc=0.0,
    prior_log_sd1_scale=0.4,
):
    num_cells, T, _ = tracks.shape
    dtype = tracks.dtype

    if K == 1:
        log_rate = numpyro.sample("log_rate", dist.Normal(prior_log_rate_loc, prior_log_rate_scale))
        rate_scalar = numpyro.deterministic("rate", jnp.exp(log_rate))
        gamma0_scalar = numpyro.deterministic("gamma0", -rate_scalar)
        gamma0 = jnp.expand_dims(gamma0_scalar, axis=0)

        log_gamma1 = numpyro.sample("log_gamma1", dist.Normal(prior_b_loc, prior_b_scale))
        gamma1_scalar = numpyro.deterministic("gamma1", jnp.exp(log_gamma1))
        gamma1 = jnp.expand_dims(gamma1_scalar, axis=0)

        sd0 = jnp.asarray([1.0 / scale], dtype=dtype)
        numpyro.deterministic("sd0", sd0)

        log_sd1_phys = numpyro.sample("log_sd1_phys", dist.Normal(prior_log_sd1_loc, prior_log_sd1_scale))
        raw_sd1 = numpyro.deterministic("raw_sd1", log_sd1_phys - jnp.log(scale))
        sd1_scalar = numpyro.deterministic("sd1", jnp.exp(raw_sd1))
        sd1 = jnp.expand_dims(sd1_scalar, axis=0)

    else:
        log_tau_loc = numpyro.sample("log_tau_loc", dist.Normal(0.0, 2.0))
        log_tau_scale = numpyro.sample("log_tau_scale", dist.HalfNormal(0.5))

        raw_gamma1_loc = numpyro.sample("raw_gamma1_loc", dist.Normal(0.0, 0.8))
        raw_gamma1_scale = numpyro.sample("raw_gamma1_scale", dist.HalfNormal(0.5))

        raw_sd0_loc = numpyro.sample("raw_sd0_loc", dist.Normal(0.0, 0.25))
        raw_sd0_scale = numpyro.sample("raw_sd0_scale", dist.HalfNormal(0.2))

        raw_sd1_loc = numpyro.sample("raw_sd1_loc", dist.Normal(0.0, 0.25))
        raw_sd1_scale = numpyro.sample("raw_sd1_scale", dist.HalfNormal(0.2))

        with numpyro.plate("classes", K):
            z_tau = numpyro.sample("z_tau", dist.Normal(0.0, 1.0))
            log_tau_cls = log_tau_loc + log_tau_scale * z_tau
            numpyro.deterministic("log_tau", log_tau_cls)

            tau = jnp.exp(log_tau_cls)
            numpyro.deterministic("tau", tau)

            gamma0 = -1.0 / tau
            numpyro.deterministic("gamma0", gamma0)

            raw_gamma1 = numpyro.sample(
                "raw_gamma1", dist.Normal(raw_gamma1_loc, raw_gamma1_scale)
            )
            gamma1 = jax.nn.softplus(raw_gamma1)
            numpyro.deterministic("gamma1", gamma1)

            raw_sd0 = numpyro.sample("raw_sd0", dist.Normal(raw_sd0_loc, raw_sd0_scale))
            sd0 = jnp.exp(raw_sd0)
            numpyro.deterministic("sd0", sd0)

            raw_sd1 = numpyro.sample("raw_sd1", dist.Normal(raw_sd1_loc, raw_sd1_scale))
            sd1 = jnp.exp(raw_sd1)
            numpyro.deterministic("sd1", sd1)

    gamma0_cell = gamma0[labels]
    gamma1_cell = gamma1[labels]
    sd0_cell = sd0[labels]
    sd1_cell = sd1[labels]

    P0_scale_pol = jnp.asarray([P0_pol_scale, P0_pol_scale], dtype=dtype)
    P0_scale_pos = jnp.asarray([P0_pos_scale, P0_pos_scale], dtype=dtype)
    P0_scale = jnp.concatenate([P0_scale_pol, P0_scale_pos])
    P0 = jnp.diag(P0_scale**2).astype(dtype)

    ll_seq = kalman_loglik_all_tracks(
        tracks,
        pad_mask,
        gamma0_cell,
        gamma1_cell,
        sd0_cell,
        sd1_cell,
        onset_idx,
        sigma_obs,
        dt,
        P0,
        ramp_width,
        lambda_pre_mult,
        pre_coupling,
    )

    numpyro.factor("kalman_ll", ll_seq.sum())

def map_warmstart(
    key,
    tracks,
    pad_mask,
    labels,
    onset_idx,
    K,
    sigma_obs,
    dt,
    scale,
    P0_pol_scale,
    P0_pos_scale,
    ramp_width,
    lambda_pre_mult,
    pre_coupling=0.0,
    prior_log_rate_loc=-0.5,
    prior_log_rate_scale=0.75,
    prior_b_loc=0.0,
    prior_b_scale=0.75,
    prior_log_sd1_loc=0.0,
    prior_log_sd1_scale=0.4,
):
    guide = AutoDelta(ou_pol_model)
    svi = SVI(ou_pol_model, guide, optax.adam(0.1), Trace_ELBO())
    res = svi.run(
        key,
        1000,
        tracks,
        pad_mask,
        labels,
        onset_idx,
        K,
        sigma_obs,
        dt,
        scale=scale,
        P0_pol_scale=P0_pol_scale,
        P0_pos_scale=P0_pos_scale,
        ramp_width=ramp_width,
        lambda_pre_mult=lambda_pre_mult,
        pre_coupling=pre_coupling,
        prior_log_rate_loc=prior_log_rate_loc,
        prior_log_rate_scale=prior_log_rate_scale,
        prior_b_loc=prior_b_loc,
        prior_b_scale=prior_b_scale,
        prior_log_sd1_loc=prior_log_sd1_loc,
        prior_log_sd1_scale=prior_log_sd1_scale,
    )
    return guide.median(res.params)
