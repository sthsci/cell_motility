from . import config as _config

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from kaljax.filter import forward


def sigmoid_onset_gate(T, onset_idx, ramp_width, dtype):
    steps = jnp.arange(T, dtype=dtype) + 1.0
    onset = jnp.asarray(onset_idx, dtype=dtype)
    return jax.nn.sigmoid((steps - onset) / ramp_width)


def _expm1_over_x(x, thresh=1e-6):
    series = 1.0 + x / 2.0 + (x**2) / 6.0 + (x**3) / 24.0
    return jnp.where(jnp.abs(x) < thresh, series, jnp.expm1(x) / x)


def _w_alpha(alpha, thresh=1e-6):
    r1 = _expm1_over_x(alpha, thresh)
    r2 = _expm1_over_x(2.0 * alpha, thresh)
    series = (1.0 / 3.0) + alpha / 4.0 + (7.0 / 60.0) * (alpha**2)
    exact = (1.0 - 2.0 * r1 + r2) / (alpha**2)
    return jnp.where(jnp.abs(alpha) < thresh, series, exact)


@partial(jit, static_argnames=("eps",))
def build_FQ_class(a, b, sd0, sd1, dt, eps=1e-8):
    alpha = a * dt
    phi = jnp.exp(alpha)
    r1 = _expm1_over_x(alpha)
    r2 = _expm1_over_x(2.0 * alpha)

    beta = b * dt * r1
    q00 = (sd0**2) * dt * r2
    q01 = 0.5 * b * (sd0**2) * (dt**2) * (r1**2)
    q11 = (sd1**2) * dt + (b**2) * (sd0**2) * (dt**3) * _w_alpha(alpha)

    zeros = jnp.zeros_like(phi)
    ones = jnp.ones_like(phi)

    F = jnp.stack(
        [
            jnp.stack([phi, zeros, zeros, zeros], axis=-1),
            jnp.stack([zeros, phi, zeros, zeros], axis=-1),
            jnp.stack([beta, zeros, ones, zeros], axis=-1),
            jnp.stack([zeros, beta, zeros, ones], axis=-1),
        ],
        axis=-2,
    )

    Q = jnp.stack(
        [
            jnp.stack([q00, zeros, q01, zeros], axis=-1),
            jnp.stack([zeros, q00, zeros, q01], axis=-1),
            jnp.stack([q01, zeros, q11, zeros], axis=-1),
            jnp.stack([zeros, q01, zeros, q11], axis=-1),
        ],
        axis=-2,
    )

    return F, Q


@partial(jit, static_argnames=("T",))
def build_FQ_timevarying_for_track(
    gamma0,
    gamma1,
    sd0,
    sd1,
    onset_idx,
    T,
    dt,
    ramp_width,
    lambda_pre_mult,
    pre_coupling=0.0,
):
    dtype = jnp.asarray(gamma0).dtype

    s_t = sigmoid_onset_gate(T, onset_idx, ramp_width, dtype)

    lambda_post = -gamma0
    lambda_pre = lambda_pre_mult * lambda_post

    gamma0_t = -(lambda_pre + (lambda_post - lambda_pre) * s_t)
    gamma1_t = pre_coupling + (gamma1 - pre_coupling) * s_t

    sd0_t = jnp.full((T,), sd0, dtype=dtype)
    sd1_t = jnp.full((T,), sd1, dtype=dtype)

    F_seq, Q_seq = jax.vmap(lambda a, b, s0, s1: build_FQ_class(a, b, s0, s1, dt))(
        gamma0_t, gamma1_t, sd0_t, sd1_t
    )

    return F_seq, Q_seq


@jit
def kalman_loglik_one_track(
    y,
    pad_mask,
    gamma0,
    gamma1,
    sd0,
    sd1,
    onset_idx,
    sigma_obs,
    dt,
    P0,
    ramp_width,
    lambda_pre_mult,
    pre_coupling=0.0,
):
    T = y.shape[0]
    dtype = y.dtype

    F_seq, Q_seq = build_FQ_timevarying_for_track(
        gamma0,
        gamma1,
        sd0,
        sd1,
        onset_idx,
        T,
        dt,
        ramp_width,
        lambda_pre_mult,
        pre_coupling,
    )

    H = jnp.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    R = (jnp.asarray(sigma_obs, dtype=dtype) ** 2) * jnp.eye(2, dtype=dtype)

    m0 = jnp.zeros((4,), dtype=dtype)

    res = forward(
        y,
        F_seq,
        H,
        Q_seq,
        R,
        m0,
        P0,
        mask=~pad_mask,
    )
    return res.ll.sum()


@jit
def kalman_loglik_all_tracks(
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
    pre_coupling=0.0,
):
    return jax.vmap(
        kalman_loglik_one_track,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None),
    )(
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
