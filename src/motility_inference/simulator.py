"""Fast, deliberately reduced killer–target migration model.

The supplied :mod:`mantishrimp` ABM is preserved. This model retains its OU
motility, periodic geometry, Hookean overlap repulsion and optional stochastic
adhesion. Killing, exhaustion, event tables, target–target and killer–killer
forces are excluded: the outcome here is migration, not cytotoxicity. This is
therefore a reduced model, not a drop-in acceleration of the complete ABM.

Local OU coefficients are frozen at the beginning of each integration step.
The *joint* velocity/displacement transition is exact for those coefficients;
spatially varying coefficients and mechanics still require a substep study.
Both engines use identical externally generated random numbers, making their
numerical equivalence testable. The compiled kernel allocates no Python objects.

``speed`` means the stationary standard deviation of each active-velocity
coordinate, not the mean scalar speed. ``jitter`` is the position Brownian
amplitude (length / sqrt(time)). All units are consistent user-defined units;
the demonstration convention is micrometres and minutes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numba import njit
except ImportError:  # A reference engine remains available without Numba.
    def njit(*args, **kwargs):
        def decorator(fn):
            fn.py_func = fn
            return fn
        return decorator


PARAMETER_NAMES = (
    "log_speed", "log_tau", "log_jitter",
    "delta_speed", "delta_tau", "delta_jitter",
)


@dataclass(frozen=True)
class SimConfig:
    """Observation design and nuisance mechanisms, independent of ``theta``."""

    n_killers: int = 12
    n_targets: int = 9
    n_frames: int = 201
    frame_dt: float = 0.25
    substeps: int = 1
    box_size: float = 180.0
    contact_distance: float = 12.0
    interaction_length: float = 10.0
    localization_sd: float = 0.15
    heterogeneity_sd: float = 0.25
    target_speed: float = 0.0
    target_tau: float = 2.0
    repulsion: float = 0.0
    binding_rate: float = 0.0
    unbinding_rate: float = 0.5
    adhesion: float = 0.0
    capture_radius_factor: float = 1.5
    spontaneous_sd: float = 0.0
    spontaneous_tau: float = 5.0

    def validate(self) -> None:
        for name in ("n_killers", "n_targets", "n_frames", "substeps"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < (2 if name == "n_frames" else 1):
                raise ValueError(f"{name} is too small")
        for name in ("frame_dt", "box_size", "contact_distance", "interaction_length", "target_tau", "spontaneous_tau"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("localization_sd", "heterogeneity_sd", "target_speed", "repulsion", "binding_rate", "unbinding_rate", "adhesion", "spontaneous_sd"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.capture_radius_factor) or self.capture_radius_factor < 1:
            raise ValueError("capture_radius_factor must be at least one")
        if self.contact_distance >= self.box_size / 2:
            raise ValueError("contact_distance must be less than half the box size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def proximity(
    positions: np.ndarray,
    targets: np.ndarray,
    box_size: float,
    contact_distance: float = 12.0,
    interaction_length: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest-target ``(gate, gap, bearing, index)`` from supplied data.

    Input shapes are ``[killer, frame, 2]`` and ``[target, frame, 2]``.
    Distances use the periodic minimum image even when killer tracks are
    unwrapped. Gap is signed centroid distance minus contact distance. At
    exact co-location the direction is zero; ties select the first target.
    NaN killer observations propagate to gap/gate/direction, with index -1.
    At least one target per frame must have finite coordinates.
    """
    x = np.asarray(positions, dtype=np.float64)
    q = np.asarray(targets, dtype=np.float64)
    if x.ndim != 3 or q.ndim != 3 or x.shape[-1] != 2 or q.shape[-1] != 2 or x.shape[1] != q.shape[1]:
        raise ValueError("positions/targets must have matching [cell, frame, 2] shapes")
    if q.shape[0] == 0 or x.shape[0] == 0:
        raise ValueError("at least one killer and target is required")
    if not np.isfinite(box_size) or box_size <= 0 or not np.isfinite(interaction_length) or interaction_length <= 0:
        raise ValueError("box_size and interaction_length must be finite and positive")
    if not np.isfinite(contact_distance) or contact_distance < 0:
        raise ValueError("contact_distance must be finite and nonnegative")
    valid_target = np.isfinite(q).all(axis=-1)
    if not valid_target.any(axis=0).all():
        raise ValueError("every frame needs at least one finite target position")
    displacement = q[None, :, :, :] - x[:, None, :, :]
    displacement -= box_size * np.rint(displacement / box_size)
    squared = np.sum(displacement * displacement, axis=-1)
    squared = np.where(np.isfinite(squared), squared, np.inf)
    nearest = np.argmin(squared, axis=1)
    gathered = np.take_along_axis(displacement, nearest[:, None, :, None], axis=1)[:, 0]
    distance = np.sqrt(np.take_along_axis(squared, nearest[:, None, :], axis=1)[:, 0])
    unit = np.divide(gathered, distance[..., None], out=np.zeros_like(gathered), where=distance[..., None] > 0)
    gap = distance - contact_distance
    gate = np.exp(-np.maximum(gap, 0.0) / interaction_length)
    missing = ~np.isfinite(x).all(axis=-1)
    gate[missing] = np.nan
    gap[missing] = np.nan
    unit[missing] = np.nan
    nearest[missing] = -1
    return gate, gap, unit, nearest


@njit(cache=True)
def _joint_ou(tau: float, speed: float, jitter: float, dt: float):
    """phi, integrated drift factor, and Cholesky of (velocity, position)."""
    a = dt / tau
    omphi = -np.expm1(-a)
    phi = 1.0 - omphi
    qvv = speed * speed * (-np.expm1(-2.0 * a))
    qvx = speed * speed * tau * omphi * omphi
    if a < 1e-3:
        h = a ** 3 * (1.0 / 3.0 + a * (-0.25 + a * (7.0 / 60.0 - a / 24.0)))
    else:
        h = a - 2.0 * omphi - 0.5 * np.expm1(-2.0 * a)
    qxx = 2.0 * speed * speed * tau * tau * h + jitter * jitter * dt
    lvv = np.sqrt(max(qvv, 0.0))
    lxv = qvx / lvv if lvv > 0.0 else 0.0
    lxx = np.sqrt(max(qxx - lxv * lxv, 0.0))
    return phi, tau * omphi, lvv, lxv, lxx


@njit(cache=True)
def _kernel(theta, offsets, initial_x, initial_targets, initial_v_noise, noise,
            target_noise, initial_target_v_noise, bond_uniforms, spontaneous_noise,
            initial_spontaneous, n_frames, substeps, frame_dt, box_size,
            contact_distance, interaction_length, target_speed, target_tau,
            repulsion, binding_rate, unbinding_rate, adhesion, capture_factor,
            spontaneous_sd, spontaneous_tau):
    nk = initial_x.shape[0]
    nt = initial_targets.shape[0]
    dt = frame_dt / substeps
    x = initial_x.copy()
    targets = initial_targets.copy()
    v = np.zeros((nk, 2))
    target_v = target_speed * initial_target_v_noise.copy()
    bound = np.zeros((nk, nt), dtype=np.bool_)
    spontaneous = spontaneous_sd * initial_spontaneous.copy()
    positions_out = np.empty((nk, n_frames, 2))
    velocity_out = np.empty((nk, n_frames, 2))
    target_out = np.empty((nt, n_frames, 2))
    gate_out = np.empty((nk, n_frames))
    spontaneous_out = np.empty((nk, n_frames))
    bound_out = np.zeros((nk, n_frames), dtype=np.int64)
    pbind = -np.expm1(-binding_rate * dt)
    punbind = -np.expm1(-unbinding_rate * dt)
    spont_phi = np.exp(-dt / spontaneous_tau)
    spont_sd = spontaneous_sd * np.sqrt(-np.expm1(-2.0 * dt / spontaneous_tau))
    target_phi, target_int, target_lvv, target_lxv, target_lxx = _joint_ou(target_tau, target_speed, 0.0, dt)
    total_steps = (n_frames - 1) * substeps

    for step in range(total_steps + 1):
        frame = step // substeps
        for i in range(nk):
            minimum = 1e300
            force_x = 0.0
            force_y = 0.0
            nbound = 0
            for j in range(nt):
                dx = targets[j, 0] - x[i, 0]
                dy = targets[j, 1] - x[i, 1]
                dx -= box_size * np.rint(dx / box_size)
                dy -= box_size * np.rint(dy / box_size)
                distance = np.sqrt(dx * dx + dy * dy)
                minimum = min(minimum, distance)
                if step < total_steps and (binding_rate > 0.0 or adhesion > 0.0):
                    if bound[i, j]:
                        if distance > capture_factor * contact_distance or bond_uniforms[step, i, j] < punbind:
                            bound[i, j] = False
                    elif distance <= contact_distance and bond_uniforms[step, i, j] < pbind:
                        bound[i, j] = True
                if bound[i, j]:
                    nbound += 1
                if distance > 1e-12:
                    magnitude = -repulsion * max(contact_distance - distance, 0.0)
                    if bound[i, j] and distance < capture_factor * contact_distance:
                        magnitude += adhesion
                    force_x += magnitude * dx / distance
                    force_y += magnitude * dy / distance
            g = np.exp(-max(minimum - contact_distance, 0.0) / interaction_length)
            speed = np.exp(theta[0] + offsets[i] + theta[3] * g + spontaneous[i])
            tau = np.exp(theta[1] + theta[4] * g)
            jitter = np.exp(theta[2] + theta[5] * g)
            if step == 0:
                v[i, 0] = speed * initial_v_noise[i, 0]
                v[i, 1] = speed * initial_v_noise[i, 1]
            if step % substeps == 0:
                positions_out[i, frame, 0] = x[i, 0]
                positions_out[i, frame, 1] = x[i, 1]
                velocity_out[i, frame, 0] = v[i, 0]
                velocity_out[i, frame, 1] = v[i, 1]
                gate_out[i, frame] = g
                spontaneous_out[i, frame] = spontaneous[i]
                bound_out[i, frame] = nbound
            if step < total_steps:
                phi, integ, lvv, lxv, lxx = _joint_ou(tau, speed, jitter, dt)
                for axis in range(2):
                    z_v = noise[step, i, axis, 0]
                    z_x = noise[step, i, axis, 1]
                    mechanical = force_x if axis == 0 else force_y
                    x[i, axis] += integ * v[i, axis] + mechanical * dt + lxv * z_v + lxx * z_x
                    v[i, axis] = phi * v[i, axis] + lvv * z_v
                spontaneous[i] = spont_phi * spontaneous[i] + spont_sd * spontaneous_noise[step, i]
        if step % substeps == 0:
            for j in range(nt):
                for axis in range(2):
                    target_out[j, frame, axis] = targets[j, axis] % box_size
        if step < total_steps and target_speed > 0.0:
            for j in range(nt):
                for axis in range(2):
                    z_v = target_noise[step, j, axis, 0]
                    z_x = target_noise[step, j, axis, 1]
                    targets[j, axis] += target_int * target_v[j, axis] + target_lxv * z_v + target_lxx * z_x
                    targets[j, axis] %= box_size
                    target_v[j, axis] = target_phi * target_v[j, axis] + target_lvv * z_v
    return positions_out, velocity_out, target_out, gate_out, spontaneous_out, bound_out


def _initial_positions(rng, config):
    # A jittered target lattice gives reproducible spatial coverage. Killers
    # are sampled independently of theta and future movement/encounters.
    side = int(np.ceil(np.sqrt(config.n_targets)))
    grid = (np.arange(side) + 0.5) * config.box_size / side
    targets = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
    targets = targets[rng.permutation(len(targets))[:config.n_targets]]
    targets += rng.uniform(-0.15, 0.15, size=targets.shape) * config.box_size / side
    targets %= config.box_size
    killers = rng.uniform(0, config.box_size, size=(config.n_killers, 2))
    # Avoid initial physical overlaps; no future outcome is used for selection.
    for i in range(config.n_killers):
        for attempt in range(10000):
            delta = targets - killers[i]
            delta -= config.box_size * np.rint(delta / config.box_size)
            if np.min(np.sum(delta * delta, axis=1)) >= config.contact_distance ** 2:
                break
            killers[i] = rng.uniform(0, config.box_size, size=2)
        else:
            raise ValueError("target occupancy is too dense to place killers without overlap")
    return killers, targets


def simulate(theta, config: SimConfig | dict | None = None, seed: int = 0,
             engine: str = "numba", *, initial_positions=None, initial_targets=None,
             cell_offsets=None) -> dict[str, Any]:
    """Simulate one independent movie; returned killer tracks are unwrapped.

    Inference must use ``positions`` and ``targets``. All ``true_*`` fields,
    ``cell_offsets``, and ``theta`` are validation labels, never input features.
    The reference engine is ``engine='numpy'`` (a Python/NumPy loop). Initial
    arrays, when supplied, are copied and never modified. Cell offsets can be
    supplied for conditional posterior-predictive simulation.
    """
    config = SimConfig() if config is None else (SimConfig(**config) if isinstance(config, dict) else config)
    config.validate()
    theta = np.array(theta, dtype=np.float64, copy=True)
    if theta.shape != (6,) or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite vector of length six")
    if np.max(np.abs(theta)) > 15:
        raise ValueError("extreme log parameters are unsupported; rescale units")
    if engine not in {"numba", "numpy", "python"}:
        raise ValueError("engine must be 'numba' or 'numpy'")
    rng = np.random.default_rng(seed)
    if (initial_positions is None) != (initial_targets is None):
        raise ValueError("provide both initial_positions and initial_targets, or neither")
    if initial_positions is None:
        initial_positions, initial_targets = _initial_positions(rng, config)
    else:
        initial_positions = np.array(initial_positions, dtype=float, copy=True)
        initial_targets = np.array(initial_targets, dtype=float, copy=True)
        if initial_positions.shape != (config.n_killers, 2) or initial_targets.shape != (config.n_targets, 2):
            raise ValueError("initial position shapes do not match configuration")
        if not np.isfinite(initial_positions).all() or not np.isfinite(initial_targets).all():
            raise ValueError("initial positions must be finite")
    if cell_offsets is None:
        offsets = rng.normal(0, config.heterogeneity_sd, config.n_killers)
    else:
        offsets = np.array(cell_offsets, dtype=float, copy=True)
        if offsets.shape != (config.n_killers,) or not np.isfinite(offsets).all():
            raise ValueError("cell_offsets must be finite with one value per killer")
    steps = (config.n_frames - 1) * config.substeps
    initial_v_noise = rng.normal(size=(config.n_killers, 2))
    noise = rng.normal(size=(steps, config.n_killers, 2, 2))
    target_noise = rng.normal(size=(steps, config.n_targets, 2, 2)) if config.target_speed else np.empty((0, 0, 0, 0))
    target_v_noise = rng.normal(size=(config.n_targets, 2))
    bond_uniforms = rng.random((steps, config.n_killers, config.n_targets)) if config.binding_rate or config.adhesion else np.empty((0, 0, 0))
    spontaneous_noise = rng.normal(size=(steps, config.n_killers))
    initial_spontaneous = rng.normal(size=config.n_killers)
    kernel = _kernel if engine == "numba" else _kernel.py_func
    result = kernel(theta, offsets, initial_positions, initial_targets, initial_v_noise,
                    noise, target_noise, target_v_noise, bond_uniforms, spontaneous_noise,
                    initial_spontaneous, config.n_frames, config.substeps,
                    config.frame_dt, config.box_size, config.contact_distance,
                    config.interaction_length, config.target_speed, config.target_tau,
                    config.repulsion, config.binding_rate, config.unbinding_rate,
                    config.adhesion, config.capture_radius_factor, config.spontaneous_sd,
                    config.spontaneous_tau)
    true_x, true_v, targets, gate, spontaneous, bound = result
    measured = true_x + rng.normal(0, config.localization_sd, size=true_x.shape)
    return {
        "positions": measured,
        "targets": targets,
        "times": np.arange(config.n_frames, dtype=float) * config.frame_dt,
        "true_positions": true_x,
        "true_velocity": true_v,
        "true_gate": gate,
        "true_spontaneous": spontaneous,
        "true_bound_count": bound,
        "cell_offsets": offsets,
        "theta": theta,
        "config": config.to_dict(),
        "seed": int(seed),
        "schema_version": 1,
    }


def save_dataset(data: dict[str, Any], path: str | Path, metadata: dict | None = None) -> Path:
    """Save arrays plus JSON metadata to a portable NPZ without pickle objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    attributes = {key: value for key, value in data.items() if not isinstance(value, np.ndarray)}
    if metadata is not None:
        attributes["metadata"] = metadata
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=np.asarray(json.dumps(attributes)))
    return path


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset produced by :func:`save_dataset` without enabling pickle."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        data.update(json.loads(str(archive["metadata_json"])))
    if "positions" not in data or "targets" not in data or "times" not in data:
        raise ValueError("not a motility dataset: positions, targets and times are required")
    return data


def benchmark_simulator(theta, config: SimConfig | None = None, *, repeats: int = 20,
                        seed: int = 0) -> dict[str, float | int]:
    """Measure actual simulation workload without hardware-specific assertions.

    The first call includes JIT/cache loading if applicable. Warm measurements
    include RNG generation, initialization, simulation and observation noise;
    the reference engine uses the same model and draws. The full original ABM
    is intentionally not included in this ratio because it has more biology.
    """
    from time import perf_counter

    if repeats < 1:
        raise ValueError("repeats must be positive")
    config = SimConfig() if config is None else config
    start = perf_counter()
    simulate(theta, config, seed=seed, engine="numba")
    first = perf_counter() - start
    warm = []
    for i in range(repeats):
        start = perf_counter()
        simulate(theta, config, seed=seed + i + 1, engine="numba")
        warm.append(perf_counter() - start)
    reference = []
    for i in range(min(3, repeats)):
        start = perf_counter()
        simulate(theta, config, seed=seed + i + 1, engine="numpy")
        reference.append(perf_counter() - start)
    median = float(np.median(warm))
    ref_median = float(np.median(reference))
    return {
        "first_call_seconds": float(first),
        "warm_median_seconds": median,
        "warm_p95_seconds": float(np.quantile(warm, 0.95)),
        "reference_median_seconds": ref_median,
        "same_model_speedup": ref_median / median,
        "simulations_per_second": 1.0 / median,
        "n_killers": config.n_killers,
        "n_targets": config.n_targets,
        "n_frames": config.n_frames,
        "substeps": config.substeps,
        "warm_repeats": repeats,
        "reference_repeats": len(reference),
    }


def from_mantishrimp_config(config, *, frame_dt: float | None = None,
                           substeps: int = 1, interaction_length: float | None = None,
                           localization_sd: float = 0.15,
                           heterogeneity_sd: float = 0.0) -> tuple[np.ndarray, SimConfig]:
    """Map the supplied ABM's motility/mechanics into the reduced model.

    For ``dp=-decay*p*dt+noise*dW`` and ``dx=coupling*p*dt+...``,
    the equivalent stationary active-velocity SD is
    ``coupling*noise/sqrt(2*decay)`` and persistence time is ``1/decay``.
    The optional source proximity effects are mapped to the same log ratios.
    A square periodic domain is required;
    translating its origin to zero has no effect on periodic relative geometry.

    This maps parameters, *not trajectory laws in all details*. The original
    starts polarity at zero, whereas this simulator initializes stationary
    velocity. The original uses Euler OU updates, adaptive timesteps, reciprocal
    mechanical forces, other pair classes, damage/death and exhaustion. Here
    targets follow independent OU paths, mechanical forces act on killers only,
    and those extra mechanisms are omitted. Binding is allowed at signed gap
    <=0, without the reference model's extra contact-tolerance shell.

    Observation interval defaults to ``max_dt``; intervals are shortened, when
    necessary, to end exactly at the requested duration. ``record_every`` in
    the original counts adaptive steps and is deliberately not equated to a
    physical interval. The caller must check mechanical timestep convergence
    and choose ``substeps`` accordingly; this function does not certify it.
    """
    config.validate()
    if config.domain.boundary != "periodic" or not np.isclose(config.domain.width, config.domain.height):
        raise ValueError("the reduced simulator requires a square periodic domain")
    motility = config.motility
    if motility.killer_polarity_decay <= 0 or motility.target_polarity_decay <= 0:
        raise ValueError("positive polarity decay is required for stationary OU mapping")
    speed = motility.killer_speed * motility.killer_polarity_noise / np.sqrt(2 * motility.killer_polarity_decay)
    jitter = motility.killer_translation_noise
    if speed <= 0 or jitter <= 0:
        raise ValueError("positive killer velocity scale and translation noise are required by log parameterization")
    if motility.target_translation_noise != 0:
        # The reduced target process has active OU motion only. Return metadata
        # through a warning rather than silently claim full parameter fidelity.
        import warnings
        warnings.warn("target translation noise is omitted by the reduced target process", UserWarning, stacklevel=2)
    requested_dt = config.max_dt if frame_dt is None else frame_dt
    if not np.isfinite(requested_dt) or requested_dt <= 0:
        raise ValueError("frame_dt must be finite and positive")
    intervals = max(1, int(np.ceil(config.duration / requested_dt - 1e-12)))
    theta = np.array([np.log(speed), np.log(1 / motility.killer_polarity_decay), np.log(jitter),
                      motility.killer_delta_speed, motility.killer_delta_tau, motility.killer_delta_jitter])
    reduced = SimConfig(
        n_killers=config.n_killers,
        n_targets=config.n_targets,
        n_frames=intervals + 1,
        frame_dt=config.duration / intervals,
        substeps=substeps,
        box_size=config.domain.width,
        contact_distance=config.killer_radius + config.target_radius,
        interaction_length=motility.target_interaction_length if interaction_length is None else interaction_length,
        localization_sd=localization_sd,
        heterogeneity_sd=heterogeneity_sd,
        target_speed=motility.target_speed * motility.target_polarity_noise / np.sqrt(2 * motility.target_polarity_decay),
        target_tau=1 / motility.target_polarity_decay,
        repulsion=config.mechanics.repulsion_kt,
        binding_rate=config.mechanics.binding_rate,
        unbinding_rate=config.mechanics.unbinding_rate,
        adhesion=config.mechanics.adhesion_kt,
        capture_radius_factor=config.mechanics.capture_radius_factor,
    )
    reduced.validate()
    return theta, reduced
