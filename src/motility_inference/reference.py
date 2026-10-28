"""Independent-generator validation using the supplied MantiShrimp engine.

The spatial design and requested physical scales are shared with the fast
benchmark, but the dynamics execute the original Python/Euler polarity ABM.
This is deliberately a cross-generator stress test, not an exact-equivalence
claim. It starts polarity at zero, uses explicit fine timesteps, changes the
polarity-to-position coupling near targets, and retains reciprocal mechanical
forces. A changing coupling is not algebraically identical to the fast
velocity-state model, which is precisely one useful misspecification check.
"""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from mantishrimp.config import (
    CytotoxicityConfig, DomainConfig, MechanicsConfig, MotilityConfig,
    OutputConfig, SimulationConfig,
)
from mantishrimp.simulation import Simulation

from .simulator import SimConfig, _initial_positions, proximity


def simulate_reference(theta, config: SimConfig | dict | None = None, seed: int = 0,
                       *, substeps: int = 10) -> dict:
    """Run the original ABM and return the common observed-trajectory schema.

    Per-cell log-speed offsets are sampled with the same hierarchical meaning
    as in the fast simulator. Killing/exhaustion and same-type repulsion are
    disabled to isolate migration. Other requested mechanics remain active.
    Target Brownian translation noise is zero; optional target active motion
    uses the requested stationary OU scale. No experimental data are used.

    Frames are selected from their known fixed-step timestamps, not interpolated.
    A final floating-point duplicate is harmless. A runtime check refuses
    off-grid snapshots. Wrapped source positions are unwrapped using minimum
    images; this requires each frame displacement to be below half a box.
    """
    config = SimConfig() if config is None else (SimConfig(**config) if isinstance(config, dict) else config)
    config.validate()
    theta = np.array(theta, dtype=float, copy=True)
    if theta.shape != (6,) or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite vector of length six")
    if not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if config.spontaneous_sd:
        raise ValueError("reference bridge does not implement spontaneous log-speed fluctuations")
    speed, tau, jitter = np.exp(theta[:3])
    if not np.isfinite([speed, tau, jitter]).all() or min(speed, tau, jitter) <= 0:
        raise ValueError("physical scales must be finite and positive")
    layout_seed, motion_seed, observation_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(layout_seed)
    initial_x, initial_targets = _initial_positions(rng, config)
    offsets = rng.normal(0, config.heterogeneity_sd, config.n_killers)
    source_seed = int(motion_seed.generate_state(1)[0])
    original_config = SimulationConfig(
        n_killers=config.n_killers, n_targets=config.n_targets,
        killer_radius=config.contact_distance / 2,
        target_radius=config.contact_distance / 2,
        duration=(config.n_frames - 1) * config.frame_dt,
        max_dt=config.frame_dt / substeps,
        max_drift_displacement=1e12,
        seed=source_seed,
        domain=DomainConfig(0, config.box_size, 0, config.box_size, "periodic"),
        motility=MotilityConfig(
            killer_polarity_decay=1 / tau,
            killer_polarity_noise=speed * np.sqrt(2 / tau),
            killer_speed=1.0,
            killer_translation_noise=jitter,
            target_polarity_decay=1 / config.target_tau,
            target_polarity_noise=config.target_speed * np.sqrt(2 / config.target_tau),
            target_speed=1.0 if config.target_speed else 0.0,
            target_translation_noise=0.0,
            target_interaction_length=config.interaction_length,
            killer_delta_speed=theta[3], killer_delta_tau=theta[4], killer_delta_jitter=theta[5],
        ),
        mechanics=MechanicsConfig(
            repulsion_kk=0.0, repulsion_tt=0.0, repulsion_kt=config.repulsion,
            adhesion_kt=config.adhesion, binding_rate=config.binding_rate,
            unbinding_rate=config.unbinding_rate,
            capture_radius_factor=config.capture_radius_factor,
            contact_tolerance_factor=0.0,
        ),
        cytotoxicity=CytotoxicityConfig(
            killing_rate_min=0, killing_rate_max=0,
            exhaustion_rate_min=0, exhaustion_rate_max=0,
            kill_probability=0,
        ),
        output=OutputConfig(record_every=substeps),
    )
    result = Simulation(original_config).run(
        initial_killer_positions=initial_x,
        initial_target_positions=initial_targets,
        killer_speed_factors=np.exp(offsets),
    )
    requested_times = np.arange(config.n_frames) * config.frame_dt
    all_times = np.sort(result.snapshots["time"].unique())
    chosen = np.argmin(np.abs(all_times[:, None] - requested_times[None, :]), axis=0)
    if not np.allclose(all_times[chosen], requested_times, atol=1e-8, rtol=1e-9):
        raise RuntimeError("reference snapshots do not match the requested fixed frame grid")
    chosen_times = all_times[chosen]
    positions = np.empty((config.n_killers, config.n_frames, 2))
    targets = np.empty((config.n_targets, config.n_frames, 2))
    for kind, array in (("killer", positions), ("target", targets)):
        for i in range(array.shape[0]):
            rows = result.snapshots[(result.snapshots["cell_type"] == kind) & (result.snapshots["cell_id"] == i)]
            rows = rows.set_index("time").loc[chosen_times]
            array[i] = rows[["x", "y"]].to_numpy()
    increments = np.diff(positions, axis=1)
    increments -= config.box_size * np.rint(increments / config.box_size)
    true_positions = np.concatenate((positions[:, :1], positions[:, :1] + np.cumsum(increments, axis=1)), axis=1)
    observed_rng = np.random.default_rng(observation_seed)
    observed = true_positions + observed_rng.normal(0, config.localization_sd, true_positions.shape)
    gate = proximity(true_positions, targets, config.box_size, config.contact_distance, config.interaction_length)[0]
    return {
        "positions": observed,
        "targets": targets,
        "times": requested_times,
        "true_positions": true_positions,
        "true_gate": gate,
        "cell_offsets": offsets,
        "theta": theta,
        "config": asdict(config),
        "seed": int(seed),
        "schema_version": 1,
        "generator": "mantishrimp_euler_polarity",
        "reference_config": original_config.to_dict(),
        "reference_substeps": int(substeps),
        "reference_notes": [
            "Initial spatial design is shared, but dynamics execute the supplied ABM independently.",
            "Initial polarity is zero, not stationary.",
            "Changing polarity coupling differs from a velocity OU with changing stationary SD.",
            "Reciprocal killer-target forces can move targets even when target_speed is zero.",
        ],
    }
