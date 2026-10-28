"""Hookean killer-target agent-based simulation engine.

The implementation follows the refined model in Szonja Skenderovic's thesis:
off-lattice Ornstein-Uhlenbeck motility, Hookean overlap repulsion, explicit
stochastic killer-target binding/unbinding, cumulative target damage and
killer exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import DomainConfig, MechanicsConfig, SimulationConfig
from .results import SimulationResult


EVENT_COLUMNS = [
    "time",
    "step",
    "event",
    "killer_id",
    "target_id",
    "value",
    "reason",
]


def _minimum_image(displacement: np.ndarray, domain: DomainConfig) -> np.ndarray:
    out = np.asarray(displacement, dtype=float).copy()
    if domain.boundary == "periodic":
        out[..., 0] -= domain.width * np.round(out[..., 0] / domain.width)
        out[..., 1] -= domain.height * np.round(out[..., 1] / domain.height)
    return out


def _kt_geometry(
    killer_positions: np.ndarray,
    target_positions: np.ndarray,
    domain: DomainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = killer_positions[:, None, :] - target_positions[None, :, :]
    displacement = _minimum_image(displacement, domain)
    distance = np.linalg.norm(displacement, axis=2)
    return displacement, distance


def _sample_clipped_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    size: int,
    *,
    lower: float,
    upper: float,
    values: np.ndarray | None = None,
) -> np.ndarray:
    if values is not None:
        values = np.asarray(values, dtype=float)
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"initial values must be a finite array of shape ({size},)")
        if np.any((values < lower) | (values > upper)):
            raise ValueError(f"initial values must lie in [{lower}, {upper}]")
        return values.copy()
    if sd == 0:
        return np.full(size, np.clip(mean, lower, upper), dtype=float)
    return np.clip(rng.normal(mean, sd, size=size), lower, upper)


def _initial_positions(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Place all cells uniformly with a pair-specific non-overlap constraint."""

    n_total = config.n_killers + config.n_targets
    radii = np.concatenate(
        [
            np.full(config.n_killers, config.killer_radius),
            np.full(config.n_targets, config.target_radius),
        ]
    )
    positions = np.empty((n_total, 2), dtype=float)
    placed: list[int] = []

    for cell_index in rng.permutation(n_total):
        radius = radii[cell_index]
        if config.domain.boundary == "confined":
            x_low, x_high = config.domain.x_min + radius, config.domain.x_max - radius
            y_low, y_high = config.domain.y_min + radius, config.domain.y_max - radius
        else:
            x_low, x_high = config.domain.x_min, config.domain.x_max
            y_low, y_high = config.domain.y_min, config.domain.y_max
        if x_high <= x_low or y_high <= y_low:
            raise ValueError("the domain is too small for the configured cell radii")

        for _ in range(config.max_initialisation_attempts):
            candidate = rng.uniform([x_low, y_low], [x_high, y_high])
            if not placed:
                positions[cell_index] = candidate
                placed.append(int(cell_index))
                break
            other_indices = np.asarray(placed, dtype=int)
            displacement = candidate - positions[other_indices]
            displacement = _minimum_image(displacement, config.domain)
            distance = np.linalg.norm(displacement, axis=1)
            minimum = (
                (radius + radii[other_indices]) * config.initial_separation_factor
            )
            if np.all(distance >= minimum):
                positions[cell_index] = candidate
                placed.append(int(cell_index))
                break
        else:
            raise RuntimeError(
                "could not place all cells without overlap; enlarge the domain, "
                "reduce cell counts/radii, or lower initial_separation_factor"
            )
    return positions


def _apply_boundary(
    positions: np.ndarray,
    polarities: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
    domain: DomainConfig,
) -> None:
    if domain.boundary == "periodic":
        positions[alive, 0] = domain.x_min + (
            positions[alive, 0] - domain.x_min
        ) % domain.width
        positions[alive, 1] = domain.y_min + (
            positions[alive, 1] - domain.y_min
        ) % domain.height
        return

    lower = np.array([domain.x_min, domain.y_min]) + radii[:, None]
    upper = np.array([domain.x_max, domain.y_max]) - radii[:, None]
    while True:
        below = alive[:, None] & (positions < lower)
        above = alive[:, None] & (positions > upper)
        outside = below | above
        if not outside.any():
            break
        positions[below] = 2 * lower[below] - positions[below]
        positions[above] = 2 * upper[above] - positions[above]
        polarities[outside] *= -1


def _bond_transitions(bound, distance, eligible, mechanics: MechanicsConfig, contact_distance, dt, rng):
    """Update bonds and yield transitions before the caller samples productivity."""
    p_bind = 1.0 - np.exp(-mechanics.binding_rate * dt)
    p_unbind = 1.0 - np.exp(-mechanics.unbinding_rate * dt)
    capture_cutoff = mechanics.capture_radius_factor * contact_distance
    # Keep row-major draw order, including productivity draws between yields.
    for killer_id, target_id in np.argwhere(bound | eligible):
        if bound[killer_id, target_id]:
            separation = distance[killer_id, target_id] > capture_cutoff
            stochastic = rng.random() < p_unbind
            if separation or stochastic:
                bound[killer_id, target_id] = False
                yield killer_id, target_id, "separation" if separation else "stochastic"
        elif rng.random() < p_bind:
            bound[killer_id, target_id] = True
            yield killer_id, target_id, None


def _forces(
    positions: np.ndarray,
    alive: np.ndarray,
    bound: np.ndarray,
    config: SimulationConfig,
    *,
    pairwise: bool = False,
) -> np.ndarray:
    """Return net or pairwise forces using batched minimum-image geometry."""

    n_k = config.n_killers
    n_total = positions.shape[0]
    radii = np.concatenate(
        [np.full(n_k, config.killer_radius),
         np.full(config.n_targets, config.target_radius)]
    )
    # Batch unordered pairs to avoid thousands of tiny Python/NumPy calls.
    first, second = np.triu_indices(n_total, k=1)
    cross = (first < n_k) & (second >= n_k)
    is_bound = np.zeros(first.size, dtype=bool)
    is_bound[cross] = bound[first[cross], second[cross] - n_k]
    active = alive[first] & (alive[second] | is_bound)
    displacement = _minimum_image(positions[first] - positions[second], config.domain)
    distance = np.linalg.norm(displacement, axis=1)
    unit = np.divide(
        displacement, distance[:, None], out=np.zeros_like(displacement),
        where=distance[:, None] != 0,
    )
    contact_distance = radii[first] + radii[second]
    stiffness = np.where(
        cross, config.mechanics.repulsion_kt,
        np.where(first < n_k, config.mechanics.repulsion_kk,
                 config.mechanics.repulsion_tt),
    )
    pair_force = (stiffness * np.maximum(contact_distance - distance, 0.0))[:, None] * unit
    adhesion = is_bound & (
        distance < config.mechanics.capture_radius_factor * contact_distance
    )
    pair_force[adhesion] -= config.mechanics.adhesion_kt * unit[adhesion]
    pair_force[~active] = 0.0
    # shortcut: Dense pairs use O(N^2) memory; use neighbour lists if memory becomes limiting.
    pairs = np.zeros((n_total, n_total, 2))
    pairs[first, second] = pair_force
    pairs[second, first] = -pair_force
    return pairs if pairwise else pairs.sum(axis=1)


def _record_snapshots(
    rows: list[dict[str, float | int | str | bool]],
    *,
    step: int,
    time: float,
    positions: np.ndarray,
    polarities: np.ndarray,
    killer_state: np.ndarray,
    target_state: np.ndarray,
    target_damage: np.ndarray,
    target_alive: np.ndarray,
    bound: np.ndarray,
    kill_probability: np.ndarray,
    primary_killer: np.ndarray,
    config: SimulationConfig,
) -> None:
    for killer_id in range(config.n_killers):
        rows.append(
            {
                "step": step,
                "time": time,
                "cell_type": "killer",
                "cell_id": killer_id,
                "x": positions[killer_id, 0],
                "y": positions[killer_id, 1],
                "polarity_x": polarities[killer_id, 0],
                "polarity_y": polarities[killer_id, 1],
                "alive": True,
                "state": killer_state[killer_id],
                "death_factor": np.nan,
                "synapses": np.flatnonzero(bound[killer_id]).tolist(),
                "kill_probability": kill_probability[killer_id],
                "primary_killer": None,
            }
        )
    for target_id in range(config.n_targets):
        global_id = config.n_killers + target_id
        rows.append(
            {
                "step": step,
                "time": time,
                "cell_type": "target",
                "cell_id": target_id,
                "x": positions[global_id, 0],
                "y": positions[global_id, 1],
                "polarity_x": polarities[global_id, 0],
                "polarity_y": polarities[global_id, 1],
                "alive": bool(target_alive[target_id]),
                "state": target_state[target_id],
                "death_factor": target_damage[target_id],
                "synapses": np.flatnonzero(bound[:, target_id]).tolist(),
                "kill_probability": np.nan,
                "primary_killer": (
                    int(primary_killer[target_id]) if primary_killer[target_id] >= 0 else None
                ),
            }
        )


def _event(
    events: list[dict[str, float | int | str | None]],
    *,
    time: float,
    step: int,
    name: str,
    killer_id: int | None = None,
    target_id: int | None = None,
    value: float | None = None,
    reason: str | None = None,
) -> None:
    events.append(
        {
            "time": float(time),
            "step": int(step),
            "event": name,
            "killer_id": killer_id,
            "target_id": target_id,
            "value": value,
            "reason": reason,
        }
    )


@dataclass(slots=True)
class Simulation:
    """Executable Hookean killer-target agent-based model."""

    config: SimulationConfig

    def __post_init__(self) -> None:
        self.config.validate()

    def run(
        self,
        *,
        progressbar: bool = False,
        initial_killer_positions: np.ndarray | None = None,
        initial_target_positions: np.ndarray | None = None,
        initial_killer_state: np.ndarray | None = None,
        initial_target_state: np.ndarray | None = None,
        initial_death_factor: np.ndarray | None = None,
        initial_kill_probability: np.ndarray | None = None,
        killer_speed_factors: np.ndarray | None = None,
    ) -> SimulationResult:
        """Run with configured sampling or explicit per-cell initial arrays.

        States and probabilities must lie in [0, 1], and death factors between
        zero and the death threshold. Supplied arrays are copied, not mutated.
        Optional positive killer_speed_factors supply per-cell motility
        heterogeneity; None preserves the original homogeneous speed.
        Set progressbar=True to display progress through simulated time.
        """
        config = self.config
        rng = np.random.default_rng(config.seed)
        n_k, n_t = config.n_killers, config.n_targets
        n_total = n_k + n_t

        if (initial_killer_positions is None) != (initial_target_positions is None):
            raise ValueError("provide both initial position arrays or neither")
        if initial_killer_positions is None:
            positions = _initial_positions(config, rng)
        else:
            killer_positions = np.asarray(initial_killer_positions, dtype=float)
            target_positions = np.asarray(initial_target_positions, dtype=float)
            if killer_positions.shape != (n_k, 2) or target_positions.shape != (n_t, 2):
                raise ValueError("initial position arrays have incompatible shapes")
            positions = np.vstack([killer_positions, target_positions]).copy()
            if not np.all(np.isfinite(positions)):
                raise ValueError("initial positions must be finite")

        polarities = np.zeros((n_total, 2), dtype=float)
        killer_state = _sample_clipped_normal(
            rng,
            config.cytotoxicity.initial_killer_state,
            config.cytotoxicity.killer_state_sd,
            n_k,
            values=initial_killer_state,
            lower=0.0,
            upper=1.0,
        )
        target_state = _sample_clipped_normal(
            rng,
            config.cytotoxicity.initial_target_state,
            config.cytotoxicity.target_state_sd,
            n_t,
            values=initial_target_state,
            lower=0.0,
            upper=1.0,
        )
        target_damage = _sample_clipped_normal(
            rng,
            config.cytotoxicity.initial_death_factor,
            config.cytotoxicity.initial_death_factor_sd,
            n_t,
            values=initial_death_factor,
            lower=0.0,
            upper=config.cytotoxicity.death_threshold,
        )
        target_alive = target_damage < config.cytotoxicity.death_threshold
        alive = np.concatenate([np.ones(n_k, dtype=bool), target_alive.copy()])
        kill_probability = _sample_clipped_normal(
            rng,
            config.cytotoxicity.kill_probability,
            config.cytotoxicity.kill_probability_sd,
            n_k,
            values=initial_kill_probability,
            lower=0.0,
            upper=1.0,
        )

        bound = np.zeros((n_k, n_t), dtype=bool)
        productive = np.zeros((n_k, n_t), dtype=bool)
        damage_contribution = np.zeros((n_k, n_t), dtype=float)
        primary_killer = np.full(n_t, -1, dtype=int)
        events: list[dict[str, float | int | str | None]] = []
        snapshot_rows: list[dict[str, float | int | str | bool]] = []

        _, initial_distance = _kt_geometry(positions[:n_k], positions[n_k:], config.domain)
        contact_distance = config.killer_radius + config.target_radius
        proximity_cutoff = contact_distance * (
            1.0 + config.mechanics.contact_tolerance_factor
        )
        proximity = (initial_distance <= proximity_cutoff) & target_alive[None, :]
        for killer_id, target_id in np.argwhere(proximity):
            _event(
                events,
                time=0.0,
                step=0,
                name="contact_started",
                killer_id=int(killer_id),
                target_id=int(target_id),
            )

        time = 0.0
        step = 0
        _record_snapshots(
            snapshot_rows,
            step=step,
            time=time,
            positions=positions,
            polarities=polarities,
            killer_state=killer_state,
            target_state=target_state,
            target_damage=target_damage,
            target_alive=target_alive,
            bound=bound,
            kill_probability=kill_probability,
            primary_killer=primary_killer,
            config=config,
        )

        radii = np.concatenate(
            [
                np.full(n_k, config.killer_radius),
                np.full(n_t, config.target_radius),
            ]
        )
        polarity_decay = np.concatenate(
            [
                np.full(n_k, config.motility.killer_polarity_decay),
                np.full(n_t, config.motility.target_polarity_decay),
            ]
        )[:, None]
        polarity_noise = np.concatenate(
            [
                np.full(n_k, config.motility.killer_polarity_noise),
                np.full(n_t, config.motility.target_polarity_noise),
            ]
        )[:, None]
        speed = np.concatenate(
            [
                np.full(n_k, config.motility.killer_speed),
                np.full(n_t, config.motility.target_speed),
            ]
        )[:, None]
        if killer_speed_factors is not None:
            speed_factors = np.asarray(killer_speed_factors, dtype=float)
            if speed_factors.shape != (n_k,) or not np.all(np.isfinite(speed_factors)) or np.any(speed_factors <= 0):
                raise ValueError("killer_speed_factors must be finite positive values, one per killer")
            speed = speed.astype(float)
            speed[:n_k, 0] *= speed_factors
        translation_noise = np.concatenate(
            [
                np.full(n_k, config.motility.killer_translation_noise),
                np.full(n_t, config.motility.target_translation_noise),
            ]
        )[:, None]

        motility = config.motility
        has_proximity_response = any((motility.killer_delta_speed,
                                      motility.killer_delta_tau,
                                      motility.killer_delta_jitter))

        last_progress = 0
        if progressbar:
            print("Simulation: [                    ]   0%", end="", flush=True)

        while time < config.duration:
            _, distance = _kt_geometry(positions[:n_k], positions[n_k:], config.domain)
            proximity_now = (distance <= proximity_cutoff) & target_alive[None, :]
            capture_cutoff = config.mechanics.capture_radius_factor * contact_distance
            if has_proximity_response:
                nearest_live = np.min(np.where(target_alive[None, :], distance, np.inf), axis=1)
                gate = np.exp(-np.maximum(nearest_live - contact_distance, 0.0) / motility.target_interaction_length)
                local_speed = speed.astype(float)
                local_decay = polarity_decay.astype(float)
                local_polarity_noise = polarity_noise.astype(float)
                local_translation_noise = translation_noise.astype(float)
                local_speed[:n_k, 0] *= np.exp(motility.killer_delta_speed * gate)
                local_decay[:n_k, 0] *= np.exp(-motility.killer_delta_tau * gate)
                local_polarity_noise[:n_k, 0] *= np.exp(-0.5 * motility.killer_delta_tau * gate)
                local_translation_noise[:n_k, 0] *= np.exp(motility.killer_delta_jitter * gate)
            else:
                # Preserve the original arithmetic/RNG path when effects are off.
                local_speed = speed
                local_decay = polarity_decay
                local_polarity_noise = polarity_noise
                local_translation_noise = translation_noise
            initial_forces = _forces(positions, alive, bound, config)
            deterministic_drift = local_speed * polarities + initial_forces
            # Release removes adhesion for live targets and the whole pair force for dead ones.
            switchable = (
                (
                    (bound & (config.mechanics.unbinding_rate > 0))
                    | (~bound & proximity_now & (config.mechanics.binding_rate > 0))
                )
                & (distance < capture_cutoff)
            )
            force_change = switchable * np.where(
                target_alive[None, :],
                config.mechanics.adhesion_kt,
                np.abs(
                    config.mechanics.repulsion_kt * np.maximum(contact_distance - distance, 0.0)
                    - config.mechanics.adhesion_kt
                ),
            )
            drift_bound = np.linalg.norm(deterministic_drift, axis=1)
            drift_bound[:n_k] += force_change.sum(axis=1)
            drift_bound[n_k:] += force_change.sum(axis=0)
            maximum_drift = float(
                np.max(drift_bound[alive], initial=0.0)
            )
            adaptive_dt = (
                config.max_drift_displacement / maximum_drift
                if maximum_drift > 0
                else config.max_dt
            )
            dt = min(config.max_dt, adaptive_dt, config.duration - time)
            next_time = time + dt
            next_step = step + 1

            for killer_id, target_id, reason in _bond_transitions(
                bound, distance, proximity_now, config.mechanics, contact_distance, dt, rng
            ):
                if reason is not None:
                    productive[killer_id, target_id] = False
                    _event(
                        events, time=time, step=step, name="synapse_ended",
                        killer_id=killer_id, target_id=target_id, reason=reason,
                    )
                else:
                    productive[killer_id, target_id] = rng.random() < kill_probability[killer_id]
                    _event(
                        events, time=time, step=step, name="synapse_formed",
                        killer_id=killer_id, target_id=target_id,
                        value=float(productive[killer_id, target_id]),
                    )
                    kill_probability[killer_id] = max(
                        0.0, kill_probability[killer_id]
                        - config.cytotoxicity.kill_probability_decay_per_synapse,
                    )

            forces = _forces(positions, alive, bound, config)
            old_polarities = polarities.copy()
            polarities[alive] = (
                old_polarities[alive]
                - local_decay[alive] * old_polarities[alive] * dt
                + local_polarity_noise[alive] * np.sqrt(dt) * rng.normal(size=(alive.sum(), 2))
            )
            positions[alive] += (
                local_speed[alive] * old_polarities[alive] * dt
                + forces[alive] * dt
                + local_translation_noise[alive]
                * np.sqrt(dt)
                * rng.normal(size=(alive.sum(), 2))
            )
            _apply_boundary(positions, polarities, radii, alive, config.domain)

            active_pairs = bound & productive
            cytotoxicity = config.cytotoxicity
            # Productive conjugation exhausts killers even after the target dies.
            exhaustion_increment = (
                active_pairs * (
                    cytotoxicity.exhaustion_rate_min
                    + (cytotoxicity.exhaustion_rate_max - cytotoxicity.exhaustion_rate_min)
                    * target_state
                ) * dt
            ).sum(axis=1)
            individual_rates = active_pairs * target_state * (
                cytotoxicity.killing_rate_min
                + (cytotoxicity.killing_rate_max - cytotoxicity.killing_rate_min)
                * killer_state[:, None]
            )
            individual_rates[:, ~target_alive] = 0.0
            target_damage[target_alive] += individual_rates[:, target_alive].sum(axis=0) * dt
            damage_contribution += individual_rates * dt
            target_damage[target_alive] -= (
                cytotoxicity.recovery_rate * target_damage[target_alive] * dt
            )
            np.maximum(target_damage, 0.0, out=target_damage)
            if cytotoxicity.instant_kill:
                target_damage[target_alive & active_pairs.any(axis=0)] = cytotoxicity.death_threshold

            dying = np.flatnonzero(target_alive & (target_damage >= cytotoxicity.death_threshold))
            target_damage[dying] = cytotoxicity.death_threshold
            target_alive[dying] = False
            alive[n_k + dying] = False
            for target_id in dying:
                contributors = np.flatnonzero(damage_contribution[:, target_id] > 0)
                primary = (
                    int(np.argmax(damage_contribution[:, target_id]))
                    if contributors.size
                    else None
                )
                if cytotoxicity.instant_kill:
                    active_killers = np.flatnonzero(active_pairs[:, target_id])
                    if active_killers.size:
                        primary = int(active_killers[0])
                if primary is not None:
                    primary_killer[target_id] = primary
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="target_killed",
                    killer_id=primary,
                    target_id=target_id,
                    value=1.0,
                )
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="target_died",
                    killer_id=primary,
                    target_id=target_id,
                    value=float(target_damage[target_id]),
                )

            killer_state *= np.clip(1.0 - exhaustion_increment, 0.0, 1.0)
            killer_state = np.clip(killer_state, 0.0, 1.0)

            _, distance_after = _kt_geometry(
                positions[:n_k], positions[n_k:], config.domain
            )
            for killer_id, target_id in np.argwhere(
                bound & (distance_after > capture_cutoff)
            ):
                bound[killer_id, target_id] = False
                productive[killer_id, target_id] = False
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="synapse_ended",
                    killer_id=int(killer_id),
                    target_id=int(target_id),
                    reason="separation",
                )

            proximity_after = (
                distance_after <= proximity_cutoff
            ) & target_alive[None, :]
            for killer_id, target_id in np.argwhere(proximity_after & ~proximity):
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="contact_started",
                    killer_id=int(killer_id),
                    target_id=int(target_id),
                )
            for killer_id, target_id in np.argwhere(proximity & ~proximity_after):
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="contact_ended",
                    killer_id=int(killer_id),
                    target_id=int(target_id),
                )
            proximity = proximity_after
            time, step = next_time, next_step

            if step % config.output.record_every == 0 or time >= config.duration:
                _record_snapshots(
                    snapshot_rows,
                    step=step,
                    time=time,
                    positions=positions,
                    polarities=polarities,
                    killer_state=killer_state,
                    target_state=target_state,
                    target_damage=target_damage,
                    target_alive=target_alive,
                    bound=bound,
                    kill_probability=kill_probability,
                    primary_killer=primary_killer,
                    config=config,
                )

            if progressbar:
                percent = min(100, int(100 * (time / config.duration)))
                if percent > last_progress:
                    print(
                        f"\rSimulation: [{'#' * (percent // 5):<20}] {percent:3d}%",
                        end="", flush=True,
                    )
                    last_progress = percent

        if progressbar:
            print()

        snapshots = pd.DataFrame(snapshot_rows)
        event_table = pd.DataFrame(events, columns=EVENT_COLUMNS)
        final_cells = snapshots[snapshots["time"] == snapshots["time"].max()].copy()
        final_cells.reset_index(drop=True, inplace=True)
        return SimulationResult(
            snapshots=snapshots,
            events=event_table,
            final_cells=final_cells,
            config=config,
        )


def simulate(
    config: SimulationConfig | None = None,
    *,
    progressbar: bool = False,
    **config_overrides,
) -> SimulationResult:
    """Run a simulation, optionally displaying progress through simulated time."""

    if config is not None and config_overrides:
        raise ValueError("pass either config or keyword overrides, not both")
    resolved = config if config is not None else SimulationConfig(**config_overrides)
    return Simulation(resolved).run(progressbar=progressbar)


__all__ = ["Simulation", "simulate"]
