"""Simulation result objects and analysis-friendly count extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import SimulationConfig

Observable = Literal["contacts", "synapses", "kills"]


@dataclass(slots=True)
class SimulationResult:
    """Trajectories, discrete events, final state, and configuration for one run."""

    snapshots: pd.DataFrame
    events: pd.DataFrame
    final_cells: pd.DataFrame
    config: SimulationConfig

    @property
    def duration(self) -> float:
        return float(self.config.duration)

    def event_counts_per_killer(self, event: str) -> np.ndarray:
        """Count an event for every killer, retaining zero-count killers."""

        out = np.zeros(self.config.n_killers, dtype=np.int64)
        if self.events.empty:
            return out
        rows = self.events[
            (self.events["event"] == event)
            & self.events["killer_id"].notna()
        ]
        if rows.empty:
            return out
        counts = rows["killer_id"].astype(int).value_counts()
        valid = counts.index.to_numpy(dtype=int)
        mask = (valid >= 0) & (valid < out.size)
        out[valid[mask]] = counts.to_numpy(dtype=int)[mask]
        return out

    def contacts_per_cell(self) -> np.ndarray:
        """Return proximity-contact episode counts for every killer cell."""

        return self.event_counts_per_killer("contact_started")

    def synapses_per_cell(self) -> np.ndarray:
        """Return bound-synapse episode counts for every killer cell."""

        return self.event_counts_per_killer("synapse_formed")

    def kills_per_cell(self) -> np.ndarray:
        """Return integer target-kill counts using primary damage attribution."""

        return self.event_counts_per_killer("target_killed")

    def counts_per_cell(self, observable: Observable) -> np.ndarray:
        """Return per-killer counts for a supported observable."""

        if observable == "contacts":
            return self.contacts_per_cell()
        if observable == "synapses":
            return self.synapses_per_cell()
        if observable == "kills":
            return self.kills_per_cell()
        raise ValueError("observable must be 'contacts', 'synapses', or 'kills'")

    def summary(self) -> dict[str, float | int]:
        """Return common population-level outcomes."""

        target_rows = self.final_cells[self.final_cells["cell_type"] == "target"]
        targets_alive = int(target_rows["alive"].sum())
        targets_killed = int(self.config.n_targets - targets_alive)
        return {
            "n_killers": self.config.n_killers,
            "n_targets": self.config.n_targets,
            "duration": self.duration,
            "contact_episodes": int(self.contacts_per_cell().sum()),
            "synapse_episodes": int(self.synapses_per_cell().sum()),
            "targets_killed": targets_killed,
            "fraction_targets_killed": targets_killed / self.config.n_targets,
        }

    def infer(self, observable: Observable = "kills", **kwargs):
        """Fit the Bayesian count-model suite to contacts, synapses, or kills."""

        from .inference import infer_result

        return infer_result(self, observable=observable, **kwargs)

    def save(self, directory: str | Path) -> Path:
        """Save portable CSV tables and a JSON configuration."""

        import json

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.snapshots.to_csv(output / "snapshots.csv", index=False)
        self.events.to_csv(output / "events.csv", index=False)
        self.final_cells.to_csv(output / "final_cells.csv", index=False)
        (output / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        return output
