#!/usr/bin/env python3
"""Numerically stable implementations of Huang et al. derandomization.

Two related algorithms are kept separate:

``appendix_c_schedule``
    The coefficient-weighted, prefix-consistent measurement stream used for
    the quantum-chemistry experiments in Appendix C, Eqs. (C6)--(C11), of
    Huang, Kueng, and Preskill, Phys. Rev. Lett. 127, 030503 (2021).
    A caller supplies the external shot budget.

``github_target_schedule``
    The public author's C++ command-line semantics: ``R`` is a target number
    of hits per unit-weight observable, targets are ``floor(w_i R)``, and the
    returned schedule has variable length.

The public reference implementation is
https://github.com/hsinyuan-huang/predicting-quantum-properties,
especially ``data_acquisition_shadow.cpp``.  Its C++ implementation compares
only cost changes for observables affected by the current single-qubit choice.
Doing the same here is essential: comparing three complete exponential sums
loses the small candidate-dependent difference once many common terms dominate.

Pauli codes are I=0, X=1, Y=2, Z=3.  Ties are resolved X, then Y, then Z, as
in the author's C++ loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


PAPER_ETA = 0.9
PAULI_ACTIONS = np.asarray((1, 2, 3), dtype=np.int8)


@dataclass(frozen=True)
class ScheduleAudit:
    """Compact diagnostics for a generated deterministic schedule."""

    mode: str
    shots: int
    observables: int
    qubits: int
    distinct_settings: int
    last_new_setting: int
    terminal_repeat_length: int
    minimum_hits: int
    maximum_hits: int
    zero_hit_observables: int
    eta: float
    seeded: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "shots": self.shots,
            "observables": self.observables,
            "qubits": self.qubits,
            "distinct_settings": self.distinct_settings,
            "last_new_setting": self.last_new_setting,
            "terminal_repeat_length": self.terminal_repeat_length,
            "minimum_hits": self.minimum_hits,
            "maximum_hits": self.maximum_hits,
            "zero_hit_observables": self.zero_hit_observables,
            "eta": self.eta,
            "seeded": self.seeded,
        }


def _validated_observables(observables: np.ndarray) -> np.ndarray:
    result = np.asarray(observables, dtype=np.int8)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("observables must be a nonempty (L, n) array")
    if np.any((result < 0) | (result > 3)):
        raise ValueError("Pauli codes must be I=0, X=1, Y=2, Z=3")
    return result


def _coefficient_weights(coefficients: np.ndarray, count: int) -> np.ndarray:
    values = np.abs(np.asarray(coefficients, dtype=float).reshape(-1))
    if values.size != count:
        raise ValueError("one coefficient is required for every observable")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all observable coefficients must be finite and nonzero")
    return values / float(np.max(values))


def _official_weights(weights: np.ndarray, count: int) -> np.ndarray:
    result = np.asarray(weights, dtype=float).reshape(-1)
    if result.size != count:
        raise ValueError("one weight is required for every observable")
    if (
        np.any(~np.isfinite(result))
        or np.any(result <= 0.0)
        or np.any(result > 1.0)
    ):
        raise ValueError("official GitHub weights must lie in (0, 1]")
    return result


def _validated_hits(initial_hits: np.ndarray | None, count: int) -> np.ndarray:
    if initial_hits is None:
        return np.zeros(count, dtype=np.int64)
    raw = np.asarray(initial_hits)
    if raw.shape != (count,) or np.any(~np.isfinite(raw)):
        raise ValueError("initial_hits must be a finite length-L vector")
    if np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError("initial_hits must contain nonnegative integers")
    return raw.astype(np.int64, copy=True)


def setting_hits(observables: np.ndarray, setting: np.ndarray) -> np.ndarray:
    """Return which observables are measured by one complete Pauli setting."""

    obs = _validated_observables(observables)
    basis = np.asarray(setting, dtype=np.int8).reshape(-1)
    if basis.shape != (obs.shape[1],) or np.any((basis < 1) | (basis > 3)):
        raise ValueError("setting must contain one X/Y/Z code per qubit")
    return np.all((obs == 0) | (obs == basis), axis=1)


def schedule_hit_counts(observables: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Count per-observable hits without materializing an L-by-T matrix."""

    obs = _validated_observables(observables)
    rows = np.asarray(schedule, dtype=np.int8)
    if rows.ndim != 2 or rows.shape[1] != obs.shape[1]:
        raise ValueError("schedule must have shape (T, n)")
    hits = np.zeros(obs.shape[0], dtype=np.int64)
    for row in rows:
        hits += np.all((obs == 0) | (obs == row), axis=1)
    return hits


def _remaining_locality_after_site(observables: np.ndarray) -> np.ndarray:
    """Return r_l(k+1) in Eq. (C8), including r_l(n)=0."""

    nonidentity = observables != 0
    suffix = np.zeros((observables.shape[1] + 1, observables.shape[0]), dtype=np.int16)
    for site in range(observables.shape[1] - 1, -1, -1):
        suffix[site] = suffix[site + 1] + nonidentity[:, site]
    return suffix[1:]


def _choose_setting(
    observables: np.ndarray,
    weights: np.ndarray,
    hits: np.ndarray,
    remaining_after: np.ndarray,
    eta: float,
    eligible: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one row by the stable affected-term cost difference.

    This follows the public C++ affected-term cost difference.  The mismatch
    contribution of every active observable is common to X/Y/Z, so it is
    cancelled analytically before summation; only the additional reward for
    matching each candidate label remains.  A shared max shift and ``expm1``
    make that reward stable.  At the final site ``r_l(k+1)=0``; hits are updated
    only after the full setting is fixed.
    """

    half_eta = eta / 2.0
    nu = -math.expm1(-half_eta)
    setting = np.empty(observables.shape[1], dtype=np.int8)
    prefix_compatible = np.ones(observables.shape[0], dtype=bool)

    for site in range(observables.shape[1]):
        labels = observables[:, site]
        active = prefix_compatible & (labels != 0)
        if eligible is not None:
            active &= eligible

        if np.any(active):
            active_indices = np.flatnonzero(active)
            active_weights = weights[active_indices]
            log_base = -half_eta * hits[active_indices] / active_weights
            shift = float(np.max(log_base))

            r_after = remaining_after[site, active_indices]
            log_match = np.log1p(
                -nu * np.power(1.0 / 3.0, r_after)
            ) / active_weights

            # This is minus the match-vs-mismatch cost delta.  Cancelling the
            # candidate-common mismatch sum also preserves exact X/Y/Z ties.
            gains = np.exp(log_base - shift) * (-np.expm1(log_match))
            label_gains = np.bincount(
                labels[active_indices], weights=gains, minlength=4
            )
            action = int(PAULI_ACTIONS[int(np.argmax(label_gains[1:4]))])
        else:
            action = 1

        setting[site] = action
        prefix_compatible &= (labels == 0) | (labels == action)

    return setting, prefix_compatible


def _audit(
    mode: str,
    schedule: np.ndarray,
    hits: np.ndarray,
    observable_count: int,
    qubits: int,
    eta: float,
    seeded: bool,
) -> ScheduleAudit:
    seen: set[tuple[int, ...]] = set()
    last_new = 0
    for index, row in enumerate(schedule, 1):
        key = tuple(map(int, row))
        if key not in seen:
            seen.add(key)
            last_new = index
    terminal = 0
    if len(schedule):
        final = schedule[-1]
        for row in schedule[::-1]:
            if np.array_equal(row, final):
                terminal += 1
            else:
                break
    return ScheduleAudit(
        mode=mode,
        shots=int(len(schedule)),
        observables=int(observable_count),
        qubits=int(qubits),
        distinct_settings=int(len(seen)),
        last_new_setting=int(last_new),
        terminal_repeat_length=int(terminal),
        minimum_hits=int(np.min(hits)),
        maximum_hits=int(np.max(hits)),
        zero_hit_observables=int(np.count_nonzero(hits == 0)),
        eta=float(eta),
        seeded=bool(seeded),
    )


def appendix_c_schedule(
    observables: np.ndarray,
    coefficients: np.ndarray,
    num_shots: int,
    *,
    initial_hits: np.ndarray | None = None,
    eta: float = PAPER_ETA,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, ScheduleAudit]:
    """Generate the weighted Appendix-C schedule for an external shot budget.

    Multiplying all ``coefficients`` by a common nonzero constant leaves the
    schedule unchanged.  Supplying ``initial_hits`` continues the same cost
    recursion after an externally supplied coverage prefix; the prefix itself
    is not silently treated as part of the unseeded paper schedule.
    """

    obs = _validated_observables(observables)
    if int(num_shots) != num_shots or num_shots < 0:
        raise ValueError("num_shots must be a nonnegative integer")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    weights = _coefficient_weights(coefficients, obs.shape[0])
    seeded = initial_hits is not None
    hits = _validated_hits(initial_hits, obs.shape[0])
    remaining_after = _remaining_locality_after_site(obs)
    rows = np.empty((int(num_shots), obs.shape[1]), dtype=np.int8)
    for shot in range(int(num_shots)):
        rows[shot], measured = _choose_setting(
            obs, weights, hits, remaining_after, eta, eligible=None
        )
        hits += measured

    audit = _audit(
        "appendix_c_fixed_budget",
        rows,
        hits,
        obs.shape[0],
        obs.shape[1],
        eta,
        seeded,
    )
    if return_audit:
        return rows, audit
    return rows


def github_target_schedule(
    observables: np.ndarray,
    weights: np.ndarray,
    measurements_per_observable: int,
    *,
    eta: float = PAPER_ETA,
    max_shots: int = 10_000_000,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, ScheduleAudit]:
    """Reproduce the public author's C++ ``-d R`` target-hit semantics.

    Here ``R`` is *not* a total shot count.  Observable ``l`` remains in the
    greedy cost until it has at least ``floor(weight_l * R)`` hits, and the
    function returns the resulting variable-length schedule.
    """

    obs = _validated_observables(observables)
    if (
        int(measurements_per_observable) != measurements_per_observable
        or measurements_per_observable < 0
    ):
        raise ValueError("measurements_per_observable must be a nonnegative integer")
    if int(max_shots) != max_shots or max_shots <= 0:
        raise ValueError("max_shots must be a positive integer")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")

    normalized = _official_weights(weights, obs.shape[0])
    targets = np.floor(normalized * int(measurements_per_observable)).astype(np.int64)
    hits = np.zeros(obs.shape[0], dtype=np.int64)
    remaining_after = _remaining_locality_after_site(obs)
    rows: list[np.ndarray] = []

    while np.any(hits < targets):
        if len(rows) >= int(max_shots):
            raise RuntimeError(
                "target schedule exceeded max_shots before all hit targets were met"
            )
        eligible = hits < targets
        setting, measured = _choose_setting(
            obs, normalized, hits, remaining_after, eta, eligible=eligible
        )
        rows.append(setting)
        hits += measured

    schedule = (
        np.asarray(rows, dtype=np.int8)
        if rows
        else np.empty((0, obs.shape[1]), dtype=np.int8)
    )
    audit = _audit(
        "github_target_hits",
        schedule,
        hits,
        obs.shape[0],
        obs.shape[1],
        eta,
        seeded=False,
    )
    if return_audit:
        return schedule, audit
    return schedule


__all__ = [
    "PAPER_ETA",
    "ScheduleAudit",
    "appendix_c_schedule",
    "github_target_schedule",
    "schedule_hit_counts",
    "setting_hits",
]
