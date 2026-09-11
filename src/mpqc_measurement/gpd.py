"""Latest sequential periodic-iSWAP Greedy Projection Decomposition (GPD)."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from ._version import __version__
from .allocation import evaluate_error, integer_range_allocation, require_positive_integer
from .models import DecompositionResult, Fragment, HamiltonianData
from .resources import gpd_gate_resources, summarize_gate_resources
from .selection import (
    CALIBRATION_SIZES,
    FrontierPoint,
    MaterialStallTracker,
    calibrate_weights,
    sector_indices,
    select_frontier,
)


@dataclass(frozen=True)
class GPDConfig:
    """Configuration of the sequential full-diagonal GPD source fit.

    ``paper_depth=4``, ``steps=600``, and one start reproduce the settings used
    for the four-qubit random-Hamiltonian experiments.  The internal circuit
    has ``paper_depth + 1`` local Euler layers.
    """

    paper_depth: int = 4
    max_terms: int = 40
    starts: int = 1
    steps: int = 600
    learning_rate: float = 1.0e-2
    seed: int = 20260808
    x64: bool = False
    calibrate: bool = True
    calibration_seed: int = 918273
    relative_improvement: float = 2.0e-3
    stall_patience: int = 5
    minimum_k: int = 10
    odd_qubit_mode: str = "reject"
    progress: bool = False

    def validate(self, n_qubits: int, shots: int) -> None:
        if self.paper_depth < 0 or self.max_terms < 1:
            raise ValueError("paper_depth must be nonnegative and max_terms positive")
        if self.starts < 1 or self.steps < 1 or self.learning_rate <= 0.0:
            raise ValueError("starts, steps, and learning_rate must be positive")
        if self.calibrate and self.max_terms < max(CALIBRATION_SIZES):
            raise ValueError("calibrated GPD requires max_terms >= 10")
        if self.minimum_k < max(CALIBRATION_SIZES):
            raise ValueError("minimum_k must be at least the calibration cutoff 10")
        if self.stall_patience < 1 or not 0.0 <= self.relative_improvement < 1.0:
            raise ValueError(
                "stall_patience must be positive and relative_improvement in [0,1)"
            )
        if int(shots) < 1:
            raise ValueError("shots must be positive")
        if n_qubits > 1 and n_qubits % 2 and self.odd_qubit_mode != "open-chain":
            raise ValueError(
                "paper-faithful periodic-iSWAP GPD requires an even qubit count; "
                "set odd_qubit_mode='open-chain' to request the documented odd-n extension"
            )
        if self.odd_qubit_mode not in {"reject", "open-chain"}:
            raise ValueError("odd_qubit_mode must be 'reject' or 'open-chain'")


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - exercised by CLI environments
        raise ImportError(
            "GPD requires JAX. Install this project with: pip install -e \".[gpd]\""
        ) from error
    return jax, jnp


def _canonical_fragment_key(jax, base_seed: int, k: int):
    key = jax.random.PRNGKey(int(base_seed))
    fragment_key = key
    for _ in range(int(k)):
        key, fragment_key = jax.random.split(key)
    return fragment_key


class _CircuitActions:
    """Dense operator/state actions with native iSWAP gates (no CNOT synthesis)."""

    def __init__(self, n_qubits: int, n_layers: int, odd_qubit_mode: str):
        self.jax, self.jnp = _jax_modules()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.dim = 2**self.n_qubits
        self.tensor_shape = (2,) * (2 * self.n_qubits)
        self.state_shape = (2,) * self.n_qubits
        self.entanglers = {}
        for parity in (0, 1):
            for inverse in (False, True):
                permutation, phase = self._entangler_map(parity, inverse, odd_qubit_mode)
                inverse_permutation = np.argsort(permutation)
                output_phase = phase[inverse_permutation]
                self.entanglers[(parity, inverse)] = (
                    self.jnp.asarray(inverse_permutation),
                    self.jnp.asarray(output_phase),
                )

    def _entangler_map(
        self, parity: int, inverse: bool, odd_qubit_mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.n_qubits <= 1:
            pairs: list[tuple[int, int]] = []
        elif self.n_qubits % 2 == 0:
            pairs = [
                (q, (q + 1) % self.n_qubits)
                for q in range(parity, self.n_qubits, 2)
            ]
        elif odd_qubit_mode == "open-chain":
            pairs = [(q, q + 1) for q in range(parity, self.n_qubits - 1, 2)]
        else:  # validated before construction
            raise ValueError("odd periodic iSWAP layers are not disjoint")
        permutation = np.empty(self.dim, dtype=np.int32)
        phase = np.empty(self.dim, dtype=np.complex128)
        phase_unit = -1j if inverse else 1j
        for index in range(self.dim):
            bits = list(map(int, f"{index:0{self.n_qubits}b}"))
            output = bits.copy()
            amplitude = 1.0 + 0.0j
            for left, right in pairs:
                if bits[left] != bits[right]:
                    amplitude *= phase_unit
                output[left], output[right] = bits[right], bits[left]
            permutation[index] = int("".join(map(str, output)), 2)
            phase[index] = amplitude
        if len(np.unique(permutation)) != self.dim:
            raise AssertionError("iSWAP layer is not a permutation")
        return permutation, phase

    def _rx(self, theta):
        c = self.jnp.cos(theta / 2)
        s = self.jnp.sin(theta / 2)
        return self.jnp.asarray(((c, -1j * s), (-1j * s, c)))

    def _ry(self, theta):
        c = self.jnp.cos(theta / 2)
        s = self.jnp.sin(theta / 2)
        return self.jnp.asarray(((c, -s), (s, c)))

    def _local_euler(self, theta):
        return self._rx(theta[2]) @ self._ry(theta[1]) @ self._rx(theta[0])

    def _conjugate_local(self, operator, gate, qubit: int):
        tensor = operator.reshape(self.tensor_shape)
        tensor = self.jnp.tensordot(gate, tensor, axes=((1,), (qubit,)))
        tensor = self.jnp.moveaxis(tensor, 0, qubit)
        tensor = self.jnp.tensordot(
            tensor, self.jnp.conj(gate), axes=((self.n_qubits + qubit,), (1,))
        )
        tensor = self.jnp.moveaxis(tensor, -1, self.n_qubits + qubit)
        return tensor.reshape((self.dim, self.dim))

    def _conjugate_entangler(self, operator, parity: int, inverse: bool):
        inverse_permutation, output_phase = self.entanglers[(parity, inverse)]
        selected = operator[
            inverse_permutation[:, None], inverse_permutation[None, :]
        ]
        return selected * output_phase[:, None] * self.jnp.conj(output_phase[None, :])

    def conjugate_forward(self, operator, parameters):
        for layer in range(self.n_layers):
            if layer:
                operator = self._conjugate_entangler(operator, layer % 2, False)
            for qubit in range(self.n_qubits):
                operator = self._conjugate_local(
                    operator, self._local_euler(parameters[layer, qubit]), qubit
                )
        return operator

    def conjugate_inverse(self, operator, parameters):
        for layer in reversed(range(self.n_layers)):
            for qubit in range(self.n_qubits):
                gate = self._local_euler(parameters[layer, qubit])
                operator = self._conjugate_local(operator, self.jnp.conj(gate.T), qubit)
            if layer:
                operator = self._conjugate_entangler(operator, layer % 2, True)
        return operator

    def _evolve_local(self, state, gate, qubit: int):
        tensor = state.reshape(self.state_shape)
        tensor = self.jnp.tensordot(gate, tensor, axes=((1,), (qubit,)))
        return self.jnp.moveaxis(tensor, 0, qubit).reshape((self.dim,))

    def _evolve_entangler(self, state, parity: int, inverse: bool):
        inverse_permutation, output_phase = self.entanglers[(parity, inverse)]
        return output_phase * state[inverse_permutation]

    def evolve_state_forward(self, state, parameters):
        for layer in range(self.n_layers):
            if layer:
                state = self._evolve_entangler(state, layer % 2, False)
            for qubit in range(self.n_qubits):
                state = self._evolve_local(
                    state, self._local_euler(parameters[layer, qubit]), qubit
                )
        return state


def build_gpd_frontier(
    data: HamiltonianData, shots: int, config: GPDConfig
) -> list[FrontierPoint]:
    shots = require_positive_integer(shots, name="shots")
    config.validate(data.n_qubits, shots)
    jax, jnp = _jax_modules()
    jax.config.update("jax_enable_x64", bool(config.x64))
    complex_dtype = jnp.complex128 if config.x64 else jnp.complex64
    real_dtype = jnp.float64 if config.x64 else jnp.float32
    actions = _CircuitActions(
        data.n_qubits, config.paper_depth + 1, config.odd_qubit_mode
    )

    def loss(parameters, residual):
        rotated = actions.conjugate_forward(residual, parameters)
        diagonal = jnp.diag(rotated)
        return jnp.real(jnp.vdot(rotated, rotated)) - jnp.real(
            jnp.vdot(diagonal, diagonal)
        )

    value_gradient = jax.jit(jax.value_and_grad(loss))
    loss_only = jax.jit(loss)
    forward = jax.jit(actions.conjugate_forward)
    inverse = jax.jit(actions.conjugate_inverse)
    unitary_from_parameters = jax.jit(
        jax.vmap(actions.evolve_state_forward, in_axes=(1, None), out_axes=1)
    )
    dimension = data.dimension
    constant = float(np.trace(data.matrix).real / dimension)
    centered_target = data.matrix - constant * np.eye(dimension, dtype=np.complex128)
    # The first source fit sees the full Hamiltonian, matching the frozen GPD
    # trajectory.  Its exact trace component is removed from the first saved
    # diagonal, leaving K=0 as the scalar-only candidate.
    fit_residual_np = np.array(data.matrix, dtype=np.complex128, copy=True)
    residual = jnp.asarray(fit_residual_np, dtype=complex_dtype)
    fragments: list[Fragment] = []
    frontier: list[FrontierPoint] = [
        FrontierPoint(
            size=0,
            constant=constant,
            fragments=[],
            approximate=constant * np.eye(dimension, dtype=np.complex128),
            residual=centered_target,
            metadata={
                "source_fit": "exact trace scalar",
                "rank_zero_candidate": True,
            },
        )
    ]
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
    term_limit = (
        int(config.max_terms)
        if config.calibrate
        else min(int(config.max_terms), int(shots))
    )
    calibration_weights: tuple[float, float] | None = None
    stall_tracker: MaterialStallTracker | None = None
    stop_reason = ""
    right_censored = True
    selection_indices = sector_indices(data.n_qubits, data.particle_number)

    def fixed_budget_score(point: FrontierPoint) -> float | None:
        assert calibration_weights is not None
        try:
            allocation = integer_range_allocation(
                [fragment.half_range for fragment in point.fragments], int(shots)
            )
        except ValueError:
            return None
        x_a = float(
            np.linalg.norm(
                point.residual[np.ix_(selection_indices, selection_indices)],
                ord="fro",
            )
        )
        x_s_squared = sum(
            fragment.half_range**2 / int(count)
            for fragment, count in zip(point.fragments, allocation)
            if int(count) > 0
        )
        return math.hypot(
            calibration_weights[0] * x_a,
            calibration_weights[1] * math.sqrt(max(x_s_squared, 0.0)),
        )

    for k in range(1, term_limit + 1):
        primary_key = _canonical_fragment_key(jax, config.seed, k)
        winners = []
        for start in range(config.starts):
            key = primary_key if start == 0 else jax.random.fold_in(primary_key, start)
            parameters = jax.random.uniform(
                key,
                (config.paper_depth + 1, data.n_qubits, 3),
                minval=-0.1 * np.pi,
                maxval=0.1 * np.pi,
                dtype=real_dtype,
            )
            first = jnp.zeros_like(parameters)
            second = jnp.zeros_like(parameters)
            for step in range(1, config.steps + 1):
                # The loss is explicitly real, but JAX may surface NumPy's
                # ComplexWarning while backpropagating through complex gates.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                    _, gradient = value_gradient(parameters, residual)
                first = beta1 * first + (1.0 - beta1) * gradient
                second = beta2 * second + (1.0 - beta2) * gradient * gradient
                corrected_first = first / (1.0 - beta1**step)
                corrected_second = second / (1.0 - beta2**step)
                parameters = parameters - config.learning_rate * corrected_first / (
                    jnp.sqrt(corrected_second) + epsilon
                )
                if config.progress and (step == 1 or step == config.steps or step % 100 == 0):
                    print(
                        f"GPD K={k} start={start} step={step} offdiag_F2="
                        f"{float(loss_only(parameters, residual)):.10g}",
                        flush=True,
                    )
            candidate_rotated = forward(residual, parameters)
            candidate_diagonal = jnp.real(jnp.diag(candidate_rotated))
            candidate_off_diagonal = candidate_rotated - jnp.diag(
                candidate_diagonal.astype(candidate_rotated.dtype)
            )
            candidate_loss = float(
                jnp.real(jnp.vdot(candidate_off_diagonal, candidate_off_diagonal))
            )
            winners.append((candidate_loss, start, parameters))
        final_loss, winning_start, parameters = min(winners, key=lambda item: (item[0], item[1]))
        rotated = forward(residual, parameters)
        raw_diagonal = jnp.real(jnp.diag(rotated))
        off_diagonal = rotated - jnp.diag(raw_diagonal.astype(rotated.dtype))
        unitary = np.asarray(
            unitary_from_parameters(jnp.eye(dimension, dtype=complex_dtype), parameters),
            dtype=np.complex128,
        )
        diagonal = np.array(raw_diagonal, dtype=float, copy=True)
        if k == 1:
            diagonal -= float(np.trace(data.matrix).real / dimension)
        midpoint = 0.5 * float(np.max(diagonal) + np.min(diagonal))
        diagonal -= midpoint
        constant += midpoint
        fragment = Fragment(
            unitary=unitary,
            diagonal=diagonal,
            label=f"gpd-{k:03d}",
            metadata={
                "parameters": np.asarray(parameters, dtype=float).tolist(),
                "winning_start": int(winning_start),
                "offdiagonal_frobenius_squared": float(final_loss),
                "paper_depth": int(config.paper_depth),
            },
        )
        fragments.append(fragment)
        residual = inverse(off_diagonal, parameters)
        residual.block_until_ready()
        fit_residual_np = np.asarray(residual, dtype=np.complex128)
        fit_residual_np = (fit_residual_np + fit_residual_np.conj().T) / 2.0
        residual = jnp.asarray(fit_residual_np, dtype=complex_dtype)
        approximate = constant * np.eye(dimension, dtype=np.complex128)
        for saved_fragment in fragments:
            approximate += saved_fragment.matrix
        approximate = (approximate + approximate.conj().T) / 2.0
        residual_np = (data.matrix - approximate + (data.matrix - approximate).conj().T) / 2.0
        frontier.append(
            FrontierPoint(
                size=k,
                constant=constant,
                fragments=list(fragments),
                approximate=approximate,
                residual=residual_np,
                metadata={
                    "residual_frobenius": float(np.linalg.norm(residual_np, ord="fro")),
                    "source_fit": "sequential full-diagonal GPD",
                },
            )
        )
        residual_norm = float(np.linalg.norm(residual_np, ord="fro"))
        if residual_norm <= 1.0e-12:
            stop_reason = f"residual converged at K={k}"
            right_censored = False
            break
        if config.calibrate and k == max(CALIBRATION_SIZES):
            weight_a, weight_s, _ = calibrate_weights(
                data, frontier, seed=config.calibration_seed
            )
            calibration_weights = (weight_a, weight_s)
            stall_tracker = MaterialStallTracker(
                relative_improvement=config.relative_improvement,
                patience=config.stall_patience,
                minimum_k=config.minimum_k,
                calibration_cutoff=max(CALIBRATION_SIZES),
            )
            for calibration_point in frontier:
                stall_tracker.update(
                    calibration_point.size,
                    fixed_budget_score(calibration_point),
                )
        elif config.calibrate and k > max(CALIBRATION_SIZES):
            assert stall_tracker is not None
            score = fixed_budget_score(frontier[-1])
            if stall_tracker.update(k, score):
                stop_reason = (
                    "no calibrated fixed-budget loss improvement above "
                    f"{config.relative_improvement:g} for {config.stall_patience} "
                    "consecutive post-calibration boundaries"
                )
                right_censored = False
                break
    if len(frontier) == 1:
        raise RuntimeError("GPD produced no nonconstant fragment")
    if not stop_reason:
        stop_reason = (
            f"reached max_terms={term_limit} before the material-stall rule"
            if config.calibrate
            else f"reached max_terms={term_limit} in explicitly uncalibrated mode"
        )
    terminal_audit = {
        "terminal_k": int(frontier[-1].size),
        "requested_max_terms": int(config.max_terms),
        "right_censored": bool(right_censored),
        "stop_reason": stop_reason,
        "relative_material_improvement": float(config.relative_improvement),
        "stall_patience": int(config.stall_patience),
        "minimum_k": int(config.minimum_k),
        "events": [] if stall_tracker is None else stall_tracker.events,
    }
    for point in frontier:
        point.metadata["frontier_audit"] = terminal_audit
    return frontier


def _gate_metadata(n_qubits: int, config: GPDConfig, allocation) -> dict:
    resource = gpd_gate_resources(n_qubits, config.paper_depth, config.odd_qubit_mode)
    return {
        "gate_resources_per_setting": (
            {**resource.as_dict(), "continuous_angles": 3 * n_qubits * (config.paper_depth + 1)}
            if len(allocation) else None
        ),
        "executed_gate_resources": summarize_gate_resources(
            [resource] * len(allocation), allocation
        ),
    }


def fit_gpd(
    data: HamiltonianData,
    shots: int,
    config: GPDConfig | None = None,
) -> DecompositionResult:
    settings = config or GPDConfig()
    total_shots = require_positive_integer(shots, name="shots")
    settings.validate(data.n_qubits, total_shots)
    constant = float(np.trace(data.matrix).real / data.dimension)
    centered = data.matrix - constant * np.eye(data.dimension, dtype=np.complex128)
    if float(np.linalg.norm(centered, ord="fro")) <= 1.0e-14:
        approximate = constant * np.eye(data.dimension, dtype=np.complex128)
        error = evaluate_error(data, approximate, [], np.zeros(0, dtype=int))
        return DecompositionResult(
            method="GPD",
            shots=total_shots,
            constant=constant,
            fragments=[],
            shot_allocation=np.zeros(0, dtype=int),
            approximate_hamiltonian=approximate,
            residual=data.matrix - approximate,
            error=error,
            selected_size=0,
            metadata={
                "config": asdict(settings),
                "package_version": __version__,
                "input_hamiltonian_sha256": data.hamiltonian_sha256,
                "constant_hamiltonian": True,
                **_gate_metadata(data.n_qubits, settings, []),
                "unused_shots": total_shots,
                "target_state_used_for_construction": False,
            },
        )
    frontier = build_gpd_frontier(data, total_shots, settings)
    point, allocation, proxy, selection = select_frontier(
        data,
        frontier,
        total_shots,
        calibrate=settings.calibrate,
        calibration_seed=settings.calibration_seed,
    )
    error = evaluate_error(
        data,
        point.approximate,
        point.fragments,
        allocation,
        calibrated_proxy_rmse=proxy if settings.calibrate else None,
    )
    return DecompositionResult(
        method="GPD",
        shots=total_shots,
        constant=point.constant,
        fragments=point.fragments,
        shot_allocation=allocation,
        approximate_hamiltonian=point.approximate,
        residual=point.residual,
        error=error,
        selected_size=point.size,
        metadata={
            "config": asdict(settings),
            "package_version": __version__,
            "input_hamiltonian_sha256": data.hamiltonian_sha256,
            "unused_shots": total_shots - int(np.sum(allocation)),
            "selection": selection,
            **_gate_metadata(data.n_qubits, settings, allocation),
            "frontier_terminal_size": frontier[-1].size,
            "selected_at_frontier_endpoint": point.size == frontier[-1].size,
            "frontier": frontier[-1].metadata.get("frontier_audit", {}),
            "selection_is_provisional": bool(
                frontier[-1].metadata.get("frontier_audit", {}).get(
                    "right_censored", False
                )
            ),
            "numeric_profile": {
                "optimizer": "float64/complex128" if settings.x64 else "float32/complex64",
                "stored_analysis": "float64/complex128",
                "paper_artifact_profile": not settings.x64,
            },
            "qubit_schedule": (
                "paper periodic-iSWAP"
                if data.n_qubits % 2 == 0
                else "explicit odd-n open-chain extension"
            ),
            "dense_preprocessing_is_exponential": True,
            "target_state_used_for_construction": False,
        },
    )
