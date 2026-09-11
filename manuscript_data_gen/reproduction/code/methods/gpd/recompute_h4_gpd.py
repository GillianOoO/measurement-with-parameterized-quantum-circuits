#!/usr/bin/env python3
"""Deterministically recompute the missing H4 GPD fragment record.

This is a self-contained JAX implementation of the retained GPD source
semantics.  At greedy step k it minimizes the squared Frobenius norm of the
off-diagonal part of U_k H^(k) U_k^dagger, stores

    Lambda_k = diag(U_k H^(k) U_k^dagger),

and updates H^(k+1) by rotating the off-diagonal part back.  The circuit uses
the archived Rx-Ry-Rx layers and alternating periodic iSWAP brickwork.

The input NPZ must contain the dense Hamiltonian ``H``, its ground-state
vector ``state``, and scalar ``energy``.  It is produced from the UCCSD H4
pickle by ``exact_gpd_tnd_variance.py``; keeping that conversion separate
lets this optimization run in a clean native-arm64 JAX environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rx(theta: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.asarray(((c, -1j * s), (-1j * s, c)))


def ry(theta: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.asarray(((c, -s), (s, c)))


def local_euler(theta: jnp.ndarray) -> jnp.ndarray:
    # Gates are applied in the archived order Rx(theta0), Ry(theta1),
    # Rx(theta2), so their combined state-action matrix is the reverse product.
    return rx(theta[2]) @ ry(theta[1]) @ rx(theta[0])


def entangler_map(n_qubits: int, parity: int, inverse: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return output index and phase for one disjoint periodic iSWAP layer."""
    dim = 2**n_qubits
    permutation = np.empty(dim, dtype=np.int32)
    phase = np.empty(dim, dtype=np.complex128)
    pairs = [(q, (q + 1) % n_qubits) for q in range(parity, n_qubits, 2)]
    phase_unit = -1j if inverse else 1j
    for index in range(dim):
        bits = list(map(int, f"{index:0{n_qubits}b}"))
        out = bits.copy()
        amplitude = 1.0 + 0.0j
        for left, right in pairs:
            if bits[left] != bits[right]:
                amplitude *= phase_unit
            out[left], out[right] = bits[right], bits[left]
        permutation[index] = int("".join(map(str, out)), 2)
        phase[index] = amplitude
    return permutation, phase


class CircuitActions:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2**n_qubits
        self.tensor_shape = (2,) * (2 * n_qubits)
        self.state_shape = (2,) * n_qubits
        self.entanglers: dict[tuple[int, bool], tuple[jnp.ndarray, jnp.ndarray]] = {}
        for parity in (0, 1):
            for inverse in (False, True):
                permutation, phase = entangler_map(n_qubits, parity, inverse)
                inv_permutation = np.argsort(permutation)
                # For U O U^dagger, output indices select their input through
                # inv_permutation and acquire the corresponding input phase.
                output_phase = phase[inv_permutation]
                self.entanglers[(parity, inverse)] = (
                    jnp.asarray(inv_permutation),
                    jnp.asarray(output_phase),
                )

    def conjugate_local(
        self, operator: jnp.ndarray, gate: jnp.ndarray, qubit: int
    ) -> jnp.ndarray:
        tensor = operator.reshape(self.tensor_shape)
        tensor = jnp.tensordot(gate, tensor, axes=((1,), (qubit,)))
        tensor = jnp.moveaxis(tensor, 0, qubit)
        # Right multiplication by gate^dagger: (gate^*)[new_bra, old_bra].
        tensor = jnp.tensordot(
            tensor, jnp.conj(gate), axes=((self.n_qubits + qubit,), (1,))
        )
        tensor = jnp.moveaxis(tensor, -1, self.n_qubits + qubit)
        return tensor.reshape((self.dim, self.dim))

    def conjugate_entangler(
        self, operator: jnp.ndarray, parity: int, inverse: bool
    ) -> jnp.ndarray:
        inv_permutation, output_phase = self.entanglers[(parity, inverse)]
        selected = operator[inv_permutation[:, None], inv_permutation[None, :]]
        return selected * output_phase[:, None] * jnp.conj(output_phase[None, :])

    def conjugate_forward(
        self, operator: jnp.ndarray, parameters: jnp.ndarray
    ) -> jnp.ndarray:
        for layer in range(self.n_layers):
            if layer != 0:
                operator = self.conjugate_entangler(operator, layer % 2, False)
            for qubit in range(self.n_qubits):
                operator = self.conjugate_local(
                    operator, local_euler(parameters[layer, qubit]), qubit
                )
        return operator

    def conjugate_inverse(
        self, operator: jnp.ndarray, parameters: jnp.ndarray
    ) -> jnp.ndarray:
        for layer in reversed(range(self.n_layers)):
            for qubit in range(self.n_qubits):
                gate = local_euler(parameters[layer, qubit])
                operator = self.conjugate_local(operator, jnp.conj(gate.T), qubit)
            if layer != 0:
                operator = self.conjugate_entangler(operator, layer % 2, True)
        return operator

    def evolve_state_local(
        self, state: jnp.ndarray, gate: jnp.ndarray, qubit: int
    ) -> jnp.ndarray:
        tensor = state.reshape(self.state_shape)
        tensor = jnp.tensordot(gate, tensor, axes=((1,), (qubit,)))
        return jnp.moveaxis(tensor, 0, qubit).reshape((self.dim,))

    def evolve_state_entangler(
        self, state: jnp.ndarray, parity: int, inverse: bool
    ) -> jnp.ndarray:
        # Recover the input->output map from the output->input representation.
        inv_permutation, output_phase = self.entanglers[(parity, inverse)]
        return output_phase * state[inv_permutation]

    def evolve_state_forward(
        self, state: jnp.ndarray, parameters: jnp.ndarray
    ) -> jnp.ndarray:
        for layer in range(self.n_layers):
            if layer != 0:
                state = self.evolve_state_entangler(state, layer % 2, False)
            for qubit in range(self.n_qubits):
                state = self.evolve_state_local(
                    state, local_euler(parameters[layer, qubit]), qubit
                )
        return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragments", type=int, default=15)
    parser.add_argument("--internal-layers", type=int, default=13)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--shots", type=int, default=2038)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.input)
    hamiltonian_np = np.asarray(archive["H"], dtype=np.complex128)
    state_np = np.asarray(archive["state"], dtype=np.complex128)
    energy = float(np.asarray(archive["energy"]))
    dim = hamiltonian_np.shape[0]
    n_qubits = int(round(np.log2(dim)))
    if hamiltonian_np.shape != (dim, dim) or 2**n_qubits != dim:
        raise ValueError(f"Invalid Hamiltonian shape: {hamiltonian_np.shape}")
    if state_np.shape != (dim,):
        raise ValueError(f"Invalid state shape: {state_np.shape}")

    complex_dtype = jnp.complex128 if args.x64 else jnp.complex64
    real_dtype = jnp.float64 if args.x64 else jnp.float32
    actions = CircuitActions(n_qubits, args.internal_layers)
    residual = jnp.asarray(hamiltonian_np, dtype=complex_dtype)
    state = jnp.asarray(state_np, dtype=complex_dtype)

    def loss_fn(parameters: jnp.ndarray, current_residual: jnp.ndarray) -> jnp.ndarray:
        rotated = actions.conjugate_forward(current_residual, parameters)
        diagonal = jnp.diag(rotated)
        total_sq = jnp.real(jnp.vdot(rotated, rotated))
        diagonal_sq = jnp.real(jnp.vdot(diagonal, diagonal))
        return total_sq - diagonal_sq

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    forward_operator = jax.jit(actions.conjugate_forward)
    inverse_operator = jax.jit(actions.conjugate_inverse)
    forward_state = jax.jit(actions.evolve_state_forward)

    key = jax.random.PRNGKey(args.seed)
    fragments: list[dict[str, object]] = []
    h_hat = np.zeros_like(hamiltonian_np)
    started = time.time()
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8

    for fragment_index in range(args.fragments):
        key, subkey = jax.random.split(key)
        parameters = jax.random.uniform(
            subkey,
            (args.internal_layers, n_qubits, 3),
            minval=-0.1 * np.pi,
            maxval=0.1 * np.pi,
            dtype=real_dtype,
        )
        first_moment = jnp.zeros_like(parameters)
        second_moment = jnp.zeros_like(parameters)
        loss_value = np.nan
        fragment_started = time.time()
        for step in range(1, args.steps + 1):
            value, gradient = value_and_grad(parameters, residual)
            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
            corrected_first = first_moment / (1 - beta1**step)
            corrected_second = second_moment / (1 - beta2**step)
            parameters = parameters - args.learning_rate * corrected_first / (
                jnp.sqrt(corrected_second) + epsilon
            )
            if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                loss_value = float(value)
                print(
                    f"fragment={fragment_index:02d} step={step:04d} "
                    f"offdiag_fro2={loss_value:.12g}",
                    flush=True,
                )

        # ``value`` above is evaluated before the last Adam parameter update.
        # Re-evaluate the objective from the final parameters so the recorded
        # loss and the residual below describe exactly the same iterate.
        rotated = forward_operator(residual, parameters)
        diagonal = jnp.real(jnp.diag(rotated))
        off_diagonal = rotated - jnp.diag(diagonal.astype(rotated.dtype))
        final_loss_value = float(jnp.real(jnp.vdot(off_diagonal, off_diagonal)))
        if not np.isfinite(final_loss_value) or final_loss_value < 0.0:
            raise FloatingPointError(
                f"Invalid final loss for fragment {fragment_index}: "
                f"{final_loss_value}"
            )
        residual = inverse_operator(off_diagonal, parameters)
        rotated_state = forward_state(state, parameters)
        probabilities = jnp.real(jnp.conj(rotated_state) * rotated_state)
        mu_k = float(jnp.dot(probabilities, diagonal))
        s_k = float(jnp.dot(probabilities, diagonal * diagonal))
        w_k = float(jnp.max(jnp.abs(diagonal)))
        for name, value in (("w_k", w_k), ("s_k", s_k)):
            if not np.isfinite(value) or value <= 0.0:
                raise FloatingPointError(
                    f"Fragment {fragment_index} has non-positive or non-finite "
                    f"{name}: {value}"
                )

        parameters_np = np.asarray(parameters, dtype=np.float64)
        diagonal_np = np.asarray(diagonal, dtype=np.float64)
        parameter_path = args.output / f"{fragment_index:02d}_param.npy"
        diagonal_path = args.output / f"{fragment_index:02d}_diag.npy"
        np.save(parameter_path, parameters_np)
        np.save(diagonal_path, diagonal_np)
        term = np.asarray(
            inverse_operator(
                jnp.diag(diagonal.astype(complex_dtype)), parameters
            ),
            dtype=np.complex128,
        )
        h_hat += term
        residual_np = np.asarray(residual, dtype=np.complex128)
        fragment = {
            "k": fragment_index + 1,
            "archive_iteration": fragment_index,
            "parameter_path": str(parameter_path.resolve()),
            "parameter_sha256": sha256(parameter_path),
            "diagonal_path": str(diagonal_path.resolve()),
            "diagonal_sha256": sha256(diagonal_path),
            "offdiagonal_frobenius_squared": final_loss_value,
            "last_preupdate_offdiagonal_frobenius_squared": float(loss_value),
            "residual_frobenius": float(np.linalg.norm(residual_np)),
            "residual_spectral": float(np.linalg.norm(residual_np, ord=2)),
            "w_k": w_k,
            "mu_k": mu_k,
            "s_k": s_k,
            "elapsed_seconds": time.time() - fragment_started,
        }
        fragments.append(fragment)
        (args.output / "checkpoint.json").write_text(
            json.dumps({"fragments": fragments}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"fragment={fragment_index:02d} complete "
            f"residual_fro={fragment['residual_frobenius']:.12g} "
            f"residual_2={fragment['residual_spectral']:.12g} "
            f"elapsed={fragment['elapsed_seconds']:.1f}s",
            flush=True,
        )

    weights = np.asarray([row["w_k"] for row in fragments], dtype=float)
    s_values = np.asarray([row["s_k"] for row in fragments], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise FloatingPointError("All fragment weights w_k must be finite and positive")
    if not np.all(np.isfinite(s_values)) or np.any(s_values <= 0.0):
        raise FloatingPointError("All fragment second moments s_k must be finite and positive")
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise FloatingPointError(
            f"Invalid norm-probability denominator sum_k w_k: {weight_sum}"
        )
    p_norm = weights / weight_sum
    sqrt_s = np.sqrt(np.maximum(s_values, 0.0))
    sqrt_s_sum = float(sqrt_s.sum())
    if not np.isfinite(sqrt_s_sum) or sqrt_s_sum <= 0.0:
        raise FloatingPointError(
            "Invalid state-aware probability denominator "
            f"sum_k sqrt(s_k): {sqrt_s_sum}"
        )
    p_state = sqrt_s / sqrt_s_sum
    if (
        not np.all(np.isfinite(p_norm))
        or np.any(p_norm <= 0.0)
        or not np.all(np.isfinite(p_state))
        or np.any(p_state <= 0.0)
    ):
        raise FloatingPointError("Computed fragment probabilities must be finite and positive")
    for row, norm_probability, state_probability in zip(fragments, p_norm, p_state):
        row["p_norm"] = float(norm_probability)
        row["p_state"] = float(state_probability)

    expectation_h = float(np.real(np.vdot(state_np, hamiltonian_np @ state_np)))
    expectation_h_hat = float(np.real(np.vdot(state_np, h_hat @ state_np)))
    bias = expectation_h_hat - expectation_h
    bias_squared = bias * bias
    m1_norm = float(np.sum(s_values / p_norm))
    m1_state = float(np.sum(s_values / p_state))
    reconstruction = hamiltonian_np - h_hat
    result = {
        "record_id": "molecular_h4_gpd_r1p2_recomputed",
        "method": "GPD",
        "availability_status": "deterministically_recomputed_missing_archive",
        "benchmark": "H4, R=1.2 Angstrom, exact ground state",
        "n_qubits": n_qubits,
        "K": args.fragments,
        "internal_layers": args.internal_layers,
        "paper_depth_L": args.internal_layers - 1,
        "optimizer": "Adam",
        "optimizer_steps_per_fragment": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "jax_enable_x64": args.x64,
        "input_npz": str(args.input.resolve()),
        "input_npz_sha256": sha256(args.input),
        "expectation_h": expectation_h,
        "ground_energy_stored": energy,
        "expectation_h_hat": expectation_h_hat,
        "bias": bias,
        "bias_squared": bias_squared,
        "mse1_norm": m1_norm,
        "one_shot_mse_upper_bound_norm": m1_norm + bias_squared,
        "mse_upper_bound_norm_at_shots": m1_norm / args.shots + bias_squared,
        "mse1_state": m1_state,
        "one_shot_mse_upper_bound_state": m1_state + bias_squared,
        "mse_upper_bound_state_at_shots": m1_state / args.shots + bias_squared,
        "variance_norm": m1_norm - expectation_h_hat**2,
        "variance_state": m1_state - expectation_h_hat**2,
        "reconstruction_frobenius_error": float(np.linalg.norm(reconstruction)),
        "reconstruction_spectral_error": float(np.linalg.norm(reconstruction, ord=2)),
        "reconstruction_hermiticity_frobenius": float(
            np.linalg.norm(h_hat - h_hat.conj().T)
        ),
        "elapsed_seconds": time.time() - started,
        "historical_identity_checks": {
            "configuration": "H4 R=1.2, K=15, internal nlayers=13",
            "historical_residual_frobenius": 0.8087912,
            "historical_residual_spectral": 0.1021879,
            "historical_rmse_at_2038": 0.028178580932626828,
            "note": "A new nonconvex run is compared with, not claimed identical to, the missing original run.",
        },
        "fragments": fragments,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()

