"""Count native iSWAP and compiled CNOT resources without gate substitution.

Counts exclude state preparation, one-qubit gates, routing and readout.
A combined two-qubit count assigns one unit to either gate; it is not a
CNOT-equivalent count or a hardware-calibrated duration/error estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Iterable

GATE_RESOURCE_CONVENTION = "native_iswap_and_compiled_cnot_v1"
CX_GIVENS = 2
CX_POOL = 3
CX_PAIR_TRANSFER_BOUND = 100


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class GateResources:
    """Two-qubit resources for one executable measurement setting."""

    cnot_count: int = 0
    iswap_count: int = 0
    two_qubit_depth: int = 0
    is_upper_bound: bool = False

    def __post_init__(self) -> None:
        for name in ("cnot_count", "iswap_count", "two_qubit_depth"):
            _nonnegative_integer(getattr(self, name), name)

    @property
    def two_qubit_count(self) -> int:
        return self.cnot_count + self.iswap_count

    def as_dict(self) -> dict:
        return {**asdict(self), "two_qubit_count": self.two_qubit_count,
                "gate_resource_convention": GATE_RESOURCE_CONVENTION}


def gpd_gate_resources(
    n_qubits: int, paper_depth: int, odd_qubit_mode: str = "reject"
) -> GateResources:
    """Periodic GPD: n/2 native iSWAPs per layer, entangling depth L.

    The explicit odd-n open-chain extension has floor(n/2) pairs in each
    layer. A single qubit has no entangling gates or entangling depth.
    """
    n = _nonnegative_integer(n_qubits, "n_qubits")
    depth = _nonnegative_integer(paper_depth, "paper_depth")
    if n < 1:
        raise ValueError("n_qubits must be positive")
    if odd_qubit_mode not in {"reject", "open-chain"}:
        raise ValueError("odd_qubit_mode must be 'reject' or 'open-chain'")
    if n > 1 and n % 2 and odd_qubit_mode != "open-chain":
        raise ValueError("periodic iSWAP GPD requires an even qubit count")
    return GateResources(iswap_count=(n // 2) * depth,
                         two_qubit_depth=depth if n > 1 else 0)


def source_gate_resources(
    family: str, depth: int, n_qubits: int = 8
) -> GateResources:
    """Legacy AGPD source resources, retaining its open-chain iSWAP layout.

    The other families retain their existing CNOT compiler conventions:
    two CNOTs per Givens, three per pool block, and a conservative 100-CNOT
    bound for the four-qubit pair-transfer block.
    """
    n = _nonnegative_integer(n_qubits, "n_qubits")
    depth = _nonnegative_integer(depth, "depth")
    if n < 2:
        raise ValueError("legacy AGPD sources require at least two qubits")
    m = n // 2
    if family == "gfro":
        return GateResources(cnot_count=2 * depth * (m - 1) * CX_GIVENS,
                             two_qubit_depth=2 * depth * CX_GIVENS)
    if family == "operator_pool":
        return GateResources(cnot_count=depth * (n // 2) * CX_POOL,
                             two_qubit_depth=depth * CX_POOL, is_upper_bound=True)
    if family == "nnk_uccgsdi":
        count = depth * (m - 1) * (2 * CX_GIVENS + CX_PAIR_TRANSFER_BOUND)
        cnot_depth = depth * (2 * CX_GIVENS + (m - 1) * CX_PAIR_TRANSFER_BOUND)
        return GateResources(cnot_count=count, two_qubit_depth=cnot_depth,
                             is_upper_bound=True)
    if family == "shallow_iswap_su2":
        layer_counts = [n // 2 if layer % 2 == 0 else (n - 1) // 2
                        for layer in range(depth)]
        return GateResources(iswap_count=sum(layer_counts),
                             two_qubit_depth=sum(count > 0 for count in layer_counts))
    raise KeyError(f"Unknown AGPD family: {family}")


def summarize_gate_resources(
    resources: Iterable[GateResources], allocation: Iterable[int] | None
) -> dict:
    """Count every executed setting shot and preserve both gate types.

    allocation=None means that a target is unreachable/unreported, not zero
    gates. An empty setting list with an empty allocation executes no gates.
    """
    settings = list(resources)
    cnot_budget = iswap_budget = total_budget = None
    if allocation is not None:
        shots = [_nonnegative_integer(value, "shots") for value in allocation]
        if len(shots) != len(settings):
            raise ValueError("allocation must contain one entry per setting")
        cnot_budget = sum(t * item.cnot_count for t, item in zip(shots, settings))
        iswap_budget = sum(t * item.iswap_count for t, item in zip(shots, settings))
        total_budget = cnot_budget + iswap_budget
    upper = any(item.is_upper_bound for item in settings)
    return {
        "gate_resource_convention": GATE_RESOURCE_CONVENTION,
        "max_two_qubit_depth": max((item.two_qubit_depth for item in settings), default=0),
        "depth_is_upper_bound": upper,
        "two_qubit_gate_budget": total_budget,
        "cnot_gate_budget": cnot_budget,
        "iswap_gate_budget": iswap_budget,
        "gate_budget_is_upper_bound": upper,
    }

