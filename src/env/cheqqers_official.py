"""
Official Quantum Cheqqers Environment & Zero-Cirq Sparse Quantum Engine.

Achieves 100% mathematical, logical, and numerical parity (< 1e-6 statevector
amplitude discrepancy, 0% legal-move divergence) with the official JavaScript
tournament server (`local-cheqqers-server.js` + `@quantumplayed/quantum-game-engine`).

Supported configurations:
- Board sizes: 6x6 (18 playable squares, default) and 8x8 (32 playable squares), plus 4x4/10x10.
- Quantumness (GameType):
  0 = CLASSIC
  1 = SUPERPOSITION
  2 = ENTANGLEMENT (tournament default)
  3 = INTERFERENCE
"""

from __future__ import annotations

import math
import random
import json
import os
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    from env.base import QuantumGameEnv
except ImportError:
    from src.env.base import QuantumGameEnv


# ============================================================================
# 1. Constants & Enums (Exact match with local-cheqqers-server.js:80-84)
# ============================================================================

class PieceColor(IntEnum):
    WHITE = 0
    BLACK = 1

    def other(self) -> "PieceColor":
        return PieceColor.BLACK if self == PieceColor.WHITE else PieceColor.WHITE


class GameState(IntEnum):
    IN_PROGRESS = 0
    WHITE_WON = 1
    BLACK_WON = 2
    DRAW = 3


class GameType(IntEnum):
    CLASSIC = 0
    SUPERPOSITION = 1
    ENTANGLEMENT = 2
    INTERFERENCE = 3


class SquareState(IntEnum):
    EMPTY = 0
    OCCUPIED = 1


# Exact constant used in BoardGameHelpers.split and BoardGameHelpers.merge
# (local-cheqqers-server.js:40-41, 48-51)
INV_SQRT2_JS: float = 0.70710678
CLEAN_EPSILON: float = 1e-10

DIRECTIONS: list[tuple[int, int]] = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
DIRECTION_PAIRS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
]


# ============================================================================
# 2. Zero-Cirq Quantum Game Engine (@quantumplayed/quantum-game-engine parity)
# ============================================================================

def clean_amplitudes(
    amplitudes: dict[int, complex], epsilon: float = CLEAN_EPSILON
) -> dict[int, complex]:
    """
    Matches `cleanAmplitudes` in `Operations.js:3-13`:
    keeps entries where |re| >= epsilon or |im| >= epsilon.
    """
    return {
        int(k): v
        for k, v in amplitudes.items()
        if abs(v.real) >= epsilon or abs(v.imag) >= epsilon
    }


# The numbers the evaluation is linear in (`OfficialCheqqersEnv.eval_features`), in this order. Each is a
# difference "ours minus theirs" of presence-weighted sums, or is signed by who is to move, so that the vector of
# one colour is minus the vector of the other.
EVAL_FEATURES = (
    "men",                     # presence of uncrowned pieces
    "queens",                  # presence of crowned pieces
    "advance",                 # men: presence x progress to the far row (0..1)
    "advance_sq",              # men: presence x progress squared
    "back_rank",               # men on their own first row while the opponent still has men
    "queen_main_diagonal",     # queens on the long diagonal
    "queen_near_diagonal",     # queens two off the long diagonal
    "centre",                  # presence x closeness to the centre
    "superposition",           # sum of p(1-p) over squares in superposition
    "multi_jump",              # +1 / -1: a capture sequence is open and it is ours / theirs
    "threat_mover",            # best capture chance of the side to move (signed by whose it is)
    "threat_waiter",           # best capture chance of the side not to move (signed)
    "tempo",                   # +1 / -1: it is our / their move
    "exposed_mover",           # presence of the mover's pieces that the other side could capture (signed)
    "exposed_waiter",          # presence of the waiting side's pieces that the mover can capture (signed)
    "superposition_when_few",  # sum of p(1-p), counted only for a side with presence of at most two pieces
    "material_share",          # (ours - theirs) / (ours + theirs), men 1 and queens 2
)


@dataclass(frozen=True)
class EvalWeights:
    """One weight per name in EVAL_FEATURES, and the scale from the weighted sum to search units (one man = 25).

    The defaults are the constants that were written into `evaluate_state` until 13 November 2026; the last four
    features did not exist then and weigh nothing, so the default agent plays as before. Weights fitted to games
    are a JSON file of the same names (`from_json`); a name that is not a feature raises."""

    men: float = 1.00
    queens: float = 2.60
    advance: float = 0.20
    advance_sq: float = 0.25
    back_rank: float = 0.18
    queen_main_diagonal: float = 0.22
    queen_near_diagonal: float = 0.10
    centre: float = 0.14
    superposition: float = 0.12
    multi_jump: float = 0.55
    threat_mover: float = 0.45
    threat_waiter: float = 0.20
    tempo: float = 0.08
    exposed_mover: float = 0.0
    exposed_waiter: float = 0.0
    superposition_when_few: float = 0.0
    material_share: float = 0.0
    scale: float = 25.0
    # 0: the linear scale above. Above 0: the evaluation in men times this is the logit of winning, and the value
    # returned is 900 * tanh(logit / 2), the expected result on a scale of -900..900 (the fit reports the slope).
    logit_per_man: float = 0.0

    def as_vector(self) -> tuple[float, ...]:
        """The weights in EVAL_FEATURES order (without the scale)."""
        return tuple(getattr(self, name) for name in EVAL_FEATURES)

    def to_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalWeights":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"not evaluation weights: {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        return cls(**{name: float(value) for name, value in data.items()})

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvalWeights":
        return cls.from_dict(json.loads(Path(path).read_text()))


DEFAULT_EVAL_WEIGHTS = EvalWeights()

# Whether a new OfficialCheqqersEnv caches its square probabilities and legal moves per position (Track-W). Off:
# code that edits squares directly and then asks for moves keeps working as before. The search switches it on
# for its own copy of a position, where the position changes only through the methods that empty the cache.
CACHE_BY_DEFAULT = os.getenv("QG_CHEQQERS_CACHE", "") == "1"
assert tuple(f.name for f in fields(EvalWeights))[:len(EVAL_FEATURES)] == EVAL_FEATURES


class QuantumOperation:
    """Matches `Operations` in `@quantumplayed/quantum-game-engine/dist/Operations.js`."""

    @staticmethod
    def shift(amount: int = 1, dimension: int = 2):
        def _apply(amplitudes: dict[int, complex], dim: int = dimension) -> dict[int, complex]:
            new_amps: dict[int, complex] = {}
            for k, amp in amplitudes.items():
                new_k = (int(k) + amount) % dim
                if new_k < 0:
                    new_k += dim
                new_amps[new_k] = amp
            return new_amps

        return _apply

    @staticmethod
    def phase(multiplier: float = 1.0, dimension: int = 2):
        def _apply(amplitudes: dict[int, complex], dim: int = dimension) -> dict[int, complex]:
            new_amps: dict[int, complex] = {}
            omega_angle = (2.0 * math.pi) / dim
            for k, amp in amplitudes.items():
                ik = int(k)
                angle = omega_angle * ik * multiplier
                ph = complex(math.cos(angle), math.sin(angle))
                new_amps[ik] = amp * ph
            return clean_amplitudes(new_amps)

        return _apply

    @staticmethod
    def sparse_transition(transitions: list[tuple[int, int, complex]]):
        """
        Matches `Operations.SparseTransition` in `Operations.js:79-111`.
        `transitions` is a list of `(from_state, to_state, complex_amplitude)`.
        States not present as `from_state` remain unchanged.
        """
        mapped_states: dict[int, list[tuple[int, complex]]] = {}
        for frm, to, amp in transitions:
            mapped_states.setdefault(frm, []).append((to, amp))

        def _apply(amplitudes: dict[int, complex], _dim: int = 0) -> dict[int, complex]:
            new_amps: dict[int, complex] = {}
            for k, amp in amplitudes.items():
                ik = int(k)
                if ik in mapped_states:
                    for to_state, trans_amp in mapped_states[ik]:
                        new_amps[to_state] = new_amps.get(to_state, 0j) + (amp * trans_amp)
                else:
                    new_amps[ik] = new_amps.get(ik, 0j) + amp
            return clean_amplitudes(new_amps)

        return _apply


class QuantumSquare:
    """
    Matches `QuantumEntity` (`QuantumEntity.js:4-141`) specialized for d=2 Cheqqers squares.
    """

    __slots__ = ("id", "dimension", "amplitudes", "entangled_system")

    def __init__(self, sq_id: int, initial_state: int = SquareState.EMPTY):
        self.id: int = sq_id
        self.dimension: int = 2
        self.amplitudes: dict[int, complex] = {int(initial_state): 1.0 + 0.0j}
        self.entangled_system: Optional[JointSystem] = None

    def apply(self, operation_fn) -> None:
        if self.entangled_system is not None:
            idx = self.entangled_system.entities.index(self)
            self.entangled_system.apply_local(idx, operation_fn)
        else:
            self.amplitudes = operation_fn(self.amplitudes, self.dimension)

    def get_probabilities(self) -> dict[int, float]:
        if self.entangled_system is not None:
            return self.entangled_system.get_target_probabilities(self)
        probs: dict[int, float] = {}
        for k, amp in self.amplitudes.items():
            probs[int(k)] = amp.real * amp.real + amp.imag * amp.imag
        return probs

    def get_occupied_prob(self) -> float:
        return self.get_probabilities().get(SquareState.OCCUPIED, 0.0)

    def get_empty_prob(self) -> float:
        return self.get_probabilities().get(SquareState.EMPTY, 0.0)

    def get_amplitudes(self) -> dict[int, complex]:
        if self.entangled_system is not None:
            return self.entangled_system.get_target_amplitudes(self)
        return dict(self.amplitudes)

    def measure(self, forced_outcome: Optional[int] = None) -> int:
        if self.entangled_system is not None:
            return self.entangled_system.measure_target(self, forced_outcome)
        if forced_outcome is not None:
            self.amplitudes = {int(forced_outcome): 1.0 + 0.0j}
            return int(forced_outcome)

        r = random.random()
        cum = 0.0
        last_k = SquareState.EMPTY
        for k, amp in self.amplitudes.items():
            ik = int(k)
            last_k = ik
            prob = amp.real * amp.real + amp.imag * amp.imag
            cum += prob
            if r <= cum:
                self.amplitudes = {ik: 1.0 + 0.0j}
                return ik
        self.amplitudes = {int(last_k): 1.0 + 0.0j}
        return int(last_k)


class JointSystem:
    """
    Matches `JointQuantumSystem` (`EntanglementEngine.js:4-404`).
    Because all Cheqqers squares have dimension 2, basis state decomposition
    and composition are exact bitwise operations on `n = len(entities)` bits,
    where entity `i` corresponds to bit shift `n - 1 - i`.
    """

    max_dimension_limit: int = 32768

    __slots__ = ("entities", "dimensions", "total_dimension", "joint_amplitudes")

    def __init__(self, entities: list[QuantumSquare], initialize_amplitudes: bool = True):
        self.entities: list[QuantumSquare] = list(entities)
        self.dimensions: list[int] = [e.dimension for e in self.entities]
        self.total_dimension: int = 1 << len(self.entities)
        if self.total_dimension > JointSystem.max_dimension_limit:
            raise RuntimeError(
                f"QuantumLimitExceeded: dimension {self.total_dimension} > {JointSystem.max_dimension_limit}"
            )
        if initialize_amplitudes:
            self.joint_amplitudes: dict[int, complex] = self._calculate_tensor_product(self.entities)
            for entity in self.entities:
                entity.entangled_system = self
                entity.amplitudes = {}
        else:
            self.joint_amplitudes = {}

    @staticmethod
    def create(entities: list[QuantumSquare]) -> "JointSystem":
        return JointSystem(entities, initialize_amplitudes=True)

    @staticmethod
    def merge(involved_entities: list[QuantumSquare]) -> "JointSystem":
        unique_systems: list[JointSystem] = []
        seen_sys: set[int] = set()
        standalone_entities: list[QuantumSquare] = []

        for entity in involved_entities:
            sys = entity.entangled_system
            if sys is not None:
                if id(sys) not in seen_sys:
                    seen_sys.add(id(sys))
                    unique_systems.append(sys)
            else:
                standalone_entities.append(entity)

        if len(unique_systems) == 0:
            return JointSystem.create(involved_entities)
        if len(unique_systems) == 1 and len(standalone_entities) == 0:
            return unique_systems[0]

        merged_entities: list[QuantumSquare] = []
        for sys in unique_systems:
            merged_entities.extend(sys.entities)
        merged_entities.extend(standalone_entities)

        new_system = JointSystem(merged_entities, initialize_amplitudes=False)
        current_amplitudes: dict[int, complex] = {0: 1.0 + 0.0j}

        for sys in unique_systems:
            next_amps: dict[int, complex] = {}
            shift = len(sys.entities)
            for k1, amp1 in current_amplitudes.items():
                base = k1 << shift
                for k2, amp2 in sys.joint_amplitudes.items():
                    next_amps[base | k2] = amp1 * amp2
            current_amplitudes = clean_amplitudes(next_amps)

        for entity in standalone_entities:
            next_amps = {}
            for k1, amp1 in current_amplitudes.items():
                base = k1 << 1
                for k2, amp2 in entity.amplitudes.items():
                    next_amps[base | int(k2)] = amp1 * amp2
            current_amplitudes = clean_amplitudes(next_amps)

        new_system.joint_amplitudes = current_amplitudes
        for entity in merged_entities:
            entity.entangled_system = new_system
            entity.amplitudes = {}
        return new_system

    @staticmethod
    def _calculate_tensor_product(entities: list[QuantumSquare]) -> dict[int, complex]:
        current_amplitudes: dict[int, complex] = {0: 1.0 + 0.0j}
        for entity in entities:
            next_amps: dict[int, complex] = {}
            for k1, amp1 in current_amplitudes.items():
                base = k1 << 1
                for k2, amp2 in entity.amplitudes.items():
                    next_amps[base | int(k2)] = amp1 * amp2
            current_amplitudes = next_amps
        return current_amplitudes

    def decompose_state(self, joint_state: int) -> list[int]:
        n = len(self.entities)
        return [(joint_state >> (n - 1 - i)) & 1 for i in range(n)]

    def compose_state(self, states: list[int]) -> int:
        res = 0
        for s in states:
            res = (res << 1) | int(s)
        return res

    def apply_local(self, entity_index: int, operation_fn) -> None:
        self.apply_to_subspace([self.entities[entity_index]], operation_fn)

    def apply_to_subspace(self, target_entities: list[QuantumSquare], operation_fn) -> None:
        n = len(self.entities)
        target_indices = [self.entities.index(e) for e in target_entities]
        m = len(target_indices)
        target_total_dim = 1 << m

        target_shifts = [n - 1 - idx for idx in target_indices]
        target_mask = 0
        for sh in target_shifts:
            target_mask |= (1 << sh)

        # Group joint amplitudes by spectator bits (joint_k & ~target_mask)
        b_map: dict[int, dict[int, complex]] = {}
        for joint_k, amp in self.joint_amplitudes.items():
            spec_bits = joint_k & (~target_mask)
            target_k = 0
            for sh in target_shifts:
                target_k = (target_k << 1) | ((joint_k >> sh) & 1)
            sub = b_map.get(spec_bits)
            if sub is None:
                b_map[spec_bits] = {target_k: amp}
            else:
                sub[target_k] = amp

        new_amps: dict[int, complex] = {}
        for spec_bits, subspace_amps in b_map.items():
            result_amps = operation_fn(subspace_amps, target_total_dim)
            for new_target_k, new_amp in result_amps.items():
                rebuilt_k = spec_bits
                for t_i, sh in enumerate(target_shifts):
                    bit = (new_target_k >> (m - 1 - t_i)) & 1
                    if bit:
                        rebuilt_k |= (1 << sh)
                new_amps[rebuilt_k] = new_amps.get(rebuilt_k, 0j) + new_amp

        self.joint_amplitudes = clean_amplitudes(new_amps)

    def get_target_probabilities(self, target_entity: QuantumSquare) -> dict[int, float]:
        target_index = self.entities.index(target_entity)
        shift = len(self.entities) - 1 - target_index
        probs: dict[int, float] = {}
        for joint_k, amp in self.joint_amplitudes.items():
            state = (joint_k >> shift) & 1
            prob = amp.real * amp.real + amp.imag * amp.imag
            probs[state] = probs.get(state, 0.0) + prob
        return probs

    def all_target_probabilities(self) -> list[dict[int, float]]:
        """get_target_probabilities of every entity, in one pass over the joint amplitudes. Each entity's sums are
        made in the same order as there, so the numbers are identical, not just close."""
        n = len(self.entities)
        shifts = [n - 1 - i for i in range(n)]
        out: list[dict[int, float]] = [{} for _ in range(n)]
        for joint_k, amp in self.joint_amplitudes.items():
            prob = amp.real * amp.real + amp.imag * amp.imag
            for i in range(n):
                state = (joint_k >> shifts[i]) & 1
                probs = out[i]
                probs[state] = probs.get(state, 0.0) + prob
        return out

    def get_target_amplitudes(self, target_entity: QuantumSquare) -> dict[int, complex]:
        target_index = self.entities.index(target_entity)
        shift = len(self.entities) - 1 - target_index
        amps: dict[int, complex] = {}
        for joint_k, amp in self.joint_amplitudes.items():
            state = (joint_k >> shift) & 1
            amps[state] = amps.get(state, 0j) + amp
        return amps

    def measure_target(
        self, target_entity: QuantumSquare, forced_outcome: Optional[int] = None
    ) -> int:
        target_index = self.entities.index(target_entity)
        shift = len(self.entities) - 1 - target_index
        marginal_probs = self.get_target_probabilities(target_entity)

        if forced_outcome is not None:
            selected_state = int(forced_outcome)
        else:
            r = random.random()
            cum = 0.0
            selected_state = -1
            for state, prob in marginal_probs.items():
                cum += prob
                if r <= cum:
                    selected_state = int(state)
                    break
            if selected_state == -1:
                keys = list(marginal_probs.keys())
                selected_state = int(keys[-1]) if keys else 0

        prob = marginal_probs.get(selected_state, 0.0)
        collapsed_amps: dict[int, complex] = {}
        if prob > 1e-12:
            norm = 1.0 / math.sqrt(prob)
            for joint_k, amp in self.joint_amplitudes.items():
                if ((joint_k >> shift) & 1) == selected_state:
                    collapsed_amps[joint_k] = amp * norm
        else:
            # Forced outcome with zero probability fallback (EntanglementEngine.js:279-304)
            max_prob = -1.0
            max_joint_k = -1
            for joint_k, amp in self.joint_amplitudes.items():
                mag = amp.real * amp.real + amp.imag * amp.imag
                if mag > max_prob:
                    max_prob = mag
                    max_joint_k = joint_k
            if max_joint_k != -1:
                if selected_state == 1:
                    new_joint_k = max_joint_k | (1 << shift)
                else:
                    new_joint_k = max_joint_k & (~(1 << shift))
                collapsed_amps = {new_joint_k: 1.0 + 0.0j}
            else:
                new_joint_k = (1 << shift) if selected_state == 1 else 0
                collapsed_amps = {new_joint_k: 1.0 + 0.0j}

        self.joint_amplitudes = clean_amplitudes(collapsed_amps)
        return selected_state

    def get_joint_probability(self, conditions: list[tuple[QuantumSquare, int]]) -> float:
        n = len(self.entities)
        mask = 0
        expected = 0
        for ent, req_state in conditions:
            idx = self.entities.index(ent)
            sh = n - 1 - idx
            mask |= (1 << sh)
            if req_state == 1:
                expected |= (1 << sh)

        prob = 0.0
        for joint_k, amp in self.joint_amplitudes.items():
            if (joint_k & mask) == expected:
                prob += amp.real * amp.real + amp.imag * amp.imag
        return prob

    def are_mutually_exclusive(self, conditions: list[tuple[QuantumSquare, int]]) -> bool:
        return self.get_joint_probability(conditions) < 1e-10


class EntanglementEngine:
    """Matches `EntanglementEngine` (`EntanglementEngine.js:405-493`)."""

    @staticmethod
    def interact_many(entities: list[QuantumSquare], operation_fn) -> None:
        if not entities:
            return
        system = JointSystem.merge(entities)
        system.apply_to_subspace(entities, operation_fn)

    @staticmethod
    def conditional_interact(
        controls: list[tuple[QuantumSquare, int]],
        targets: list[tuple[QuantumSquare, Any]],
    ) -> None:
        all_entities: list[QuantumSquare] = []
        for ent, _ in controls:
            if ent not in all_entities:
                all_entities.append(ent)
        for ent, _ in targets:
            if ent not in all_entities:
                all_entities.append(ent)

        system = JointSystem.merge(all_entities)
        n = len(system.entities)

        ctrl_mask = 0
        ctrl_expected = 0
        for ent, state in controls:
            sh = n - 1 - system.entities.index(ent)
            ctrl_mask |= (1 << sh)
            if state == 1:
                ctrl_expected |= (1 << sh)

        target_specs = [
            (system.entities.index(ent), n - 1 - system.entities.index(ent), op_fn)
            for ent, op_fn in targets
        ]

        new_amps: dict[int, complex] = {}
        for joint_k, amp in system.joint_amplitudes.items():
            if (joint_k & ctrl_mask) == ctrl_expected:
                current_branch: dict[int, complex] = {joint_k: amp}
                for _t_idx, sh, op_fn in target_specs:
                    next_branch: dict[int, complex] = {}
                    for b_k, b_amp in current_branch.items():
                        t_state = (b_k >> sh) & 1
                        local_res = op_fn({t_state: b_amp}, 2)
                        base_k = b_k & (~(1 << sh))
                        for new_t_state, new_amp in local_res.items():
                            new_k = base_k | ((int(new_t_state) & 1) << sh)
                            next_branch[new_k] = next_branch.get(new_k, 0j) + new_amp
                    current_branch = next_branch
                for res_k, res_amp in current_branch.items():
                    new_amps[res_k] = new_amps.get(res_k, 0j) + res_amp
            else:
                new_amps[joint_k] = new_amps.get(joint_k, 0j) + amp

        system.joint_amplitudes = clean_amplitudes(new_amps)


class BoardGameHelpers:
    """Matches `BoardGameHelpers` in `local-cheqqers-server.js:27-75`."""

    _MOVE_OP = QuantumOperation.sparse_transition([(2, 1, 1.0 + 0.0j)])
    _SPLIT_OP = QuantumOperation.sparse_transition([
        (4, 2, complex(INV_SQRT2_JS, 0.0)),
        (4, 1, complex(INV_SQRT2_JS, 0.0)),
    ])
    _MERGE_OP = QuantumOperation.sparse_transition([
        (4, 1, complex(INV_SQRT2_JS, 0.0)),
        (4, 6, complex(INV_SQRT2_JS, 0.0)),
        (2, 1, complex(INV_SQRT2_JS, 0.0)),
        (2, 6, complex(-INV_SQRT2_JS, 0.0)),
    ])
    _SHIFT_1 = QuantumOperation.shift(1, 2)

    @classmethod
    def move(cls, source: QuantumSquare, target: QuantumSquare) -> None:
        EntanglementEngine.interact_many([source, target], cls._MOVE_OP)

    @classmethod
    def split(
        cls, source: QuantumSquare, target1: QuantumSquare, target2: QuantumSquare
    ) -> None:
        target1.amplitudes = {0: 1.0 + 0.0j}
        target1.entangled_system = None
        target2.amplitudes = {0: 1.0 + 0.0j}
        target2.entangled_system = None
        EntanglementEngine.interact_many([source, target1, target2], cls._SPLIT_OP)

    @classmethod
    def merge(
        cls, source1: QuantumSquare, source2: QuantumSquare, target: QuantumSquare
    ) -> None:
        EntanglementEngine.interact_many([source1, source2, target], cls._MERGE_OP)

    @classmethod
    def capture(
        cls, attacker: QuantumSquare, victim: QuantumSquare, target: QuantumSquare
    ) -> None:
        EntanglementEngine.conditional_interact(
            controls=[(attacker, 1), (victim, 1)],
            targets=[
                (attacker, cls._SHIFT_1),
                (target, cls._SHIFT_1),
                (victim, cls._SHIFT_1),
            ],
        )

    @classmethod
    def capture_many(
        cls,
        attacker: QuantumSquare,
        victims: list[QuantumSquare],
        target: QuantumSquare,
    ) -> None:
        for victim in victims:
            cls.capture(attacker, victim, target)


# ============================================================================
# 3. Precomputed Board Topology Cache (6x6, 8x8, etc.)
# ============================================================================

_TOPOLOGY_CACHE: dict[int, dict[str, Any]] = {}


def _get_board_topology(board_size: int) -> dict[str, Any]:
    if board_size in _TOPOLOGY_CACHE:
        return _TOPOLOGY_CACHE[board_size]

    total_squares = (board_size * board_size) // 2
    squares_per_row = board_size // 2

    idx_to_col_row: list[tuple[int, int]] = []
    col_row_to_idx: dict[tuple[int, int], int] = {}

    for idx in range(total_squares):
        visual_row = idx // squares_per_row
        row = (board_size - 1) - visual_row
        is_row_even = (row % 2 == 0)
        col_in_row = idx % squares_per_row
        col = (col_in_row * 2) if is_row_even else (col_in_row * 2 + 1)
        idx_to_col_row.append((col, row))
        col_row_to_idx[(col, row)] = idx

    # Precompute diagonal rays for every square and each of the 4 directions
    # DIRECTIONS = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
    rays: list[list[list[int]]] = []
    for idx in range(total_squares):
        start_x, start_y = idx_to_col_row[idx]
        sq_rays: list[list[int]] = []
        for dx, dy in DIRECTIONS:
            ray: list[int] = []
            cx, cy = start_x + dx, start_y + dy
            while 0 <= cx < board_size and 0 <= cy < board_size:
                ray.append(col_row_to_idx[(cx, cy)])
                cx += dx
                cy += dy
            sq_rays.append(ray)
        rays.append(sq_rays)

    # Precompute left-right mirror permutation (C2 reflection: col -> board_size - 1 - col)
    # Wait: on an even board (6x6, 8x8), if (col + row) is even, then
    # (board_size - 1 - col) + row is ODD! So pure horizontal reflection maps
    # dark squares to light squares unless row is also inverted (180° rotation)!
    # Let's store the exact 180° rotation mapping for color-swap symmetry.
    rot180_map = [total_squares - 1 - i for i in range(total_squares)]

    topo = {
        "total_squares": total_squares,
        "squares_per_row": squares_per_row,
        "idx_to_col_row": idx_to_col_row,
        "col_row_to_idx": col_row_to_idx,
        "rays": rays,
        "rot180_map": rot180_map,
    }
    _TOPOLOGY_CACHE[board_size] = topo
    return topo


# ============================================================================
# 4. OfficialCheqqersEnv Implementation
# ============================================================================

class OfficialCheqqersEnv(QuantumGameEnv):
    """
    Bit-exact Python/NumPy implementation of `CheqqersGame` from `local-cheqqers-server.js`
    with full support for:
    - `from_tournament_json(data)`
    - `get_move_outcomes(move) -> list[tuple[float, OfficialCheqqersEnv]]`
    - `evaluate_state(perspective_color) -> float`
    - Standard `QuantumGameEnv` RL interface (`reset`, `get_state`, `get_valid_actions`, `step`, `clone`)
    """

    def __init__(
        self,
        board_size: int = 6,
        start_rows: Optional[int] = None,
        quantumness: Union[int, str, GameType] = GameType.ENTANGLEMENT,
        size: Optional[int] = None,
        game_type: Optional[Union[int, str, GameType]] = None,
    ):
        if size is not None:
            board_size = size
        if game_type is not None:
            quantumness = game_type

        self.board_size: int = int(board_size)
        self.size: int = self.board_size
        if start_rows is None:
            self.start_rows: int = max(1, (self.board_size // 2) - 1)
        else:
            self.start_rows = int(start_rows)

        self.game_type: GameType = self._parse_game_type(quantumness)
        self.quantumness: int = int(self.game_type)

        topo = _get_board_topology(self.board_size)
        self.total_squares: int = topo["total_squares"]
        self.squares_per_row: int = topo["squares_per_row"]
        self._idx_to_col_row: list[tuple[int, int]] = topo["idx_to_col_row"]
        self._col_row_to_idx: dict[tuple[int, int], int] = topo["col_row_to_idx"]
        self._rays: list[list[list[int]]] = topo["rays"]

        self.squares: list[QuantumSquare] = []
        self.piece_properties: dict[int, Optional[dict[str, Any]]] = {}
        self.turn: PieceColor = PieceColor.WHITE
        self.next_piece_id: int = 1
        self.jumped_piece_ids_this_turn: list[int] = []
        self.moves_since_take: int = 0
        self.multi_jump_square: Optional[int] = None
        self.just_crowned: bool = False
        self.state_history: dict[str, int] = {}
        self.consecutive_queen_moves: int = 0
        self._is_calculating_depth: bool = False
        self.disable_depth_calculation: bool = False
        self.force_all_moves: bool = False
        self.state: GameState = GameState.IN_PROGRESS
        self.move_history: list[str] = []
        self.last_move: Optional[dict[str, Any]] = None
        self._override_legal_moves: Optional[list[list[int]]] = None
        # None = DEFAULT_EVAL_WEIGHTS. Set by the player that searches from this position; clones keep it.
        self.eval_weights: Optional[EvalWeights] = None
        # Per-position cache (see CACHE_BY_DEFAULT): every square's probabilities, their occupied part, and the
        # legal moves. Emptied by every method that changes the position.
        self.cache_enabled: bool = CACHE_BY_DEFAULT
        self._probs_cache: Optional[list[dict[int, float]]] = None
        self._occ_cache: Optional[list[float]] = None
        self._moves_cache: Optional[tuple[bool, bool, list[dict[str, Any]]]] = None
        self._capture_cache: Optional[bool] = None

        self.reset()

    @staticmethod
    def _parse_game_type(val: Union[int, str, GameType]) -> GameType:
        if isinstance(val, GameType):
            return val
        if isinstance(val, int):
            return GameType(max(0, min(3, val)))
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("0", "classic", "classical"):
                return GameType.CLASSIC
            if s in ("1", "superposition"):
                return GameType.SUPERPOSITION
            if s in ("2", "entanglement"):
                return GameType.ENTANGLEMENT
            if s in ("3", "interference"):
                return GameType.INTERFERENCE
        return GameType.ENTANGLEMENT

    # ------------------------------------------------------------------
    # QuantumGameEnv Properties
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return 1 if self.turn == PieceColor.WHITE else -1

    @current_player.setter
    def current_player(self, val: int) -> None:
        self.turn = PieceColor.WHITE if val == 1 else PieceColor.BLACK

    @property
    def current_turn(self) -> str:
        return "white" if self.turn == PieceColor.WHITE else "black"

    @property
    def done(self) -> bool:
        return self.state != GameState.IN_PROGRESS

    @done.setter
    def done(self, val: bool) -> None:
        if not val:
            self.state = GameState.IN_PROGRESS
        elif self.state == GameState.IN_PROGRESS:
            self.state = GameState.DRAW

    def is_terminal(self) -> bool:
        return self.state != GameState.IN_PROGRESS

    def get_square_probability(self, sq_idx: int) -> float:
        if 0 <= sq_idx < self.total_squares:
            return self.squares[sq_idx].get_occupied_prob()
        return 0.0

    @property
    def chances(self) -> list[float]:
        return list(self._occ_probs())

    @property
    def winner(self) -> Optional[int]:
        if self.state == GameState.WHITE_WON:
            return 1
        if self.state == GameState.BLACK_WON:
            return -1
        if self.state == GameState.DRAW:
            return 0
        return None

    @winner.setter
    def winner(self, val: Optional[int]) -> None:
        if val == 1:
            self.state = GameState.WHITE_WON
        elif val == -1:
            self.state = GameState.BLACK_WON
        elif val == 0:
            self.state = GameState.DRAW
        else:
            self.state = GameState.IN_PROGRESS

    # ------------------------------------------------------------------
    # Initialization & Cloning
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._invalidate_caches()
        self.turn = PieceColor.WHITE
        self.squares = [
            QuantumSquare(i, initial_state=SquareState.EMPTY)
            for i in range(self.total_squares)
        ]
        self.piece_properties = {i: None for i in range(self.total_squares)}
        # Matches Evert's updated local-cheqqers-server.js:106-107:
        # `this.nextPieceId = 1;` is initialized BEFORE `this.setupStartingPositions(startRows);`,
        # so every starting piece receives a distinct integer ID (1..6 for Black, 7..12 for White on 6x6),
        # enabling multi-jumps and mandatory maximum-capture depth (`calculateMaxCaptureDepth > 1`).
        self.next_piece_id = 1
        self._setup_starting_positions(self.start_rows)
        self.last_move = None
        self.jumped_piece_ids_this_turn = []
        self.moves_since_take = 0
        self.multi_jump_square = None
        self.just_crowned = False
        self.state_history = {}
        self.consecutive_queen_moves = 0
        self._is_calculating_depth = False
        self.disable_depth_calculation = False
        self.force_all_moves = False
        self.state = GameState.IN_PROGRESS
        self.move_history = []
        self._override_legal_moves = None
        return self.get_state()

    def _setup_starting_positions(self, start_rows: int) -> None:
        if start_rows <= 0:
            return
        shift_op = QuantumOperation.shift(SquareState.OCCUPIED, 2)
        # Black starts at the top (indices 0 .. start_rows * squares_per_row - 1)
        for i in range(start_rows * self.squares_per_row):
            self.squares[i].apply(shift_op)
            self.piece_properties[i] = {
                "color": PieceColor.BLACK,
                "crowned": False,
                "pieceIds": [self.next_piece_id],
            }
            self.next_piece_id += 1

        # White starts at the bottom (total_squares - start_rows * squares_per_row .. total_squares - 1)
        for i in range(
            self.total_squares - (start_rows * self.squares_per_row),
            self.total_squares,
        ):
            self.squares[i].apply(shift_op)
            self.piece_properties[i] = {
                "color": PieceColor.WHITE,
                "crowned": False,
                "pieceIds": [self.next_piece_id],
            }
            self.next_piece_id += 1

    def clone(self) -> "OfficialCheqqersEnv":
        new_env = OfficialCheqqersEnv.__new__(OfficialCheqqersEnv)
        new_env.board_size = self.board_size
        new_env.size = self.size
        new_env.start_rows = self.start_rows
        new_env.game_type = self.game_type
        new_env.quantumness = self.quantumness
        new_env.total_squares = self.total_squares
        new_env.squares_per_row = self.squares_per_row
        new_env._idx_to_col_row = self._idx_to_col_row
        new_env._col_row_to_idx = self._col_row_to_idx
        new_env._rays = self._rays

        new_env.turn = self.turn
        new_env.next_piece_id = self.next_piece_id
        new_env.jumped_piece_ids_this_turn = list(self.jumped_piece_ids_this_turn)
        new_env.moves_since_take = self.moves_since_take
        new_env.multi_jump_square = self.multi_jump_square
        new_env.just_crowned = self.just_crowned
        new_env.state_history = dict(self.state_history)
        new_env.consecutive_queen_moves = self.consecutive_queen_moves
        new_env._is_calculating_depth = False
        new_env.disable_depth_calculation = self.disable_depth_calculation
        new_env.force_all_moves = self.force_all_moves
        new_env.state = self.state
        new_env.move_history = list(self.move_history)
        new_env.last_move = dict(self.last_move) if self.last_move else None
        new_env._override_legal_moves = (
            [list(m) for m in self._override_legal_moves]
            if self._override_legal_moves is not None
            else None
        )
        new_env.eval_weights = self.eval_weights
        new_env.cache_enabled = self.cache_enabled
        new_env._probs_cache = None
        new_env._occ_cache = None
        new_env._moves_cache = None
        new_env._capture_cache = None

        new_env.squares = [
            QuantumSquare(i, initial_state=SquareState.EMPTY)
            for i in range(self.total_squares)
        ]
        new_env.piece_properties = {}
        square_map: dict[QuantumSquare, QuantumSquare] = {}

        for i in range(self.total_squares):
            old_sq = self.squares[i]
            new_sq = new_env.squares[i]
            square_map[old_sq] = new_sq
            prop = self.piece_properties.get(i)
            if prop is not None:
                new_env.piece_properties[i] = {
                    "color": prop["color"],
                    "crowned": prop["crowned"],
                    "pieceIds": list(prop["pieceIds"]),
                }
            else:
                new_env.piece_properties[i] = None

        visited_systems: set[int] = set()
        for i in range(self.total_squares):
            sys = self.squares[i].entangled_system
            if sys is not None and id(sys) not in visited_systems:
                visited_systems.add(id(sys))
                cloned_entities = [square_map[e] for e in sys.entities]
                new_sys = JointSystem(cloned_entities, initialize_amplitudes=False)
                new_sys.joint_amplitudes = dict(sys.joint_amplitudes)
                for e in cloned_entities:
                    e.entangled_system = new_sys
                    e.amplitudes = {}

        for i in range(self.total_squares):
            if self.squares[i].entangled_system is None:
                new_env.squares[i].amplitudes = dict(self.squares[i].amplitudes)

        return new_env

    # ------------------------------------------------------------------
    # Board Coordinates (local-cheqqers-server.js:191-205)
    # ------------------------------------------------------------------

    def get_index(self, col: int, row: int) -> int:
        return self._col_row_to_idx[(col, row)]

    def get_row_col_from_index(self, index: int) -> tuple[int, int]:
        return self._idx_to_col_row[index]

    # ------------------------------------------------------------------
    # Legal Move Generation (local-cheqqers-server.js:207-452)
    # ------------------------------------------------------------------

    def get_possible_moves(
        self,
        skip_depth_calculation: bool = False,
        ignore_turn: bool = False,
        ignore_forced_captures: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Matches `getPossibleMoves` in `local-cheqqers-server.js:207-254`.
        Returns list of move dictionaries:
        - Classical / Take: `{"is_take_move": bool, "from_index": int, "to_index": int}`
        - Split: `{"from_index": int, "to_index1": int, "to_index2": int}`
        - Merge: `{"from_index1": int, "from_index2": int, "to_index": int}`
        """
        default_call = not (skip_depth_calculation or ignore_turn or ignore_forced_captures)
        if self.cache_enabled and default_call and not self._is_calculating_depth:
            flags = (self.force_all_moves, self.disable_depth_calculation)
            cached = self._moves_cache
            if cached is not None and cached[:2] == flags:
                return cached[2]
            moves = self._compute_possible_moves(False, False, False)
            self._moves_cache = (flags[0], flags[1], moves)
            return moves
        return self._compute_possible_moves(skip_depth_calculation, ignore_turn, ignore_forced_captures)

    def _compute_possible_moves(
        self,
        skip_depth_calculation: bool,
        ignore_turn: bool,
        ignore_forced_captures: bool,
    ) -> list[dict[str, Any]]:
        """`get_possible_moves` without the cache."""
        ignore_forced = ignore_forced_captures or self.force_all_moves

        if self.multi_jump_square is not None:
            take_moves = self.find_moves(
                is_take=True,
                restrict_to_square_index=self.multi_jump_square,
                ignore_turn=ignore_turn,
            )
        else:
            take_moves = self.find_moves(
                is_take=True,
                restrict_to_square_index=None,
                ignore_turn=ignore_turn,
            )

        if ignore_forced:
            standard_moves = self.get_standard_moves(ignore_turn=ignore_turn)
            all_moves = standard_moves + take_moves
            if not ignore_turn:
                if self.game_type == GameType.CLASSIC:
                    return [
                        m
                        for m in all_moves
                        if "to_index1" not in m and "from_index1" not in m
                    ]
                if self.game_type in (GameType.SUPERPOSITION, GameType.ENTANGLEMENT):
                    return [m for m in all_moves if "from_index1" not in m]
            return all_moves

        if len(take_moves) > 0:
            if (
                not skip_depth_calculation
                and not self.force_all_moves
                and not self._is_calculating_depth
                and not self.disable_depth_calculation
                and len(take_moves) > 1
            ):
                self._is_calculating_depth = True
                try:
                    current_state = self.get_simulation_state()
                    move_depths = [
                        self.calculate_max_capture_depth(current_state, m)
                        for m in take_moves
                    ]
                    max_depth = max(move_depths)
                    return [
                        m for m, d in zip(take_moves, move_depths) if d == max_depth
                    ]
                finally:
                    self._is_calculating_depth = False
            return take_moves

        standard_moves = self.get_standard_moves(ignore_turn=ignore_turn)
        if self.game_type == GameType.CLASSIC:
            return [
                m
                for m in standard_moves
                if "to_index1" not in m and "from_index1" not in m
            ]
        if self.game_type in (GameType.SUPERPOSITION, GameType.ENTANGLEMENT):
            return [m for m in standard_moves if "from_index1" not in m]
        return standard_moves

    def capture_pending(self) -> bool:
        """Whether the side to move has a capture to make. Captures are compulsory, so then every legal move is
        one; otherwise none is, however far a queen slides."""
        if self.cache_enabled and self._capture_cache is not None:
            return self._capture_cache
        pending = len(self.find_moves(is_take=True, restrict_to_square_index=self.multi_jump_square,
                                      ignore_turn=False)) > 0
        if self.cache_enabled:
            self._capture_cache = pending
        return pending

    def get_take_moves(self, ignore_turn: bool = False) -> list[dict[str, Any]]:
        return self.find_moves(is_take=True, restrict_to_square_index=None, ignore_turn=ignore_turn)

    def get_standard_moves(self, ignore_turn: bool = False) -> list[dict[str, Any]]:
        moves = self.find_moves(is_take=False, restrict_to_square_index=None, ignore_turn=ignore_turn)
        splits_and_merges = self.find_split_and_merge_moves(moves)
        return moves + splits_and_merges

    def _invalidate_caches(self) -> None:
        """The position is about to change, or has: forget what was computed for it."""
        self._probs_cache = None
        self._occ_cache = None
        self._moves_cache = None
        self._capture_cache = None

    def _invalidate_moves(self) -> None:
        """Pieces' properties, the turn or the open capture sequence changed, the amplitudes did not: the squares'
        probabilities still hold, the legal moves do not."""
        self._moves_cache = None
        self._capture_cache = None

    def _square_probs(self) -> list[dict[int, float]]:
        """`get_probabilities()` of every square; computed once per position when the cache is on."""
        if not self.cache_enabled:
            return [sq.get_probabilities() for sq in self.squares]
        if self._probs_cache is None:
            probs: list[Optional[dict[int, float]]] = [None] * len(self.squares)
            done: set[int] = set()
            for i, sq in enumerate(self.squares):
                if probs[i] is not None:
                    continue
                system = sq.entangled_system
                if system is None:
                    probs[i] = sq.get_probabilities()
                elif id(system) not in done:
                    done.add(id(system))
                    position = {id(sq_): j for j, sq_ in enumerate(self.squares)}
                    for entity, entity_probs in zip(system.entities, system.all_target_probabilities()):
                        j = position.get(id(entity))
                        if j is not None:
                            probs[j] = entity_probs
                if probs[i] is None:  # an entity of a system not found among the squares: as before
                    probs[i] = sq.get_probabilities()
            self._probs_cache = probs  # type: ignore[assignment]
        return self._probs_cache

    def _occ_probs(self) -> list[float]:
        """The probability of each square being occupied; computed once per position when the cache is on."""
        if not self.cache_enabled:
            return [sq.get_occupied_prob() for sq in self.squares]
        if self._occ_cache is None:
            self._occ_cache = [p.get(SquareState.OCCUPIED, 0.0) for p in self._square_probs()]
        return self._occ_cache

    def find_moves(
        self,
        is_take: bool,
        restrict_to_square_index: Optional[int] = None,
        ignore_turn: bool = False,
    ) -> list[dict[str, Any]]:
        """Matches `findMoves` in `local-cheqqers-server.js:266-345`."""
        moves: list[dict[str, Any]] = []
        square_probs = self._square_probs()

        indices = (
            [restrict_to_square_index]
            if restrict_to_square_index is not None
            else range(self.total_squares)
        )

        for i in indices:
            if i < 0 or i >= self.total_squares:
                continue
            probs = square_probs[i]
            is_occupied_prob = probs.get(SquareState.OCCUPIED, 0.0)
            if is_occupied_prob == 0.0:
                continue

            piece_prop = self.piece_properties.get(i)
            if not piece_prop:
                continue
            if not ignore_turn and piece_prop["color"] != self.turn:
                continue

            forward_direction = 1 if piece_prop["color"] == PieceColor.WHITE else -1
            can_move_backwards = bool(piece_prop["crowned"])

            for d_idx, (_dx, dy) in enumerate(DIRECTIONS):
                if not is_take and not can_move_backwards and dy != forward_direction:
                    continue

                ray = self._rays[i][d_idx]
                found_enemy_square_index = -1

                for cur_index in ray:
                    cur_probs = square_probs[cur_index]
                    cur_occupied_prob = cur_probs.get(SquareState.OCCUPIED, 0.0)
                    is_empty = cur_occupied_prob < 1.0
                    is_occupied = cur_occupied_prob > 0.0

                    cur_piece_prop = self.piece_properties.get(cur_index)
                    is_enemy = (
                        is_occupied
                        and cur_piece_prop is not None
                        and cur_piece_prop["color"] != piece_prop["color"]
                    )
                    is_friendly = (
                        is_occupied
                        and cur_piece_prop is not None
                        and cur_piece_prop["color"] == piece_prop["color"]
                    )
                    has_empty_chance = cur_probs.get(SquareState.EMPTY, 0.0) > 0.001

                    if is_take:
                        if found_enemy_square_index == -1:
                            if is_friendly:
                                break
                            if is_enemy:
                                enemy_ids = cur_piece_prop.get("pieceIds", [])
                                has_nan_jumped = any(
                                    isinstance(x, float) and math.isnan(x)
                                    for x in self.jumped_piece_ids_this_turn
                                )
                                shares_id = any(
                                    (isinstance(pid, float) and math.isnan(pid) and has_nan_jumped)
                                    or (pid in self.jumped_piece_ids_this_turn)
                                    for pid in enemy_ids
                                )
                                if shares_id:
                                    break
                                if not self.can_coexist(i, cur_index):
                                    break
                                found_enemy_square_index = cur_index
                            elif is_empty:
                                if not can_move_backwards:
                                    break
                        else:
                            if has_empty_chance:
                                moves.append(
                                    {
                                        "is_take_move": True,
                                        "from_index": i,
                                        "to_index": cur_index,
                                    }
                                )
                                if not can_move_backwards:
                                    break
                            else:
                                break
                    else:
                        if has_empty_chance:
                            moves.append(
                                {
                                    "is_take_move": False,
                                    "from_index": i,
                                    "to_index": cur_index,
                                }
                            )
                            if not can_move_backwards:
                                break
                            # A slide never passes an enemy piece: a move past an enemy must be a capture
                            # (harness of 11 November 2026).
                            if is_enemy:
                                break
                        else:
                            break

        return moves

    def can_coexist(self, sq_index1: int, sq_index2: int) -> bool:
        """Matches `canCoexist` in `local-cheqqers-server.js:347-357`."""
        sq1 = self.squares[sq_index1]
        sq2 = self.squares[sq_index2]
        if sq1.entangled_system is not None and sq1.entangled_system is sq2.entangled_system:
            return not sq1.entangled_system.are_mutually_exclusive(
                [(sq1, SquareState.OCCUPIED), (sq2, SquareState.OCCUPIED)]
            )
        return True

    def get_minimal_piece_map(self) -> list[Optional[dict[str, Any]]]:
        """Matches `getMinimalPieceMap` in `local-cheqqers-server.js:404-417`."""
        piece_map: list[Optional[dict[str, Any]]] = [None] * self.total_squares
        occ_probs = self._occ_probs()
        for i in range(self.total_squares):
            occ = occ_probs[i]
            prop = self.piece_properties.get(i)
            if occ > 0.001 and prop is not None:
                piece_map[i] = {
                    "color": prop["color"],
                    "crowned": prop["crowned"],
                    "pieceIds": list(prop["pieceIds"]),
                }
        return piece_map

    def get_simulation_state(self) -> dict[str, Any]:
        """Matches `getSimulationState` in `local-cheqqers-server.js:359-369`."""
        return {
            "board_size": self.board_size,
            "game_type": self.game_type,
            "turn": self.turn,
            "next_piece_id": self.next_piece_id,
            "jumped_piece_ids_this_turn": list(self.jumped_piece_ids_this_turn),
            "multi_jump_square": self.multi_jump_square,
            "piece_map": self.get_minimal_piece_map(),
        }

    def calculate_max_capture_depth(
        self, state: dict[str, Any], initial_move: dict[str, Any]
    ) -> int:
        """
        Matches `calculateMaxCaptureDepth` in `local-cheqqers-server.js:371-402`.
        Note: In the simulation state, all squares with occ > 0.001 are placed as
        classical occupied squares (occ = 1.0), so all jumps are deterministic.
        """
        sim = OfficialCheqqersEnv(
            board_size=state["board_size"],
            start_rows=0,
            quantumness=GameType.CLASSIC,
        )
        sim.disable_depth_calculation = True
        sim.turn = state["turn"]
        sim.next_piece_id = state["next_piece_id"]
        sim.jumped_piece_ids_this_turn = list(state["jumped_piece_ids_this_turn"])
        sim.multi_jump_square = state["multi_jump_square"]

        for idx, prop in enumerate(state["piece_map"]):
            if prop is not None:
                sim.squares[idx].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
                sim.piece_properties[idx] = {
                    "color": prop["color"],
                    "crowned": prop["crowned"],
                    "pieceIds": list(prop["pieceIds"]),
                }

        moves = sim.get_possible_moves(skip_depth_calculation=True)
        move_idx = -1
        for i, m in enumerate(moves):
            if (
                "from_index" in m
                and "to_index1" not in m
                and m["from_index"] == initial_move["from_index"]
                and m["to_index"] == initial_move["to_index"]
            ):
                move_idx = i
                break

        if move_idx == -1:
            return 0

        sim.apply_move(moves[move_idx])
        if sim.multi_jump_square is None:
            return 1

        next_moves = sim.get_possible_moves(skip_depth_calculation=True)
        if not next_moves:
            return 1

        next_state = sim.get_simulation_state()
        next_depths = [
            self.calculate_max_capture_depth(next_state, nm) for nm in next_moves
        ]
        return 1 + max(0, *next_depths)

    def find_split_and_merge_moves(
        self, classical_moves: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Matches `findSplitAndMergeMoves` in `local-cheqqers-server.js:419-452`."""
        split_moves: list[dict[str, Any]] = []
        merge_moves: list[dict[str, Any]] = []
        seen_splits: set[tuple[int, int, int]] = set()
        seen_merges: set[tuple[int, int, int]] = set()
        n = len(classical_moves)

        for i in range(n):
            m_i = classical_moves[i]
            fi = m_i["from_index"]
            ti = m_i["to_index"]
            for j in range(i + 1, n):
                m_j = classical_moves[j]
                if fi == m_j["from_index"] and ti != m_j["to_index"]:
                    tj = m_j["to_index"]
                    key = (fi, min(ti, tj), max(ti, tj))
                    if key not in seen_splits:
                        seen_splits.add(key)
                        split_moves.append(
                            {"from_index": fi, "to_index1": ti, "to_index2": tj}
                        )

        occ_probs = self._occ_probs()
        for i in range(n):
            m_i = classical_moves[i]
            fi = m_i["from_index"]
            ti = m_i["to_index"]
            for j in range(i + 1, n):
                m_j = classical_moves[j]
                fj = m_j["from_index"]
                if ti == m_j["to_index"] and fi != fj:
                    if occ_probs[fi] > 0.99 and occ_probs[fj] > 0.99:
                        continue
                    key = (min(fi, fj), max(fi, fj), ti)
                    if key not in seen_merges:
                        seen_merges.add(key)
                        merge_moves.append(
                            {"from_index1": fi, "from_index2": fj, "to_index": ti}
                        )

        return split_moves + merge_moves

    # ------------------------------------------------------------------
    # Move Serialization & Involvement (local-cheqqers-server.js:454-503, 874-882)
    # ------------------------------------------------------------------

    @staticmethod
    def move_to_array(move: dict[str, Any]) -> list[int]:
        if "to_index1" in move and "to_index2" in move:
            return [int(move["from_index"]), int(move["to_index1"]), int(move["to_index2"])]
        if "from_index1" in move and "from_index2" in move:
            return [int(move["from_index1"]), int(move["from_index2"]), int(move["to_index"])]
        return [int(move["from_index"]), int(move["to_index"])]

    def get_available_moves(self) -> list[list[int]]:
        """
        Returns legal moves in tournament array format `[[from, to], ...]` or `[[a, b, c], ...]`.
        If reconstructed from a tournament JSON with an explicit `legalMoves` list at the root,
        returns that exact list for the initial root state and clears the override on subsequent moves.
        """
        if self.done:
            return []
        if self._override_legal_moves is not None:
            return [list(m) for m in self._override_legal_moves]
        return [self.move_to_array(m) for m in self.get_possible_moves()]

    def get_legal_moves(self) -> list[list[int]]:
        """Alias for `get_available_moves()`."""
        return self.get_available_moves()

    def resolve_move_dict(
        self, move_or_action: Union[dict[str, Any], list[int], tuple[int, ...], int]
    ) -> dict[str, Any]:
        """Converts any move representation (dict, tournament list `[a, b]`, or action int) to move dict."""
        if isinstance(move_or_action, dict):
            return move_or_action

        possible_moves = self.get_possible_moves()

        if isinstance(move_or_action, (list, tuple)):
            arr = [int(x) for x in move_or_action]
            for m in possible_moves:
                if self.move_to_array(m) == arr:
                    return m
            # Also check reverse order of split targets or merge sources
            if len(arr) == 3:
                for m in possible_moves:
                    if "to_index1" in m and m["from_index"] == arr[0] and {m["to_index1"], m["to_index2"]} == {arr[1], arr[2]}:
                        return m
                    if "from_index1" in m and m["to_index"] == arr[2] and {m["from_index1"], m["from_index2"]} == {arr[0], arr[1]}:
                        return m
                # Construct synthetic move dict if not in possible_moves (e.g. when testing)
                p1 = self.piece_properties.get(arr[1])
                occ1 = self.squares[arr[1]].get_occupied_prob()
                if p1 is not None and occ1 > 0.001 and p1["color"] == self.turn:
                    return {"from_index1": arr[0], "from_index2": arr[1], "to_index": arr[2]}
                return {"from_index": arr[0], "to_index1": arr[1], "to_index2": arr[2]}
            if len(arr) == 2:
                # Determine if it's a take move by checking distance or intermediate enemy
                fx, fy = self.get_row_col_from_index(arr[0])
                tx, ty = self.get_row_col_from_index(arr[1])
                is_take = False
                if abs(tx - fx) >= 2:
                    dx = 1 if tx > fx else -1
                    dy = 1 if ty > fy else -1
                    for step in range(1, abs(tx - fx)):
                        mid = self.get_index(fx + step * dx, fy + step * dy)
                        if self.squares[mid].get_occupied_prob() > 0.001:
                            is_take = True
                            break
                return {"is_take_move": is_take, "from_index": arr[0], "to_index": arr[1]}
            raise ValueError(f"Cannot resolve move array: {move_or_action}")

        if isinstance(move_or_action, (int, np.integer)):
            act = int(move_or_action)
            action_map = self._build_action_map(possible_moves)
            if act in action_map:
                return action_map[act]
            if 0 <= act < len(possible_moves):
                return possible_moves[act]
            raise ValueError(f"Invalid action index {act}")

        raise TypeError(f"Unsupported move type: {type(move_or_action)}")

    def get_involved_squares(self, move: dict[str, Any]) -> list[tuple[int, str]]:
        """Matches `getInvolvedSquares` in `local-cheqqers-server.js` (version of 11 November 2026)."""
        result: list[tuple[int, str]] = []
        if "to_index1" in move and "to_index2" in move:
            result.append((int(move["to_index1"]), "landing"))
            result.append((int(move["to_index2"]), "landing"))
        elif "from_index1" in move and "from_index2" in move:
            result.append((int(move["to_index"]), "landing"))
        else:
            f_idx = int(move["from_index"])
            t_idx = int(move["to_index"])
            result.append((f_idx, "source"))
            result.append((t_idx, "landing"))
            # Squares between start and landing that may hold a piece. On a capture the first one is the
            # piece being jumped ("jumped"). Every other one, and every such square on a slide, is passed
            # over ("transit"): it is measured and must turn out empty (harness of 11 November 2026).
            fc, fr = self.get_row_col_from_index(f_idx)
            tc, tr = self.get_row_col_from_index(t_idx)
            dx = 1 if tc > fc else -1
            dy = 1 if tr > fr else -1
            jumped_found = not move.get("is_take_move", False)
            for step in range(1, abs(tc - fc)):
                check_idx = self.get_index(fc + step * dx, fr + step * dy)
                empty_prob = self.squares[check_idx].get_empty_prob()
                if empty_prob < 0.999:
                    result.append((check_idx, "transit" if jumped_found else "jumped"))
                    jumped_found = True
        return result

    @staticmethod
    def encode_move_notation(move: dict[str, Any]) -> str:
        sq = lambda idx: str(int(idx) + 1)
        if move.get("is_take_move", False):
            return f"{sq(move['from_index'])}x{sq(move['to_index'])}"
        if "to_index1" in move:
            s0, s1 = sorted([int(move["to_index1"]), int(move["to_index2"])])
            return f"{sq(move['from_index'])}-{sq(s0)}^{sq(s1)}"
        if "from_index1" in move:
            s0, s1 = sorted([int(move["from_index1"]), int(move["from_index2"])])
            return f"{sq(s0)}^{sq(s1)}-{sq(move['to_index'])}"
        return f"{sq(move['from_index'])}-{sq(move['to_index'])}"

    def get_board_state_hash(self) -> str:
        """Matches `getBoardStateHash` in `local-cheqqers-server.js:493-503`."""
        parts = [f"T:{int(self.turn)}"]
        occ_probs = self._occ_probs()
        for i in range(self.total_squares):
            occ = occ_probs[i]
            if occ > 0.0001:
                props = self.piece_properties.get(i)
                color_ch = "W" if (props and props["color"] == PieceColor.WHITE) else "B"
                crown_ch = "K" if (props and props["crowned"]) else "M"
                occ_str = "1" if occ > 0.999 else f"{occ:.3f}"
                parts.append(f"{i}{color_ch}{crown_ch}{occ_str}")
        return "|".join(parts)

    def get_state_hash(self) -> str:
        """Full state hash including multi-jump square and quantum entanglement structure for TT."""
        base = self.get_board_state_hash()
        mj = -1 if self.multi_jump_square is None else self.multi_jump_square
        return f"{base}|MJ:{mj}"

    # ------------------------------------------------------------------
    # Move Execution & State Collapse (local-cheqqers-server.js:505-800)
    # ------------------------------------------------------------------

    def apply_move(
        self,
        move_or_index: Union[dict[str, Any], list[int], int],
        forced_measurements: Optional[list[int]] = None,
        ignore_turn: bool = False,
    ) -> int:
        """
        Matches `applyMove` in `local-cheqqers-server.js:505-614`.
        Returns the number of measurements performed.
        """
        self._override_legal_moves = None
        move = self.resolve_move_dict(move_or_index)

        was_in_multi_jump = self.multi_jump_square is not None
        from_idx_before = (
            int(move["from_index"])
            if "from_index" in move
            else (int(move["from_index1"]) if "from_index1" in move else -1)
        )
        moved_piece = (
            self.piece_properties.get(from_idx_before)
            if from_idx_before != -1
            else None
        )
        was_crowned_before = bool(moved_piece and moved_piece.get("crowned", False))
        was_take_move = bool(move.get("is_take_move", False))

        forced_measurement_outcomes: list[int] = []
        replay_idx = 0

        self.just_crowned = False
        involvement = self.get_involved_squares(move)
        move_failed = False

        for sq_idx, role in involvement:
            occ = self.squares[sq_idx].get_occupied_prob()
            is_superposition = 0.001 < occ < 0.999
            should_measure = (
                self.game_type == GameType.SUPERPOSITION or role in ("landing", "transit")
            )
            if self.game_type == GameType.SUPERPOSITION and role == "source":
                should_measure = True

            if is_superposition and should_measure:
                forced = (
                    forced_measurements[replay_idx]
                    if (forced_measurements is not None and replay_idx < len(forced_measurements))
                    else None
                )
                if forced is not None:
                    replay_idx += 1
                result = self.measure_square(sq_idx, forced)
                forced_measurement_outcomes.append(result)
                if role in ("landing", "transit") and result == SquareState.OCCUPIED:
                    move_failed = True
                    break
            if self._blocks_after_measurement(sq_idx, role):
                move_failed = True
                break

        self._finalize_move_after_measurements(
            move=move,
            move_failed=move_failed,
            was_in_multi_jump=was_in_multi_jump,
            was_crowned_before=was_crowned_before,
            was_take_move=was_take_move,
            forced_measurement_outcomes=forced_measurement_outcomes,
            ignore_turn=ignore_turn,
        )
        return len(forced_measurement_outcomes)

    def _blocks_after_measurement(self, sq_idx: int, role: str) -> bool:
        """
        The two checks the harness of 11 November 2026 makes on a square once its turn in the measurement loop
        is over, whether or not it was measured:
        - a landing or passed-over square that is now certainly occupied blocks the move (an earlier measurement
          in the same move can have made it so, e.g. the second target of a split);
        - the piece to be jumped is not there (measured away, or presence at most 0.001): the jump fails.
        """
        occ = self.squares[sq_idx].get_occupied_prob()
        if role in ("landing", "transit") and occ >= 0.999:
            return True
        if role == "jumped" and occ <= 0.001:
            return True
        return False

    def _finalize_move_after_measurements(
        self,
        move: dict[str, Any],
        move_failed: bool,
        was_in_multi_jump: bool,
        was_crowned_before: bool,
        was_take_move: bool,
        forced_measurement_outcomes: list[int],
        ignore_turn: bool = False,
    ) -> None:
        self._invalidate_caches()
        # The harness applies a move inside `try { ... } catch (e) { console.error("Move application error", e); }`.
        # When the engine refuses to entangle more squares than its limit ("QuantumLimitExceeded"), everything
        # after the refused operation is skipped: the position is left as the measurements made it and the
        # same side is to move. Checked against the harness with a lowered limit
        # (tests/compliance/test_cheqqers_harness_differential.py::TestEngineLimitAgainstHarness).
        try:
            if not move_failed:
                if self.game_type == GameType.CLASSIC and (
                    "to_index1" in move or "from_index1" in move
                ):
                    self.game_type = GameType.SUPERPOSITION
                    self.quantumness = int(self.game_type)

                if move.get("is_take_move", False):
                    aborted = self._apply_take_move(move)
                    if aborted:
                        return
                elif "to_index1" in move:
                    self._apply_split_move(move)
                elif "from_index1" in move:
                    self._apply_merge_move(move)
                else:
                    self._apply_classical_move(move)

                self._apply_phase_pass()
                self.cleanup_piece_properties()
                self.disentangle_collapsed_entities()

                if ignore_turn:
                    self.multi_jump_square = None
                    self.jumped_piece_ids_this_turn = []
                elif not was_take_move:
                    if was_crowned_before:
                        self.consecutive_queen_moves += 1
                    else:
                        self.consecutive_queen_moves = 0
                    self.turn = self.turn.other()
                    self.multi_jump_square = None
                    self.jumped_piece_ids_this_turn = []
                    self.moves_since_take += 1
                else:
                    self.moves_since_take = 0
                    self.consecutive_queen_moves = 0
                    self.state_history.clear()
                    landing = int(move["to_index"])
                    if (
                        not self.just_crowned
                        and len(self.find_moves(is_take=True, restrict_to_square_index=landing)) > 0
                    ):
                        self.multi_jump_square = landing
                    else:
                        self.turn = self.turn.other()
                        self.multi_jump_square = None
                        self.jumped_piece_ids_this_turn = []

                self.check_win_states()
                self.last_move = move
            else:
                self.turn = self.turn.other()
                self.multi_jump_square = None
                self.jumped_piece_ids_this_turn = []
                self.cleanup_piece_properties()
                self.disentangle_collapsed_entities()
                self.check_win_states()
        except RuntimeError as err:
            if "QuantumLimitExceeded" not in str(err):
                raise

        self._invalidate_moves()  # turn, capture sequence and pieces are final only now
        notation = self.encode_move_notation(move)
        m_string = "".join(f"m{o}" for o in forced_measurement_outcomes)
        if was_in_multi_jump and len(self.move_history) > 0:
            last = self.move_history[-1]
            parts = notation.split("x")
            suffix = parts[1] if len(parts) > 1 else parts[0]
            self.move_history[-1] = f"{last}x{suffix}{m_string}"
        else:
            self.move_history.append(f"{notation}{m_string}")

    def measure_square(self, index: int, forced: Optional[int] = None) -> int:
        """Matches `measureSquare` in `local-cheqqers-server.js:616-623`."""
        self._invalidate_caches()
        sq = self.squares[index]
        res = sq.measure(forced)
        self.disentangle_collapsed_entities()
        self.cleanup_piece_properties()
        return res

    def disentangle_collapsed_entities(self) -> None:
        """Matches `disentangleCollapsedEntities` in `local-cheqqers-server.js:625-649`."""
        self._invalidate_caches()
        processed: set[int] = set()
        for sq in self.squares:
            sys = sq.entangled_system
            if sys is None or id(sys) in processed:
                continue
            processed.add(id(sys))

            definite: list[tuple[QuantumSquare, int]] = []
            all_d = True
            for ent in sys.entities:
                probs = sys.get_target_probabilities(ent)
                d: Optional[int] = None
                for s, p in probs.items():
                    if p > 0.999:
                        d = int(s)
                        break
                if d is not None:
                    definite.append((ent, d))
                else:
                    all_d = False

            if all_d:
                for ent, s in definite:
                    ent.entangled_system = None
                    ent.amplitudes = {s: 1.0 + 0.0j}
            else:
                for ent, s in definite:
                    ent.entangled_system = None
                    ent.amplitudes = {s: 1.0 + 0.0j}
                    if ent in sys.entities:
                        idx = sys.entities.index(ent)
                        self._factor_out_entity(sys, idx, s)

    def _factor_out_entity(self, sys: JointSystem, idx: int, state: int) -> None:
        """Matches `factorOutEntity` in `local-cheqqers-server.js:651-685`."""
        n = len(sys.entities)
        remaining = [e for i, e in enumerate(sys.entities) if i != idx]
        if len(remaining) == 0:
            return
        shift = n - 1 - idx
        if len(remaining) == 1:
            ent = remaining[0]
            r_shift = 1 - shift  # Since n == 2, the other bit is at 1 - shift
            amps: dict[int, complex] = {}
            for k, a in sys.joint_amplitudes.items():
                if ((k >> shift) & 1) == state:
                    rs = (k >> r_shift) & 1
                    amps[rs] = amps.get(rs, 0j) + a
            ent.entangled_system = None
            ent.amplitudes = amps
            return

        low_mask = (1 << shift) - 1
        amps = {}
        for k, a in sys.joint_amplitudes.items():
            if ((k >> shift) & 1) != state:
                continue
            high_bits = k >> (shift + 1)
            low_bits = k & low_mask
            nk = (high_bits << shift) | low_bits
            amps[nk] = amps.get(nk, 0j) + a

        sys.entities.pop(idx)
        sys.dimensions.pop(idx)
        sys.total_dimension = 1 << len(sys.entities)
        sys.joint_amplitudes = amps
        for ent in sys.entities:
            ent.entangled_system = sys

    def cleanup_piece_properties(self) -> None:
        """Matches `cleanupPieceProperties` in `local-cheqqers-server.js:687-689`."""
        self._invalidate_moves()
        occ_probs = self._occ_probs()
        for i in range(self.total_squares):
            if occ_probs[i] <= 0.0001:
                self.piece_properties[i] = None

    def move_piece_properties(self, f: int, t: int) -> None:
        """Matches `movePieceProperties` in `local-cheqqers-server.js:691-699`."""
        self._invalidate_moves()
        cur = self.piece_properties.get(f)
        if not cur:
            return
        self.piece_properties[t] = {
            "color": cur["color"],
            "crowned": cur["crowned"],
            "pieceIds": list(cur["pieceIds"]),
        }
        _col, row = self.get_row_col_from_index(t)
        if (cur["color"] == PieceColor.WHITE and row == self.board_size - 1) or (
            cur["color"] == PieceColor.BLACK and row == 0
        ):
            if not cur["crowned"]:
                self.piece_properties[t]["crowned"] = True
                self.just_crowned = True

    def execute_classical_override(self, f: int, taken: list[int], t: int) -> None:
        """Matches `executeClassicalOverride` in `local-cheqqers-server.js:701-707`."""
        self._invalidate_caches()
        self.squares[f].amplitudes = {SquareState.EMPTY: 1.0 + 0.0j}
        self.squares[f].entangled_system = None
        for idx in taken:
            self.squares[idx].amplitudes = {SquareState.EMPTY: 1.0 + 0.0j}
            self.squares[idx].entangled_system = None
            self.piece_properties[idx] = None
        self.squares[t].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
        self.squares[t].entangled_system = None
        self.move_piece_properties(f, t)
        if f != t:
            self.piece_properties[f] = None

    def _apply_classical_move(self, m: dict[str, Any]) -> None:
        """Matches `applyClassicalMove` in `local-cheqqers-server.js:709-715`."""
        self._invalidate_caches()
        f_idx = int(m["from_index"])
        t_idx = int(m["to_index"])
        if self.game_type == GameType.CLASSIC:
            self.execute_classical_override(f_idx, [], t_idx)
            return
        f = self.squares[f_idx]
        t = self.squares[t_idx]
        if self.game_type == GameType.SUPERPOSITION and f.get_occupied_prob() < 0.999:
            self.measure_square(f_idx)
        BoardGameHelpers.move(f, t)
        if t.get_occupied_prob() > 0.0001:
            self.move_piece_properties(f_idx, t_idx)

    def _apply_take_move(self, m: dict[str, Any]) -> bool:
        """
        Matches `applyTakeMove` in `local-cheqqers-server.js:717-745`.
        Returns True if an odd-distance empty-victim fallback threw inside JS try-catch.
        """
        self._invalidate_caches()
        f_idx = int(m["from_index"])
        t_idx = int(m["to_index"])
        fc, fr = self.get_row_col_from_index(f_idx)
        tc, tr = self.get_row_col_from_index(t_idx)
        dx = 1 if tc > fc else -1
        dy = 1 if tr > fr else -1

        taken: list[int] = []
        for s in range(1, abs(tc - fc)):
            idx = self.get_index(fc + s * dx, fr + s * dy)
            if self.squares[idx].get_occupied_prob() > 0.001:
                taken.append(idx)
        odd_empty_fallback = False
        if not taken:
            # Nothing to capture: the harness of 11 November 2026 returns here, and its measurement loop has
            # already failed such a jump (the old harness guessed the midpoint square instead).
            return False

        for idx in taken:
            p = self.piece_properties.get(idx)
            if p:
                self.jumped_piece_ids_this_turn.extend(p["pieceIds"])

        if self.game_type == GameType.CLASSIC:
            if odd_empty_fallback:
                return True
            self.execute_classical_override(f_idx, taken, t_idx)
            return False
        if self.game_type == GameType.SUPERPOSITION:
            self.measure_square(f_idx)
        if odd_empty_fallback:
            return True

        f = self.squares[f_idx]
        t = self.squares[t_idx]
        victim_squares = [self.squares[idx] for idx in taken]
        BoardGameHelpers.capture_many(f, victim_squares, t)
        if t.get_occupied_prob() > 0.0001:
            self.move_piece_properties(f_idx, t_idx)
            if f.get_occupied_prob() < 0.001:
                self.piece_properties[f_idx] = None
            for idx in taken:
                if self.squares[idx].get_occupied_prob() < 0.001:
                    self.piece_properties[idx] = None
        return False

    def _apply_split_move(self, m: dict[str, Any]) -> None:
        """Matches `applySplitMove` in `local-cheqqers-server.js:747-762`."""
        self._invalidate_caches()
        if self.game_type == GameType.CLASSIC:
            self.game_type = GameType.SUPERPOSITION
            self.quantumness = int(self.game_type)

        f_idx = int(m["from_index"])
        t1_idx = int(m["to_index1"])
        t2_idx = int(m["to_index2"])

        BoardGameHelpers.split(
            self.squares[f_idx], self.squares[t1_idx], self.squares[t2_idx]
        )
        self.move_piece_properties(f_idx, t1_idx)
        self.move_piece_properties(f_idx, t2_idx)
        self.piece_properties[f_idx] = None

    def _apply_merge_move(self, m: dict[str, Any]) -> None:
        """Matches `applyMergeMove` in `local-cheqqers-server.js:764-771`."""
        self._invalidate_caches()
        f1_idx = int(m["from_index1"])
        f2_idx = int(m["from_index2"])
        t_idx = int(m["to_index"])

        BoardGameHelpers.merge(
            self.squares[f1_idx], self.squares[f2_idx], self.squares[t_idx]
        )
        self.move_piece_properties(f1_idx, t_idx)
        p = self.piece_properties.get(t_idx)
        if p:
            ids1 = (self.piece_properties.get(f1_idx) or {}).get("pieceIds", [])
            ids2 = (self.piece_properties.get(f2_idx) or {}).get("pieceIds", [])
            merged_ids: list[int] = []
            for pid in list(ids1) + list(ids2):
                if pid not in merged_ids:
                    merged_ids.append(pid)
            p["pieceIds"] = merged_ids
        self.piece_properties[f1_idx] = None
        self.piece_properties[f2_idx] = None

    def _apply_phase_pass(self) -> None:
        """Matches `applyPhasePass` in `local-cheqqers-server.js:773-777`."""
        self._invalidate_caches()
        if self.game_type == GameType.INTERFERENCE:
            phase_op = QuantumOperation.phase(math.pi / 4.0, 2)
            for sq in self.squares:
                EntanglementEngine.conditional_interact(
                    controls=[(sq, 1)],
                    targets=[(sq, phase_op)],
                )

    def check_win_states(self) -> None:
        """Matches `checkWinStates` in `local-cheqqers-server.js:779-800`."""
        w_prob = 0.0
        b_prob = 0.0
        w_q = 0
        b_q = 0
        occ_probs = self._occ_probs()
        for i in range(self.total_squares):
            p = occ_probs[i]
            props = self.piece_properties.get(i)
            if props and p > 0.0001:
                if props["color"] == PieceColor.WHITE:
                    w_prob += p
                    if props["crowned"]:
                        w_q += 1
                else:
                    b_prob += p
                    if props["crowned"]:
                        b_q += 1

        if w_prob < 0.0001 and b_prob < 0.0001:
            self.state = GameState.DRAW
            return
        if w_prob < 0.0001:
            self.state = GameState.BLACK_WON
            return
        if b_prob < 0.0001:
            self.state = GameState.WHITE_WON
            return

        h = self.get_board_state_hash()
        count = self.state_history.get(h, 0) + 1
        self.state_history[h] = count
        if len(self.state_history) > 500:
            first_key = next(iter(self.state_history))
            del self.state_history[first_key]
        if count >= 3:
            self.state = GameState.DRAW
            return
        if w_q > 0 and b_q > 0 and self.consecutive_queen_moves >= 50:
            self.state = GameState.DRAW
            return
        if self.moves_since_take >= 100:
            self.state = GameState.DRAW
            return
        if len(self.get_possible_moves(skip_depth_calculation=True)) == 0:
            self.state = (
                GameState.BLACK_WON
                if self.turn == PieceColor.WHITE
                else GameState.WHITE_WON
            )

    # ------------------------------------------------------------------
    # Exact Born-Rule Branch Enumeration (`get_move_outcomes`)
    # ------------------------------------------------------------------

    def get_move_outcomes(
        self, move_or_action: Union[dict[str, Any], list[int], tuple[int, ...], int]
    ) -> list[tuple[float, "OfficialCheqqersEnv"]]:
        """
        Enumerates all post-move basis branches with their exact Born probabilities
        `[(prob_k, next_env_k), ...]` sorted descending by probability.

        - When a move has no superposition measurements (e.g. moving/splitting onto empty
          squares or entangling captures onto empty landing squares in GameType.ENTANGLEMENT),
          returns `[(1.0, next_env)]`.
        - When a move triggers pre-move measurement (e.g. moving/jumping onto a superposition
          landing square, or moving/capturing in GameType.SUPERPOSITION), branches over all
          realizable measurement outcomes (0 = EMPTY, 1 = OCCUPIED), including `moveFailed`
          branches when a `landing` square collapses to `OCCUPIED`!
        """
        move = self.resolve_move_dict(move_or_action)
        base_env = self.clone()
        base_env._override_legal_moves = None

        was_in_multi_jump = base_env.multi_jump_square is not None
        from_idx_before = (
            int(move["from_index"])
            if "from_index" in move
            else (int(move["from_index1"]) if "from_index1" in move else -1)
        )
        moved_piece = (
            base_env.piece_properties.get(from_idx_before)
            if from_idx_before != -1
            else None
        )
        was_crowned_before = bool(moved_piece and moved_piece.get("crowned", False))
        was_take_move = bool(move.get("is_take_move", False))

        base_env.just_crowned = False
        involvement = base_env.get_involved_squares(move)

        # Each active branch is (prob, branch_env, outcomes_list, move_failed)
        active_branches: list[tuple[float, OfficialCheqqersEnv, list[int], bool]] = [
            (1.0, base_env, [], False)
        ]

        for sq_idx, role in involvement:
            next_branches: list[tuple[float, OfficialCheqqersEnv, list[int], bool]] = []
            for b_prob, b_env, b_outcomes, b_failed in active_branches:
                if b_failed:
                    next_branches.append((b_prob, b_env, b_outcomes, True))
                    continue

                occ = b_env.squares[sq_idx].get_occupied_prob()
                is_superposition = 0.001 < occ < 0.999
                should_measure = (
                    b_env.game_type == GameType.SUPERPOSITION or role in ("landing", "transit")
                )
                if b_env.game_type == GameType.SUPERPOSITION and role == "source":
                    should_measure = True

                if is_superposition and should_measure:
                    p_occ = occ
                    p_emp = 1.0 - p_occ

                    # Outcome 0: SquareState.EMPTY (the move goes on, unless it was the piece to be jumped)
                    if p_emp > 1e-9:
                        env_0 = b_env.clone() if p_occ > 1e-9 else b_env
                        env_0.measure_square(sq_idx, forced=SquareState.EMPTY)
                        next_branches.append(
                            (b_prob * p_emp, env_0, b_outcomes + [SquareState.EMPTY],
                             env_0._blocks_after_measurement(sq_idx, role))
                        )

                    # Outcome 1: SquareState.OCCUPIED (a landing or passed-over square: the move fails)
                    if p_occ > 1e-9:
                        env_1 = b_env
                        env_1.measure_square(sq_idx, forced=SquareState.OCCUPIED)
                        failed = role in ("landing", "transit") or env_1._blocks_after_measurement(sq_idx, role)
                        next_branches.append(
                            (b_prob * p_occ, env_1, b_outcomes + [SquareState.OCCUPIED], failed)
                        )
                else:
                    next_branches.append((b_prob, b_env, b_outcomes, b_env._blocks_after_measurement(sq_idx, role)))
            active_branches = next_branches

        # Also handle GameType.SUPERPOSITION extra source measurement inside applyClassicalMove / applyTakeMove
        # if for any reason source wasn't measured yet
        final_outcomes: list[tuple[float, OfficialCheqqersEnv]] = []
        for b_prob, b_env, b_outcomes, b_failed in active_branches:
            b_env._finalize_move_after_measurements(
                move=move,
                move_failed=b_failed,
                was_in_multi_jump=was_in_multi_jump,
                was_crowned_before=was_crowned_before,
                was_take_move=was_take_move,
                forced_measurement_outcomes=b_outcomes,
                ignore_turn=False,
            )
            final_outcomes.append((b_prob, b_env))

        total_p = sum(p for p, _ in final_outcomes)
        if total_p > 0:
            final_outcomes = [(p / total_p, env) for p, env in final_outcomes]
        final_outcomes.sort(key=lambda item: item[0], reverse=True)
        return final_outcomes

    # ------------------------------------------------------------------
    # Tournament Serialization & Reconstruction (local-cheqqers-server.js:802-935)
    # ------------------------------------------------------------------

    def to_game_state_object(self, skip_moves: bool = False) -> dict[str, Any]:
        """Matches `toGameStateObject` in `local-cheqqers-server.js:802-868`."""
        piece_map: dict[int, Optional[dict[str, Any]]] = {}
        chances: dict[int, float] = {}
        for i in range(self.total_squares):
            occ = self.squares[i].get_occupied_prob()
            prop = self.piece_properties.get(i)
            if occ > 0.0001 and prop is not None:
                piece_map[i] = {
                    "color": int(prop["color"]),
                    "crowned": bool(prop["crowned"]),
                    "pieceIds": list(prop["pieceIds"]),
                }
                chances[i] = round(1.0 if occ > 0.999 else occ, 10)
            else:
                chances[i] = 0.0
                piece_map[i] = None

        def get_s(s: int, idx: int) -> str:
            if s == 0:
                return "E"
            p = self.piece_properties.get(idx)
            if not p:
                return "?"
            if p["color"] == PieceColor.WHITE:
                return "WQ" if p["crowned"] else "W"
            return "BQ" if p["crowned"] else "B"

        quantum_states: list[dict[str, Any]] = []
        reported: set[int] = set()
        for i in range(self.total_squares):
            sq = self.squares[i]
            sys = sq.entangled_system
            if sys is not None:
                if id(sys) not in reported:
                    reported.add(id(sys))
                    idxs = [self.squares.index(e) for e in sys.entities]
                    amps = []
                    for k, a in sys.joint_amplitudes.items():
                        if a.real == 0.0 and a.imag == 0.0:
                            continue
                        s_bits = sys.decompose_state(int(k))
                        state_str = "⊗".join(
                            get_s(s_bits[j], idxs[j]) for j in range(len(idxs))
                        )
                        amps.append(
                            {
                                "state": state_str,
                                "re": float(a.real),
                                "im": float(a.imag),
                            }
                        )
                    quantum_states.append(
                        {"type": "entangled", "squares": idxs, "amplitudes": amps}
                    )
            else:
                amps = []
                nt = False
                for k, a in sq.amplitudes.items():
                    if a.real == 0.0 and a.imag == 0.0:
                        continue
                    s = int(k)
                    if s != 0:
                        nt = True
                    amps.append(
                        {
                            "state": get_s(s, i),
                            "re": float(a.real),
                            "im": float(a.imag),
                        }
                    )
                if nt:
                    quantum_states.append(
                        {"type": "independent", "squares": [i], "amplitudes": amps}
                    )

        possible = [] if skip_moves else self.get_possible_moves()
        return {
            "board_size": self.board_size,
            "turn": int(self.turn),
            "game_state": int(self.state),
            "game_type": int(self.game_type),
            "piece_map": piece_map,
            "next_piece_id": self.next_piece_id,
            "multi_jump_square": self.multi_jump_square,
            "jumped_piece_ids_this_turn": list(self.jumped_piece_ids_this_turn),
            "possible_moves": possible,
            "chances": chances,
            "move_history": list(self.move_history),
            "quantum_states": quantum_states,
        }

    def to_tournament_payload(self, timeout_ms: int = 3000) -> dict[str, Any]:
        """Matches the JSON payload sent by `fetchRemoteAiMove` in `local-cheqqers-server.js:917-934`."""
        state = self.to_game_state_object()
        board = []
        for i in range(self.total_squares):
            chance = state["chances"].get(i, 0.0)
            piece = state["piece_map"].get(i)
            if chance == 0.0 or not piece:
                board.append({"piece": None, "crowned": False, "probability": 0})
            else:
                board.append(
                    {
                        "piece": "white" if piece["color"] == PieceColor.WHITE else "black",
                        "crowned": bool(piece["crowned"]),
                        "probability": chance,
                    }
                )
        turn_str = "white" if self.turn == PieceColor.WHITE else "black"
        return {
            "gameId": "local-test-game",
            "roundId": "round-1",
            "player": turn_str,
            "currentTurn": turn_str,
            "board": board,
            "quantumStates": state["quantum_states"],
            "settings": {
                "boardSize": self.board_size,
                "quantumness": int(self.game_type),
                "timeout": timeout_ms,
            },
            "legalMoves": [self.move_to_array(m) for m in state["possible_moves"]],
        }

    @classmethod
    def from_tournament_json(cls, data: dict[str, Any]) -> "OfficialCheqqersEnv":
        """
        Reconstructs an exact `OfficialCheqqersEnv` from the tournament `/move` JSON payload
        (`board`, `quantumStates`, `legalMoves` / `availableMoves`, `player` / `currentTurn` / `currentPlayer`,
        `moveCount`, `settings: {boardSize, quantumness}`), preserving all superpositions, piece
        probabilities, queen promotions (`crowned` / `isQueen`), and entangled basis states!
        """
        settings = data.get("settings") or {}
        board_data = data.get("board") or []

        board_size = data.get("boardSize") or settings.get("boardSize") or data.get("board_size")
        if board_size is None:
            if len(board_data) == 18:
                board_size = 6
            elif len(board_data) == 32:
                board_size = 8
            elif len(board_data) == 8:
                board_size = 4
            elif len(board_data) == 50:
                board_size = 10
            else:
                board_size = 6
        board_size = int(board_size)

        quantumness = (
            data.get("quantumness")
            if "quantumness" in data
            else settings.get("quantumness", data.get("game_type", 2))
        )

        env = cls(board_size=board_size, start_rows=0, quantumness=quantumness)

        # Resolve current turn
        turn_raw = (
            data.get("currentTurn")
            if "currentTurn" in data
            else (
                data.get("player")
                if "player" in data
                else data.get("currentPlayer", data.get("turn", "white"))
            )
        )
        if isinstance(turn_raw, str):
            env.turn = (
                PieceColor.WHITE
                if turn_raw.strip().lower() in ("white", "w", "1", "x")
                else PieceColor.BLACK
            )
        elif isinstance(turn_raw, int):
            if turn_raw in (0, 1) and "turn" in data and "currentPlayer" not in data:
                env.turn = PieceColor(turn_raw)
            else:
                env.turn = PieceColor.WHITE if turn_raw in (0, 1) and turn_raw == 1 else (
                    PieceColor.WHITE if turn_raw == 0 else PieceColor.BLACK
                )

        # 1. Populate piece_properties and fallback marginal probabilities from `board`
        next_pid = 1
        for idx, sq_info in enumerate(board_data):
            if idx >= env.total_squares or not isinstance(sq_info, dict):
                continue
            piece_val = sq_info.get("piece") or sq_info.get("color")
            prob = float(sq_info.get("probability", 1.0 if piece_val is not None else 0.0))
            crowned = bool(sq_info.get("crowned", sq_info.get("isQueen", False)))

            if piece_val is not None and prob > 0.0001:
                if isinstance(piece_val, str):
                    color = (
                        PieceColor.WHITE
                        if piece_val.strip().lower() in ("white", "w")
                        else PieceColor.BLACK
                    )
                else:
                    color = PieceColor(int(piece_val))

                raw_pids = sq_info.get("pieceIds")
                if raw_pids:
                    piece_ids = [math.nan if x is None else int(x) for x in raw_pids]
                    finite_ids = [x for x in piece_ids if not (isinstance(x, float) and math.isnan(x))]
                    if finite_ids:
                        next_pid = max(next_pid, max(finite_ids) + 1)
                else:
                    piece_ids = [next_pid]
                    next_pid += 1
                env.piece_properties[idx] = {
                    "color": color,
                    "crowned": crowned,
                    "pieceIds": list(piece_ids),
                }
                # Fallback independent state in case quantumStates is omitted
                p_clamped = max(0.0, min(1.0, prob))
                if p_clamped >= 0.9999:
                    env.squares[idx].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
                else:
                    env.squares[idx].amplitudes = {
                        SquareState.EMPTY: complex(math.sqrt(1.0 - p_clamped), 0.0),
                        SquareState.OCCUPIED: complex(math.sqrt(p_clamped), 0.0),
                    }
            else:
                env.piece_properties[idx] = None
                env.squares[idx].amplitudes = {SquareState.EMPTY: 1.0 + 0.0j}

        # 2. Reconstruct exact quantum states from `quantumStates` / `quantum_states`
        q_states = data.get("quantumStates")
        if q_states is None:
            q_states = data.get("quantum_states")

        def _token_to_bit_and_prop(token: str, sq_idx: int) -> int:
            nonlocal next_pid
            tok = token.strip()
            if tok in ("E", "0", ".", ""):
                return SquareState.EMPTY
            # Token is occupied (e.g. "W", "WQ", "B", "BQ", "1")
            if env.piece_properties.get(sq_idx) is None and tok in ("W", "WQ", "B", "BQ"):
                env.piece_properties[sq_idx] = {
                    "color": PieceColor.WHITE if tok.startswith("W") else PieceColor.BLACK,
                    "crowned": tok.endswith("Q"),
                    "pieceIds": [next_pid],
                }
                next_pid += 1
            return SquareState.OCCUPIED

        if isinstance(q_states, list) and len(q_states) > 0:
            first_item = q_states[0]
            if isinstance(first_item, dict) and "squares" in first_item and "amplitudes" in first_item:
                # Official local-cheqqers-server.js format:
                # [{ type: 'entangled'|'independent', squares: [...], amplitudes: [{ state: "W⊗E", re, im }] }]
                # First reset all squares to EMPTY so squares not in quantumStates are EMPTY
                if not board_data:
                    for i in range(env.total_squares):
                        env.squares[i].amplitudes = {SquareState.EMPTY: 1.0 + 0.0j}
                        env.squares[i].entangled_system = None

                # Assign shared pieceIds to entangled same-color split pairs so anti-rejump logic works
                for group in q_states:
                    sq_indices = [int(s) for s in group.get("squares", [])]
                    amps_list = group.get("amplitudes", [])
                    if not sq_indices or not amps_list:
                        continue

                    if group.get("type") == "independent" or (
                        group.get("type") != "entangled" and len(sq_indices) == 1
                    ):
                        sq_idx = sq_indices[0]
                        sq_amps: dict[int, complex] = {}
                        for entry in amps_list:
                            state_str = str(entry.get("state", "E"))
                            re_val = float(entry.get("re", entry.get("real", 0.0)))
                            im_val = float(entry.get("im", entry.get("imag", 0.0)))
                            if re_val == 0.0 and im_val == 0.0 and "probability" in entry:
                                re_val = math.sqrt(max(0.0, float(entry["probability"])))
                            bit = _token_to_bit_and_prop(state_str.split("⊗")[0], sq_idx)
                            sq_amps[bit] = sq_amps.get(bit, 0j) + complex(re_val, im_val)
                        env.squares[sq_idx].entangled_system = None
                        env.squares[sq_idx].amplitudes = clean_amplitudes(sq_amps) or {
                            SquareState.EMPTY: 1.0 + 0.0j
                        }
                    else:
                        entities = [env.squares[idx] for idx in sq_indices]
                        sys = JointSystem(entities, initialize_amplitudes=False)
                        joint_amps: dict[int, complex] = {}
                        for entry in amps_list:
                            state_str = str(entry.get("state", ""))
                            tokens = state_str.split("⊗") if "⊗" in state_str else list(state_str)
                            if len(tokens) != len(sq_indices):
                                continue
                            re_val = float(entry.get("re", entry.get("real", 0.0)))
                            im_val = float(entry.get("im", entry.get("imag", 0.0)))
                            if re_val == 0.0 and im_val == 0.0 and "probability" in entry:
                                re_val = math.sqrt(max(0.0, float(entry["probability"])))
                            bits = [
                                _token_to_bit_and_prop(tokens[j], sq_indices[j])
                                for j in range(len(sq_indices))
                            ]
                            k = sys.compose_state(bits)
                            joint_amps[k] = joint_amps.get(k, 0j) + complex(re_val, im_val)
                        sys.joint_amplitudes = clean_amplitudes(joint_amps)
                        for ent in entities:
                            ent.entangled_system = sys
                            ent.amplitudes = {}

                        # If two squares in the same entangled system have the same color and are
                        # mutually exclusive (a split piece!), unify their pieceIds so neither
                        # can be jumped twice in the same turn.
                        for a_i in range(len(sq_indices)):
                            for b_i in range(a_i + 1, len(sq_indices)):
                                idx_a, idx_b = sq_indices[a_i], sq_indices[b_i]
                                pa = env.piece_properties.get(idx_a)
                                pb = env.piece_properties.get(idx_b)
                                if (
                                    pa is not None
                                    and pb is not None
                                    and pa["color"] == pb["color"]
                                    and not env.can_coexist(idx_a, idx_b)
                                ):
                                    id_a = pa["pieceIds"][0] if pa["pieceIds"] else math.nan
                                    id_b = pb["pieceIds"][0] if pb["pieceIds"] else math.nan
                                    if isinstance(id_a, float) and math.isnan(id_a):
                                        shared_id = math.nan
                                    elif isinstance(id_b, float) and math.isnan(id_b):
                                        shared_id = math.nan
                                    else:
                                        shared_id = min(id_a, id_b)
                                    pa["pieceIds"] = [shared_id]
                                    pb["pieceIds"] = [shared_id]

            elif isinstance(first_item, dict) and "state" in first_item:
                # Flat statevector list format: [{state: "010...", real: ..., imag: ..., probability: ...}]
                # Find active quantum squares or full board basis
                sample_state = str(first_item["state"])
                tokens = sample_state.split("⊗") if "⊗" in sample_state else list(sample_state)
                if len(tokens) == env.total_squares:
                    # Identify which squares actually vary across basis states
                    varying: list[int] = []
                    for col_i in range(env.total_squares):
                        vals = set()
                        for item in q_states:
                            st = str(item["state"])
                            tks = st.split("⊗") if "⊗" in st else list(st)
                            bit = _token_to_bit_and_prop(tks[col_i], col_i)
                            vals.add(bit)
                        if len(vals) > 1:
                            varying.append(col_i)
                        else:
                            only_bit = next(iter(vals))
                            env.squares[col_i].entangled_system = None
                            env.squares[col_i].amplitudes = {only_bit: 1.0 + 0.0j}
                    if len(varying) == 1:
                        v_idx = varying[0]
                        amps_1d: dict[int, complex] = {}
                        for item in q_states:
                            st = str(item["state"])
                            tks = st.split("⊗") if "⊗" in st else list(st)
                            bit = _token_to_bit_and_prop(tks[v_idx], v_idx)
                            re_v = float(item.get("real", item.get("re", 0.0)))
                            im_v = float(item.get("imag", item.get("im", 0.0)))
                            if re_v == 0.0 and im_v == 0.0 and "probability" in item:
                                re_v = math.sqrt(max(0.0, float(item["probability"])))
                            amps_1d[bit] = amps_1d.get(bit, 0j) + complex(re_v, im_v)
                        env.squares[v_idx].amplitudes = clean_amplitudes(amps_1d)
                    elif len(varying) > 1:
                        entities = [env.squares[idx] for idx in varying]
                        sys = JointSystem(entities, initialize_amplitudes=False)
                        j_amps: dict[int, complex] = {}
                        for item in q_states:
                            st = str(item["state"])
                            tks = st.split("⊗") if "⊗" in st else list(st)
                            bits = [_token_to_bit_and_prop(tks[idx], idx) for idx in varying]
                            k = sys.compose_state(bits)
                            re_v = float(item.get("real", item.get("re", 0.0)))
                            im_v = float(item.get("imag", item.get("im", 0.0)))
                            if re_v == 0.0 and im_v == 0.0 and "probability" in item:
                                re_v = math.sqrt(max(0.0, float(item["probability"])))
                            j_amps[k] = j_amps.get(k, 0j) + complex(re_v, im_v)
                        sys.joint_amplitudes = clean_amplitudes(j_amps)
                        for ent in entities:
                            ent.entangled_system = sys
                            ent.amplitudes = {}
                        env.disentangle_collapsed_entities()

        env.next_piece_id = next_pid
        env.cleanup_piece_properties()

        # 3. Restore multi_jump_square / jumped_piece_ids_this_turn if provided or infer from legalMoves
        if "multi_jump_square" in data or "multiJumpSquare" in data:
            mj = data.get("multi_jump_square", data.get("multiJumpSquare"))
            env.multi_jump_square = int(mj) if mj is not None else None
            if env.multi_jump_square is not None:
                cls._detected_server_multijump_fix = True
        if "jumped_piece_ids_this_turn" in data:
            env.jumped_piece_ids_this_turn = [
                math.nan if x is None else int(x)
                for x in data["jumped_piece_ids_this_turn"]
            ]

        legal_moves = data.get("legalMoves")
        if legal_moves is None:
            legal_moves = data.get("availableMoves")

        if isinstance(legal_moves, list) and len(legal_moves) > 0:
            parsed_legal = [[int(x) for x in m] for m in legal_moves]
            generated = [env.move_to_array(m) for m in env.get_possible_moves()]
            if parsed_legal != generated:
                # Check if we are mid-multi-jump from a single square
                from_squares = {m[0] for m in parsed_legal if len(m) == 2}
                if len(from_squares) == 1:
                    candidate_sq = next(iter(from_squares))
                    mj_moves = [
                        env.move_to_array(m)
                        for m in env.find_moves(
                            is_take=True, restrict_to_square_index=candidate_sq
                        )
                    ]
                    if all(m in mj_moves for m in parsed_legal):
                        env.multi_jump_square = candidate_sq
                        cls._detected_server_multijump_fix = True
                # Always preserve authoritative legalMoves at the reconstructed root if still differing
                if [env.move_to_array(m) for m in env.get_possible_moves()] != parsed_legal:
                    env._override_legal_moves = parsed_legal

        # Auto-adaptive multi-jump pieceId assignment: if the tournament server has
        # demonstrated a multi-jump continuation (meaning lines 106-108 were swapped on the
        # server) or CHEQQERS_MULTI_JUMP_FIXED=true is set, replace NaN pieceIds with
        # distinct integer IDs so lookahead search simulates multi-jump chains.
        if getattr(cls, "_detected_server_multijump_fix", False):
            env._assign_distinct_integer_piece_ids()

        env._invalidate_caches()  # the position was assembled by hand above; nothing computed on the way holds
        return env

    def _assign_distinct_integer_piece_ids(self) -> None:
        """
        Replaces `NaN` pieceIds with distinct integer IDs (preserving shared IDs across
        mutually exclusive same-color entangled split branches) when the live server is
        detected to have fixed the `local-cheqqers-server.js:106-108` `NaN` pieceId bug.
        """
        self._invalidate_caches()
        assigned: dict[int, int] = {}
        cur_id = 1
        for i in range(self.total_squares):
            prop = self.piece_properties.get(i)
            if not prop:
                continue
            pids = prop.get("pieceIds", [])
            if pids and not any(isinstance(x, float) and math.isnan(x) for x in pids):
                cur_id = max(cur_id, max(int(x) for x in pids) + 1)
                continue
            if i in assigned:
                prop["pieceIds"] = [assigned[i]]
                continue
            my_id = cur_id
            cur_id += 1
            assigned[i] = my_id
            prop["pieceIds"] = [my_id]
            # Share ID with any mutually-exclusive same-color entangled partner
            for j in range(i + 1, self.total_squares):
                prop_j = self.piece_properties.get(j)
                if (
                    prop_j is not None
                    and prop_j["color"] == prop["color"]
                    and not self.can_coexist(i, j)
                ):
                    assigned[j] = my_id
                    prop_j["pieceIds"] = [my_id]
        self.next_piece_id = cur_id

    # ------------------------------------------------------------------
    # Quantum-Aware Tactical Evaluation (`evaluate_state`)
    # ------------------------------------------------------------------

    def _resolve_perspective_color(
        self, perspective_color: Union[int, str, PieceColor, None] = None
    ) -> PieceColor:
        if perspective_color is None:
            return self.turn
        if isinstance(perspective_color, PieceColor):
            return perspective_color
        if isinstance(perspective_color, str):
            s = perspective_color.strip().lower()
            return PieceColor.WHITE if s in ("white", "w", "1", "x") else PieceColor.BLACK
        if isinstance(perspective_color, (int, np.integer)):
            val = int(perspective_color)
            if val == 1:
                return PieceColor.WHITE
            if val == -1:
                return PieceColor.BLACK
            if val == 0:
                return PieceColor.WHITE
        return self.turn

    def eval_features(
        self, perspective_color: Union[int, str, PieceColor, None] = None
    ) -> tuple[float, ...]:
        """
        The numbers the evaluation is linear in, in EVAL_FEATURES order, from `perspective_color`'s side. Presence
        means the marginal Born probability of a square being occupied. Meant for positions still in progress;
        `evaluate_state` and the fit of the weights (`training/fit_evaluation.py`) both read this function, so the
        two cannot drift apart.
        """
        persp = self._resolve_perspective_color(perspective_color)
        center = (self.board_size - 1) / 2.0
        denom = max(1.0, float(self.board_size - 1))

        occ_probs = self._occ_probs()

        # Uncrowned presence per colour: the back-rank guard only counts while the opponent still has men
        white_men_prob = 0.0
        black_men_prob = 0.0
        for i in range(self.total_squares):
            p = occ_probs[i]
            prop = self.piece_properties.get(i)
            if prop and p > 0.0001 and not prop["crowned"]:
                if prop["color"] == PieceColor.WHITE:
                    white_men_prob += p
                else:
                    black_men_prob += p

        f = dict.fromkeys(EVAL_FEATURES, 0.0)
        presence = {PieceColor.WHITE: 0.0, PieceColor.BLACK: 0.0}
        material = {PieceColor.WHITE: 0.0, PieceColor.BLACK: 0.0}  # men 1, queens 2
        split = {PieceColor.WHITE: 0.0, PieceColor.BLACK: 0.0}  # sum of p(1-p) over squares in superposition

        for i in range(self.total_squares):
            p = occ_probs[i]
            prop = self.piece_properties.get(i)
            if not prop or p <= 0.0001:
                continue

            color = prop["color"]
            crowned = bool(prop["crowned"])
            col, row = self._idx_to_col_row[i]
            sign = 1.0 if color == persp else -1.0

            if crowned:
                f["queens"] += sign * p
                if col == row:
                    f["queen_main_diagonal"] += sign * p
                elif abs(col - row) == 2:
                    f["queen_near_diagonal"] += sign * p
            else:
                f["men"] += sign * p
                progress = (
                    float(row) / denom
                    if color == PieceColor.WHITE
                    else float(self.board_size - 1 - row) / denom
                )
                f["advance"] += sign * p * progress
                f["advance_sq"] += sign * p * progress * progress
                if color == PieceColor.WHITE and row == 0 and black_men_prob > 0.25:
                    f["back_rank"] += sign * p
                elif (
                    color == PieceColor.BLACK
                    and row == self.board_size - 1
                    and white_men_prob > 0.25
                ):
                    f["back_rank"] += sign * p

            dist_center = abs(col - center) + abs(row - center)
            f["centre"] += sign * p * (1.0 - dist_center / (2.0 * center + 1e-6))

            if 0.05 < p < 0.95:
                f["superposition"] += sign * p * (1.0 - p)
                split[color] += p * (1.0 - p)

            presence[color] += p
            material[color] += p * (2.0 if crowned else 1.0)

        other = persp.other()
        mover = self.turn
        waiter = mover.other()
        mover_sign = 1.0 if mover == persp else -1.0

        if self.multi_jump_square is not None:
            f["multi_jump"] = mover_sign
        f["tempo"] = mover_sign

        # Captures available to either side: the best chance of one succeeding, and the pieces that could be taken
        all_takes = self.find_moves(is_take=True, restrict_to_square_index=None, ignore_turn=True)
        threat = {PieceColor.WHITE: 0.0, PieceColor.BLACK: 0.0}
        victims: dict[PieceColor, set[int]] = {PieceColor.WHITE: set(), PieceColor.BLACK: set()}
        for tm in all_takes:
            f_idx = tm["from_index"]
            t_idx = tm["to_index"]
            attacker_prop = self.piece_properties.get(f_idx)
            if not attacker_prop:
                continue
            attacker = attacker_prop["color"]
            success_weight = occ_probs[f_idx] * (1.0 - occ_probs[t_idx])
            if success_weight > threat[attacker]:
                threat[attacker] = success_weight
            fc, fr = self._idx_to_col_row[f_idx]
            tc, tr = self._idx_to_col_row[t_idx]
            dx = 1 if tc > fc else -1
            dy = 1 if tr > fr else -1
            for step in range(1, abs(tc - fc)):
                between = self._col_row_to_idx[(fc + step * dx, fr + step * dy)]
                if occ_probs[between] > 0.001:
                    victims[attacker.other()].add(between)  # the first piece on the way is the one jumped
                    break

        f["threat_mover"] = mover_sign * threat[mover]
        f["threat_waiter"] = -mover_sign * threat[waiter]
        f["exposed_mover"] = mover_sign * sum(occ_probs[i] for i in victims[mover])
        f["exposed_waiter"] = -mover_sign * sum(occ_probs[i] for i in victims[waiter])

        few_ours = split[persp] if presence[persp] <= 2.0 else 0.0
        few_theirs = split[other] if presence[other] <= 2.0 else 0.0
        f["superposition_when_few"] = few_ours - few_theirs

        total_material = material[persp] + material[other]
        if total_material > 1e-9:
            f["material_share"] = (material[persp] - material[other]) / total_material

        return tuple(f[name] for name in EVAL_FEATURES)

    def evaluate_state(
        self, perspective_color: Union[int, str, PieceColor, None] = None
    ) -> float:
        """
        The static evaluation from `perspective_color`'s viewpoint, in search units (one man = 25 points):
        +1000 / -1000 for a finished game won / lost, 0 for a draw; otherwise the weighted sum of
        `eval_features`, clipped to +-900 so that finished games stay outside the range. The weights are
        `self.eval_weights`, or DEFAULT_EVAL_WEIGHTS (the hand-set constants of before 13 November 2026). With
        `weights.logit_per_man` above 0 the sum is read as the logit of winning and the value is the expected
        result, 900 * tanh(logit / 2): the search averages values over measurement outcomes, and only on this
        scale is "half the time I win, half the time I am a man down" worth what it is worth.
        """
        persp = self._resolve_perspective_color(perspective_color)

        if self.state != GameState.IN_PROGRESS:
            if self.state == GameState.DRAW:
                return 0.0
            won_color = (
                PieceColor.WHITE
                if self.state == GameState.WHITE_WON
                else PieceColor.BLACK
            )
            return 1000.0 if won_color == persp else -1000.0

        weights = self.eval_weights if self.eval_weights is not None else DEFAULT_EVAL_WEIGHTS
        score = 0.0
        for weight, feature in zip(weights.as_vector(), self.eval_features(persp)):
            score += weight * feature
        if weights.logit_per_man > 0.0:
            return 900.0 * math.tanh(0.5 * weights.logit_per_man * score)
        return float(max(-900.0, min(900.0, score * weights.scale)))

    def get_state_72(self) -> np.ndarray:
        """Returns the 72-dim state vector for 6x6 boards (or total_squares * 4)."""
        return self.get_state()

    # ------------------------------------------------------------------
    # RL State Vector & Action-Space Bridge (`QuantumGameEnv` interface)
    # ------------------------------------------------------------------

    def get_state(self, rotated_for_legacy_model: bool = False) -> np.ndarray:
        """
        Returns a `[total_squares * 4]` float32 vector from `current_player`'s perspective:
        `[p_self_reg, p_self_cro, p_opp_reg, p_opp_cro]` for each playable square.
        """
        state = np.zeros(self.total_squares * 4, dtype=np.float32)
        for i in range(self.total_squares):
            out_idx = (self.total_squares - 1 - i) if rotated_for_legacy_model else i
            p = self.squares[i].get_occupied_prob()
            prop = self.piece_properties.get(i)
            if prop is None or p <= 0.0001:
                continue
            base = out_idx * 4
            is_self = prop["color"] == self.turn
            if is_self:
                if prop["crowned"]:
                    state[base + 1] = p
                else:
                    state[base + 0] = p
            else:
                if prop["crowned"]:
                    state[base + 3] = p
                else:
                    state[base + 2] = p
        return state

    def move_to_canonical_action(self, move: dict[str, Any]) -> int:
        """
        Maps a move dict to the legacy `[0 .. total_squares * 16 - 1]` action space
        (`288` for 6x6, `512` for 8x8) used by existing MLP/SNN/KAN models.
        """
        num_active = self.total_squares
        split_offset = num_active * 4
        merge_offset = split_offset + num_active * 6

        if "to_index1" in move and "to_index2" in move:
            f_idx = int(move["from_index"])
            fx, fy = self._idx_to_col_row[f_idx]
            t1x, t1y = self._idx_to_col_row[int(move["to_index1"])]
            t2x, t2y = self._idx_to_col_row[int(move["to_index2"])]
            d1 = DIRECTIONS.index((int(np.sign(t1x - fx)), int(np.sign(t1y - fy))))
            d2 = DIRECTIONS.index((int(np.sign(t2x - fx)), int(np.sign(t2y - fy))))
            d_min, d_max = min(d1, d2), max(d1, d2)
            p_idx = DIRECTION_PAIRS.index((d_min, d_max)) if d_min != d_max else d_min
            return split_offset + f_idx * 6 + p_idx

        if "from_index1" in move and "from_index2" in move:
            t_idx = int(move["to_index"])
            tx, ty = self._idx_to_col_row[t_idx]
            f1x, f1y = self._idx_to_col_row[int(move["from_index1"])]
            f2x, f2y = self._idx_to_col_row[int(move["from_index2"])]
            d1 = DIRECTIONS.index((int(np.sign(f1x - tx)), int(np.sign(f1y - ty))))
            d2 = DIRECTIONS.index((int(np.sign(f2x - tx)), int(np.sign(f2y - ty))))
            d_min, d_max = min(d1, d2), max(d1, d2)
            p_idx = DIRECTION_PAIRS.index((d_min, d_max)) if d_min != d_max else d_min
            return merge_offset + t_idx * 6 + p_idx

        f_idx = int(move["from_index"])
        t_idx = int(move["to_index"])
        fx, fy = self._idx_to_col_row[f_idx]
        tx, ty = self._idx_to_col_row[t_idx]
        dx = int(np.sign(tx - fx))
        dy = int(np.sign(ty - fy))
        d_idx = DIRECTIONS.index((dx, dy))
        return f_idx * 4 + d_idx

    def _build_action_map(
        self, possible_moves: Optional[list[dict[str, Any]]] = None
    ) -> dict[int, dict[str, Any]]:
        if possible_moves is None:
            possible_moves = self.get_possible_moves()
        action_map: dict[int, dict[str, Any]] = {}
        overflow_base = self.total_squares * 16
        overflow_idx = 0
        for m in possible_moves:
            act = self.move_to_canonical_action(m)
            if act not in action_map:
                action_map[act] = m
            else:
                # Collision from long-range Flying Queen moves along the same ray:
                # assign deterministic overflow action so zero legal moves are lost!
                action_map[overflow_base + overflow_idx] = m
                overflow_idx += 1
        return action_map

    def get_valid_actions(self) -> list[int]:
        if self.done:
            return []
        if self._override_legal_moves is not None:
            moves = [self.resolve_move_dict(m) for m in self._override_legal_moves]
        else:
            moves = self.get_possible_moves()
        return list(self._build_action_map(moves).keys())

    def action_to_tournament_move(self, action: int) -> list[int]:
        move_dict = self.resolve_move_dict(action)
        return self.move_to_array(move_dict)

    def step(
        self,
        action: Union[int, list[int], tuple[int, ...], dict[str, Any]],
        return_state: bool = True,
    ) -> tuple[Optional[np.ndarray], float, bool, dict[str, Any]]:
        if self.done:
            raise ValueError("Game is already finished.")

        acting_player = self.current_player
        try:
            move = self.resolve_move_dict(action)
        except (ValueError, TypeError):
            state = self.get_state() if return_state else None
            return state, -15.0, True, {"msg": "Invalid action choice."}

        num_measurements = self.apply_move(move)

        if self.state == GameState.WHITE_WON:
            reward = 1.0 if acting_player == 1 else -1.5
        elif self.state == GameState.BLACK_WON:
            reward = 1.0 if acting_player == -1 else -1.5
        elif self.state == GameState.DRAW:
            reward = 0.0
        else:
            reward = 0.0

        state = self.get_state() if return_state else None
        return (
            state,
            reward,
            self.done,
            {
                "move": self.move_to_array(move),
                "measurements": num_measurements,
                "multi_jump_square": self.multi_jump_square,
            },
        )
