"""
Adversarial Compliance & Parity Test Suite for OfficialTiqTaqToeEnv,
OfficialCheqqersEnv, ChanceQMCTSPlayer, and ExpectiCheqqersPlayer.

Verifies 100% mathematical, logical, and numerical parity against the
official Node.js tournament servers (`local-tiqtaqtoe-server.js` and
`local-cheqqers-server.js` + `@quantumplayed/quantum-game-engine`).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from env.tiqtaqtoe_official import (
    OfficialTiqTaqToeEnv,
    TicTacResult,
    TicTacRules,
    derive_entanglements,
)
from env.cheqqers_official import (
    OfficialCheqqersEnv,
    PieceColor,
    GameState,
    GameType,
    SquareState,
    INV_SQRT2_JS,
)
from search.chance_q_mcts import ChanceQMCTSPlayer
from search.expecti_cheqqers import ExpectiCheqqersPlayer


# ==============================================================================
# 1. OfficialTiqTaqToeEnv Parity Tests
# ==============================================================================


def test_tiqtaqtoe_initial_state_and_available_moves():
    env = OfficialTiqTaqToeEnv(quantumness=3, decoherence=0, round_number=1)
    assert env.current_turn == "X"
    assert env.current_player == 1
    assert not env.done
    assert env.winner is None

    moves = env.get_available_moves(deduplicate_measurements=False)
    # 9 classical moves [i] + 36 quantum moves [i, j] (i < j) + 0 measurement moves
    assert len(moves) == 45
    assert [0] in moves
    assert [0, 1] in moves
    assert [0, "M"] not in moves


def test_tiqtaqtoe_sqrtswap_amplitudes_and_entanglement():
    env = OfficialTiqTaqToeEnv(quantumness=3)
    # Move 1: X plays quantum [0, 1]
    # Shift(0, 1) -> X00000000, then SqrtSwap(0, 1):
    # c_stay = 0.5 + 0.5j on X00000000, c_swap = 0.5 - 0.5j on 0X0000000
    env.apply_move([0, 1])
    assert env.current_turn == "O"
    assert len(env.quantum_state) == 2
    amp_x0 = env.quantum_state["X00000000"]
    amp_0x = env.quantum_state["0X0000000"]
    assert math.isclose(amp_x0.real, 0.5, abs_tol=1e-6)
    assert math.isclose(amp_x0.imag, 0.5, abs_tol=1e-6)
    assert math.isclose(amp_0x.real, 0.5, abs_tol=1e-6)
    assert math.isclose(amp_0x.imag, -0.5, abs_tol=1e-6)

    # Squares 0 and 1 must be in the same entanglement pair
    assert env.pairs[0] == [0, 1]
    assert env.square_to_pair[0] == 0
    assert env.square_to_pair[1] == 0

    # At quantumness=3, O CAN entangle with X's half-occupied square 1 and empty square 2!
    assert env.is_legal_move([1, 2])
    # O CANNOT play classical [0] because square 0 is not empty
    assert not env.is_legal_move([0])
    # O CAN measure [0, "M"] or [1, "M"]
    assert env.is_legal_move([0, "M"])
    assert env.is_legal_move([1, "M"])


def test_tiqtaqtoe_from_tournament_payload_and_move_outcomes():
    env = OfficialTiqTaqToeEnv(quantumness=3)
    env.apply_move([0, 1])  # X in 0, 1
    env.apply_move([1, 2])  # O entangles 1, 2

    payload = env.to_tournament_payload()
    reconstructed = OfficialTiqTaqToeEnv.from_tournament_payload(payload)

    assert reconstructed.current_turn == env.current_turn
    assert reconstructed.pairs == env.pairs
    assert reconstructed.measured_squares == env.measured_squares
    assert reconstructed.empty_squares == env.empty_squares
    for k, v in env.quantum_state.items():
        assert k in reconstructed.quantum_state
        assert abs(reconstructed.quantum_state[k] - v) < 1e-6

    # Enumerate outcomes of measuring square 0 ([0, "M"])
    outcomes = reconstructed.get_move_outcomes([0, "M"])
    assert len(outcomes) >= 2
    total_prob = sum(p for p, _ in outcomes)
    assert math.isclose(total_prob, 1.0, abs_tol=1e-6)
    for p, child_env in outcomes:
        assert p > 0.0
        # Connected component {0, 1, 2} was measured
        assert child_env.measured_squares[0] or child_env.empty_squares[0]


def test_tiqtaqtoe_many_worlds_and_simultaneous_win_both():
    env = OfficialTiqTaqToeEnv(quantumness=3)
    # Construct a quantum state where branch 1 is X winning (XXXOO0000) and branch 2 is BOTH winning (XXXOOO000)
    s2 = 1.0 / math.sqrt(2.0)
    env.quantum_state = {
        "XXXOO0000": complex(s2, 0.0),
        "XXXOOO000": complex(s2, 0.0),
    }
    stats = env.compute_many_worlds()
    assert math.isclose(stats["xWinProb"], 0.5, abs_tol=1e-6)
    assert math.isclose(stats["bothProb"], 0.5, abs_tol=1e-6)
    assert math.isclose(stats["oWinProb"], 0.0, abs_tol=1e-6)
    # In official tournament scoring, BOTH gives +1 point to X and +1 point to O
    assert math.isclose(stats["xExpectedPoints"], 1.0, abs_tol=1e-6)
    assert math.isclose(stats["oExpectedPoints"], 0.5, abs_tol=1e-6)


def test_tiqtaqtoe_d4_symmetry_equivariance():
    env = OfficialTiqTaqToeEnv(quantumness=3)
    env.apply_move([0, 4])
    env.apply_move([2])

    base_state = env.get_state()
    assert base_state.shape == (111,)

    for sym_idx in range(8):
        sym_env = env.apply_d4_symmetry(sym_idx)
        sym_state_from_env = sym_env.get_state()
        sym_state_direct = OfficialTiqTaqToeEnv.transform_state_111_d4(
            base_state, sym_idx, inverse=False
        )
        np.testing.assert_allclose(sym_state_from_env, sym_state_direct, atol=1e-5)

        # Applying inverse symmetry recovers original state and quantum amplitudes
        recovered_env = sym_env.apply_d4_symmetry(sym_idx, inverse=True)
        np.testing.assert_allclose(recovered_env.get_state(), base_state, atol=1e-5)
        for k, amp in env.quantum_state.items():
            assert abs(recovered_env.quantum_state[k] - amp) < 1e-6


def test_chance_q_mcts_player_tactical_and_legal_move():
    # Setup a board where X has measured X at squares 0 and 1, and square 2 is empty.
    # X to move must immediately find the winning classical move [2]!
    env = OfficialTiqTaqToeEnv(quantumness=3)
    env.apply_move([0])  # X at 0 (unmeasured classical)
    env.apply_move([3])  # O at 3
    env.apply_move([1])  # X at 1
    env.apply_move([4])  # O at 4
    # Force squares 0 and 1 to be measured X so [2] is a 1-move win if measured or via MCTS
    env.measured_squares[0] = True
    env.measured_squares[1] = True

    player = ChanceQMCTSPlayer(
        model=None,
        c_puct=1.5,
        num_searches=64,
        time_limit_ms=500.0,
        use_d4_symmetry=True,
    )
    legal_moves = env.get_available_moves(deduplicate_measurements=False)
    move = player.get_move(env, available_moves=legal_moves)
    assert move in legal_moves


# ==============================================================================
# 2. OfficialCheqqersEnv & ExpectiCheqqersPlayer Parity Tests
# ==============================================================================


def test_cheqqers_initial_setup_and_legal_moves():
    env = OfficialCheqqersEnv(board_size=6, quantumness=GameType.ENTANGLEMENT)
    assert env.total_squares == 18
    assert env.turn == PieceColor.WHITE

    # White occupies bottom 2 rows (squares 12..17), Black occupies top 2 rows (squares 0..5)
    for sq in range(0, 6):
        assert env.piece_properties[sq] is not None
        assert env.piece_properties[sq]["color"] == PieceColor.BLACK
        assert math.isclose(env.squares[sq].get_occupied_prob(), 1.0, abs_tol=1e-6)
    for sq in range(6, 12):
        assert env.piece_properties[sq] is None
        assert math.isclose(env.squares[sq].get_occupied_prob(), 0.0, abs_tol=1e-6)
    for sq in range(12, 18):
        assert env.piece_properties[sq] is not None
        assert env.piece_properties[sq]["color"] == PieceColor.WHITE
        assert math.isclose(env.squares[sq].get_occupied_prob(), 1.0, abs_tol=1e-6)

    legal = env.get_legal_moves()
    # White row 1 (squares 12, 13, 14):
    # sq 12 (col 1, row 1) -> can move to sq 9 (col 0, row 2) or sq 10 (col 2, row 2), plus split [12, 9, 10]
    assert [12, 9] in legal
    assert [12, 10] in legal
    assert [12, 9, 10] in legal


def test_cheqqers_backward_men_capture_and_max_capture_depth():
    """
    Verifies two critical official rules from local-cheqqers-server.js:
    1. Uncrowned men CAN capture backwards (`canMoveBackwards = pieceProp.crowned || isTake`).
    2. Mandatory maximum-capture-depth filter (`calculateMaxCaptureDepth`): when a 2-jump
       chain and a 1-jump capture are both available, ONLY the 2-jump capture is legal!
    """
    env = OfficialCheqqersEnv(board_size=6, start_rows=0, quantumness=GameType.ENTANGLEMENT)
    # Square 7 is at (col=3, row=3)
    c7, r7 = env.get_row_col_from_index(7)
    assert (c7, r7) == (3, 3)

    # Backward neighbor of (3, 3) along (-1, -1) is (2, 2) [sq 10] -> landing (1, 1) [sq 12]
    enemy_back_sq = env.get_index(2, 2)
    land_back_sq = env.get_index(1, 1)
    assert enemy_back_sq == 10
    assert land_back_sq == 12

    env.squares[7].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env.piece_properties[7] = {"color": PieceColor.WHITE, "crowned": False, "pieceIds": [1]}

    env.squares[10].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env.piece_properties[10] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [2]}

    # Check backward capture by uncrowned White man is generated!
    legal_single = env.get_legal_moves()
    assert legal_single == [[7, 12]]

    # Now construct a board with a 2-jump chain vs a 1-jump capture:
    # White man A at sq 15 (0, 0) -> jumps sq 12 (1, 1) to sq 10 (2, 2),
    # then from sq 10 (2, 2) backward-jumps sq 13 (3, 1) to sq 17 (4, 0) (Depth 2!)
    env2 = OfficialCheqqersEnv(board_size=6, start_rows=0, quantumness=GameType.ENTANGLEMENT)
    sq_w_a = env2.get_index(0, 0)      # 15
    sq_b_1 = env2.get_index(1, 1)      # 12
    sq_mid = env2.get_index(2, 2)      # 10
    sq_b_2 = env2.get_index(3, 1)      # 13
    sq_end = env2.get_index(4, 0)      # 17
    assert (sq_w_a, sq_b_1, sq_mid, sq_b_2, sq_end) == (15, 12, 10, 13, 17)

    env2.squares[sq_w_a].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env2.piece_properties[sq_w_a] = {"color": PieceColor.WHITE, "crowned": False, "pieceIds": [1]}
    env2.squares[sq_b_1].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env2.piece_properties[sq_b_1] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [2]}
    env2.squares[sq_b_2].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env2.piece_properties[sq_b_2] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [3]}

    # White man B at (1, 3) [sq 6] with Black at (2, 4) [sq 4] -> lands at (3, 5) [sq 1] (Depth 1!)
    sq_w_b = env2.get_index(1, 3)      # 6
    sq_b_3 = env2.get_index(2, 4)      # 4
    sq_land_b = env2.get_index(3, 5)   # 1
    env2.squares[sq_w_b].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env2.piece_properties[sq_w_b] = {"color": PieceColor.WHITE, "crowned": False, "pieceIds": [4]}
    env2.squares[sq_b_3].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env2.piece_properties[sq_b_3] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [5]}

    sim_state = env2.get_simulation_state()
    d_chain = env2.calculate_max_capture_depth(
        sim_state, {"is_take_move": True, "from_index": sq_w_a, "to_index": sq_mid}
    )
    d_single = env2.calculate_max_capture_depth(
        sim_state, {"is_take_move": True, "from_index": sq_w_b, "to_index": sq_land_b}
    )
    assert d_chain == 2
    assert d_single == 1

    # Because d_chain (2) > d_single (1), ONLY [sq_w_a, sq_mid] ([15, 10]) is legal!
    legal2 = env2.get_legal_moves()
    assert legal2 == [[sq_w_a, sq_mid]]


def test_cheqqers_flying_queens_long_range_slide_and_capture():
    env = OfficialCheqqersEnv(board_size=6, start_rows=0, quantumness=GameType.ENTANGLEMENT)
    # Place a crowned White Queen at sq 15 (col 0, row 0)
    # Main diagonal (+1, +1): (0,0)[15] -> (1,1)[12] -> (2,2)[10] -> (3,3)[7] -> (4,4)[5] -> (5,5)[2]
    env.squares[15].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env.piece_properties[15] = {"color": PieceColor.WHITE, "crowned": True, "pieceIds": [1]}

    # Place a Black piece far away off the main diagonal so game is not over, at (1, 5) [sq 0]
    env.squares[0].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env.piece_properties[0] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [2]}

    legal_slides = env.get_legal_moves()
    # Flying Queen can slide to any empty square along the ray: 12, 10, 7, 5, 2!
    assert [15, 12] in legal_slides
    assert [15, 10] in legal_slides
    assert [15, 7] in legal_slides
    assert [15, 5] in legal_slides
    assert [15, 2] in legal_slides

    # Now place a Black piece at sq 10 (2, 2); squares 7 (3, 3), 5 (4, 4), 2 (5, 5) behind it are empty!
    env.squares[10].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env.piece_properties[10] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [3]}

    legal_takes = env.get_legal_moves()
    # Mandatory capture: Flying Queen at 15 can jump over 10 and land on 7, 5, or 2!
    assert [15, 7] in legal_takes
    assert [15, 5] in legal_takes
    assert [15, 2] in legal_takes
    assert [15, 12] not in legal_takes


def test_cheqqers_split_entanglement_and_tournament_json_roundtrip():
    env = OfficialCheqqersEnv(board_size=6, quantumness=GameType.ENTANGLEMENT)
    # White splits sq 12 -> sq 9 and sq 10
    env.apply_move([12, 9, 10])
    assert math.isclose(env.squares[12].get_occupied_prob(), 0.0, abs_tol=1e-6)
    assert math.isclose(
        env.squares[9].get_occupied_prob(), INV_SQRT2_JS * INV_SQRT2_JS, abs_tol=1e-5
    )
    assert math.isclose(
        env.squares[10].get_occupied_prob(), INV_SQRT2_JS * INV_SQRT2_JS, abs_tol=1e-5
    )
    assert not env.can_coexist(9, 10)

    payload = env.to_tournament_payload()
    reconstructed = OfficialCheqqersEnv.from_tournament_json(payload)
    assert reconstructed.turn == env.turn
    assert reconstructed.get_legal_moves() == env.get_legal_moves()
    assert not reconstructed.can_coexist(9, 10)


def test_cheqqers_move_outcomes_and_expecti_player():
    env = OfficialCheqqersEnv(board_size=6, quantumness=GameType.ENTANGLEMENT)
    env.apply_move([12, 9, 10])  # White splits 12 -> 9, 10
    env.apply_move([3, 6])       # Black moves 3 -> 6
    env.apply_move([13, 12])     # White moves 13 -> 12
    # Now Black at 6 (col 0, row 3) can jump over White superposition at 9 (col 1, row 2) to 13 (col 2, row 1)
    # OR move onto superposition square 9 ([6, 9]) which triggers landing measurement (50% moveFailed)!
    outcomes_landing = env.get_move_outcomes([6, 9])
    assert len(outcomes_landing) == 2
    assert math.isclose(sum(p for p, _ in outcomes_landing), 1.0, abs_tol=1e-5)

    # Evaluate state and run ExpectiCheqqersPlayer
    val_black = env.evaluate_state(PieceColor.BLACK)
    val_white = env.evaluate_state(PieceColor.WHITE)
    assert math.isclose(val_black, -val_white, abs_tol=50.0)  # asymmetric only by small tempo term

    player = ExpectiCheqqersPlayer(max_depth=4, time_limit_ms=500.0)
    legal = env.get_legal_moves()
    chosen = player.get_move(env, available_moves=legal)
    assert chosen in legal


def test_cheqqers_evert_v2_multijump_and_piece_ids_parity():
    """
    Verifies Evert van Nieuwenburg's updated local-cheqqers-server.js rule fix:
    `this.nextPieceId = 1` is initialized BEFORE `this.setupStartingPositions(startRows)`,
    giving starting pieces distinct integer IDs 1..12 on a 6x6 board (2 start rows),
    enabling multi-jump chains across distinct enemy pieces while still blocking
    re-jumping a split ghost branch of the same quantum piece in a single turn.
    """
    env = OfficialCheqqersEnv(board_size=6, start_rows=2, quantumness=GameType.ENTANGLEMENT)
    # Black pieces 0..5 have pieceIds [1]..[6]; White pieces 12..17 have pieceIds [7]..[12]
    black_ids = [env.piece_properties[i]["pieceIds"] for i in range(0, 6)]
    white_ids = [env.piece_properties[i]["pieceIds"] for i in range(12, 18)]
    assert black_ids == [[1], [2], [3], [4], [5], [6]]
    assert white_ids == [[7], [8], [9], [10], [11], [12]]
    assert env.next_piece_id == 13

    # Verify anti-rejump of split ghost branches vs valid multi-jump of distinct pieces:
    # Suppose Black piece #1 splits onto sq 12 (1,1) and sq 13 (3,1).
    # White man at sq 15 (0,0) jumps sq 12 (1,1) -> lands on sq 10 (2,2).
    # Can White then jump sq 13 (3,1) -> sq 17 (4,0)?
    # NO if sq 13 is the other split branch of piece #1 (same pieceId [1])!
    # YES if sq 13 is a distinct Black piece #2 (pieceId [2])!
    env_split = OfficialCheqqersEnv(board_size=6, start_rows=0, quantumness=GameType.ENTANGLEMENT)
    env_split.squares[15].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env_split.piece_properties[15] = {"color": PieceColor.WHITE, "crowned": False, "pieceIds": [7]}
    # Black piece #1 at sq 9 (0, 2) splits to sq 12 (1, 1) and sq 6 (1, 3) -> let's place Black at sq 9 and split to 12, 13
    env_split.squares[10].amplitudes = {SquareState.OCCUPIED: 1.0 + 0.0j}
    env_split.piece_properties[10] = {"color": PieceColor.BLACK, "crowned": False, "pieceIds": [1]}
    env_split.turn = PieceColor.BLACK
    # Black at 10 (2, 2) splits to 12 (1, 1) and 13 (3, 1)
    env_split.apply_move([10, 12, 13])
    assert env_split.piece_properties[12]["pieceIds"] == [1]
    assert env_split.piece_properties[13]["pieceIds"] == [1]
    assert env_split.turn == PieceColor.WHITE

    # White at 15 (0, 0) jumps over 12 (1, 1) to 10 (2, 2); force measurement of landing=EMPTY
    env_split.apply_move([15, 10])
    # Because sq 13 shares pieceId=1 with already-jumped sq 12, White CANNOT jump sq 13 in the same turn!
    assert env_split.multi_jump_square is None
    assert env_split.turn == PieceColor.BLACK

