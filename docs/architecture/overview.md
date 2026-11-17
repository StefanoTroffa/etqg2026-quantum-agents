# 🗺️ System Map & Overview

This document provides a high-level overview of the server deployment architecture and data/control flow of the Quantum RL tournament player.

---

## 🎯 Purpose of the System
The system acts as an AI move prediction backend. It listens for incoming HTTP POST requests containing game states from a tournament manager or interactive game board and returns the selected best legal move within a strict latency constraint.

---

## 👥 System Actors

```
┌─────────────────┐           ┌──────────────┐           ┌──────────────┐
│  Tournament     ├──────────>│ Ngrok Tunnel ├──────────>│ Python Server│
│  Runner Client  │<──────────┤ (HTTPS to    │<──────────┤ (Port 8081)  │
└─────────────────┘           │  local port) │           └──────┬───────┘
                              └──────────────┘                  │
                                                                ▼
                                                         ┌──────────────┐
                                                         │Model Registry│
                                                         │ (MLP/KAN/SNN)│
                                                         └──────────────┘
```

1. **Tournament Runner Client**: The external engine hosting the game, generating states and legal move lists.
2. **ngrok Tunnel**: Provides a secure HTTPS endpoint (the June address, now retired) mapping directly to the local WSL laptop.
3. **Python Server (`server.py`)**: The HTTP backend serving AI predictions.
4. **Model Registry**: Memory-cached PyTorch neural networks containing our trained policies.

---

## 🔁 Data & Control Flow
1. **Request**: The tournament client sends a `POST /move` request containing the current board configuration (positions and measurement probabilities) and `legalMoves`.
2. **Parsing**: The server deserializes the payload, parses the board shape to distinguish between Tic-Tac-Toe and Checkers, and extracts the valid action indices.
3. **Inference**:
   * For **Tic-Tac-Toe**: Runs Q-MCTS search guided by MLP/KAN model predictions.
   * For **Checkers**: Runs iterative deepening Minimax search with a strict timer constraint.
4. **Response**: The server translates the chosen action back into the original tournament move format and returns it as a JSON payload (`{"move": selected_move}`).

---

## 🚨 Critical Invariants
* **Strict Move Budget**: The roundtrip time measured by the organisers, network included, must stay under **3.0 seconds for Cheqqers and 5.0 seconds for TiqTaqToe** (arena instructions of 8 November 2026; the 1.0 second written here until 10 November was the June assumption). If the server takes longer, the game is forfeited. See `critical-workflows.md`, CW1.
* **November 2026 state of this map**: the TiqTaqToe inference step above is now exact tables first (`src/search/exact_tiqtaqtoe.py`), then Q-MCTS as fallback; Cheqqers is expectimax (`src/search/expecti_cheqqers.py`). In the arena week the tunnel is the organisers' connector, not ngrok.
* **Format Correctness**: The move returned must be strictly selected from the received `legalMoves` array. Returning an arbitrary move or incorrect format will result in a forfeit.
