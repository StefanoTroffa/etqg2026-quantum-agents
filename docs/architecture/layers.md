# 📜 Layer Contracts & Ownership

This document defines the ownership boundaries, interface signatures, and dependencies of the logical layers in the Quantum RL codebase.

---

## 1. The Server Gateway Layer
* **Purpose**: Gateway for incoming tournament requests. Handles HTTP POST/GET requests and JSON serialization.
* **File Directory**: `server.py`
* **Owns**: Route mappings (`/move`, `/api/ai_move`, `/status`), deserialization wrappers, and HTTP response success/error headers.
* **Must Not Know**: Internal model configurations, GPU/CUDA state management, or rollout algorithm implementations.
* **Upstream**: Tournament runner.
* **Downstream**: Environment Wrapper Layer, Player Layer.
* **Verification**: `curl -I <public address>/health` (status check; the June ngrok address is retired).

---

## 2. The Environment Wrapper Layer
* **Purpose**: Maps JSON tournament board payloads into Python environment simulator objects.
* **File Directory**: `src/env/`
* **Owns**: State reconstruction (`reconstruct_tiqtaqtoe_state_111`, `reconstruct_env`), action masking, legal moves mappings, and rotation calculations.
* **Must Not Know**: Model weight paths or search player state.
* **Upstream**: Server Gateway Layer.
* **Downstream**: Player Layer, Simulator Engine.
* **Verification**: Unit tests in `tests/compliance/` or `tests/env/`.

---

## 3. The Model & Weights Registry Layer
* **Purpose**: Instantiates neural networks (MLPNet, KANNet, SpikingNet) and handles parameter deserialization.
* **File Directory**: `src/models/`, `results/*.pth`
* **Owns**: Model architectures, `load_state_dict` overrides for key mapping/fallbacks, and weight loading routines.
* **Must Not Know**: HTTP protocol, routing endpoints, or game simulator logic.
* **Upstream**: None.
* **Downstream**: Server Gateway Layer, Player Layer.
* **Verification**: `python3 src/models/mlp.py` (Verify instantiation and test mock forward passes).

---

## 4. The Search & Player Logic Layer
* **Purpose**: Evaluates legal actions using lookahead search guided by the model weights.
* **File Directory**: `src/search/`, `evaluate.py`
* **Owns**: Q-MCTS search tree, Monte Carlo rollouts, Minimax depth pruning, and search cache databases (`mcts_checkers_cache.db`).
* **Must Not Know**: HTTP socket state or routing endpoints.
* **Upstream**: Environment Wrapper Layer.
* **Downstream**: Model Layer.
* **Verification**: Isolated benchmarks (`profile_checkers.py`).
