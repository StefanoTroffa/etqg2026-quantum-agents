# Credits

## Authors
- **Chris Chalkias** (Leiden University): the exact solution of TiqTaqToe (`exact/`): the rules port verified
  against the organisers' engine, the solver and its tables, the move server and the tie-break analysis; the
  independent replication of the Cheqqers result on Snellius.
- **Stefano Troffa**: the Cheqqers agent: the C++ simulator and searches (`native/cheqqers/`), the fitted
  expectimax player (FW), the policy-value transformer, its Gumbel self-play training and the MCTS that plays it;
  the move server (`server.py`) and the evaluation tools.

## Acknowledgements
- **Marien Raat** and **Tommaso Zendron**, who competed as their own team, for discussions and sparring during the
  tournament.
- **Luuk van den Nouweland** and **Marien Raat** for the original Cheqqers prototype (GPL, not included here), the
  starting point of our June agent.

## The tournament
- **Vincenzo Lipardi** and **Evert van Nieuwenburg**: the 1st European Tournament in Quantum Games (ETQG 2026), the
  quantum game engine `@quantumplayed/quantum-game-engine`, the local testers and the sparring arena
  (github.com/VincenzoLipardi/ETQG_2026). The organisers' files are not redistributed here; download them from
  their repository.
- **The LMU team**: their published June winner (github.com/michaelkoelle/cheqqers-ai, MIT) served as an external
  benchmark.

## Methods we build on
- D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through
  self-play", Science 362 (2018) (AlphaZero).
- I. Danihelka, A. Guez, J. Schrittwieser, D. Silver, "Policy improvement by planning with Gumbel", ICLR 2022.
- S. Ross, G. Gordon, D. Bagnell, "A reduction of imitation learning and structured prediction to no-regret
  online learning", AISTATS 2011 (DAgger).

## Tools
Development used AI coding assistants (Claude Code and others) under the authors' direction; every result reported
here was measured, and the measurements are in `experiments/CONCLUDED_EXPERIMENTS.md`.
