"""
Leiden Cross-Team Sparring Arena & Blind-Spot Harvester Package.

Provides head-to-head and round-robin tournament evaluation, Elo + 95% CI,
SPRT log-likelihood ratio (LLR) gating, HTTP `/move` endpoint adapters,
and automated blind-spot harvesting + teacher distillation for TiqTaqToe
and Quantum Cheqqers.
"""

from .leiden_sparring_hub import (
    HttpEndpointAgent,
    InProcessCheqqersAgent,
    InProcessTiqTaqToeAgent,
    LeagueConfiguredAgent,
    LeagueModelSpec,
    PopulationLeagueManager,
    SeriesResult,
    SparringAgent,
    compute_elo_diff_and_ci,
    compute_sprt_llr,
    run_cheqqers_series,
    run_round_robin,
    run_tiqtaqtoe_series,
    solve_hard_cases,
)

__all__ = [
    "SparringAgent",
    "HttpEndpointAgent",
    "InProcessTiqTaqToeAgent",
    "InProcessCheqqersAgent",
    "LeagueModelSpec",
    "LeagueConfiguredAgent",
    "PopulationLeagueManager",
    "SeriesResult",
    "compute_elo_diff_and_ci",
    "compute_sprt_llr",
    "run_tiqtaqtoe_series",
    "run_cheqqers_series",
    "run_round_robin",
    "solve_hard_cases",
]
