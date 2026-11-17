"""The time a move may take, from the limit the organisers send in each request.

The arena and the organisers' harness tell the agent its limit in `settings`: `timeout` in milliseconds for
Cheqqers (3000 in the arena), `moveTime` in seconds for TiqTaqToe (5). The arena counts network time and allows
500 ms more (arena-setup.md, 8 November 2026). The search keeps a fixed margin below the limit for the network,
the request handling and the search's own overshoot: 800 ms under 3,000 ms gave replies of at most 2,343 ms as the
arena measured them on 13 November 2026, and 1,000 ms under 5,000 ms for TiqTaqToe. A short limit keeps half.

Environment variables override, for experiments: CHEQQERS_TIME_LIMIT_MS, CHEQQERS_MAX_DEPTH, TTT_TIME_LIMIT_MS.
"""

from __future__ import annotations

import os
from typing import Any, Optional

ARENA_CHEQQERS_TIMEOUT_MS = 3000.0
ARENA_TIQTAQTOE_MOVE_TIME_S = 5.0
CHEQQERS_MARGIN_MS = 800.0
TIQTAQTOE_MARGIN_MS = 1000.0
# High enough that the clock, not the cap, ends the search: a cap of 5 won 1 of 7 games against other teams on
# 13 November 2026, stopping after 0.2-1.1 s in the positions where it lost.
CHEQQERS_DEFAULT_DEPTH = 12


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _budget(limit_ms: float, margin_ms: float) -> float:
    return max(limit_ms - margin_ms, limit_ms / 2.0)


def cheqqers_budget(settings: Optional[dict[str, Any]]) -> tuple[float, int]:
    """(milliseconds the search may use, its depth cap) for a Cheqqers request with these `settings`."""
    override = _positive(os.getenv("CHEQQERS_TIME_LIMIT_MS"))
    if override is not None:
        time_ms = override
    else:
        limit = _positive((settings or {}).get("timeout")) or ARENA_CHEQQERS_TIMEOUT_MS
        time_ms = _budget(limit, CHEQQERS_MARGIN_MS)
    depth = int(_positive(os.getenv("CHEQQERS_MAX_DEPTH")) or CHEQQERS_DEFAULT_DEPTH)
    return time_ms, depth


def tiqtaqtoe_budget_ms(settings: Optional[dict[str, Any]]) -> float:
    """Milliseconds the TiqTaqToe search may use (it answers only when the exact tables cannot)."""
    override = _positive(os.getenv("TTT_TIME_LIMIT_MS"))
    if override is not None:
        return override
    seconds = _positive((settings or {}).get("moveTime")) or ARENA_TIQTAQTOE_MOVE_TIME_S
    return _budget(seconds * 1000.0, TIQTAQTOE_MARGIN_MS)
