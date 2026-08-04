"""Validation-only checkpoint and architecture selection utilities.

The functions in this module deliberately avoid test metrics.  They implement
an auditable, pre-specified lexicographic policy:

1. maximise the primary validation metric (Dice by default);
2. when candidates are within the configured Dice tolerance, minimise HD95;
3. then minimise ASSD;
4. then maximise Boundary-F1;
5. finally prefer the earlier epoch / deterministic name.

The policy is stored in every checkpoint and selection lock so the final test
set can be evaluated only after the decision has been frozen.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_SELECTION_POLICY: Dict[str, Any] = {
    "primary": {"metric": "dice", "mode": "max", "tolerance": 0.001},
    "tie_breakers": [
        {"metric": "missing_prediction_rate", "mode": "min", "tolerance": 0.0},
        {"metric": "hd95", "mode": "min", "tolerance": 0.0},
        {"metric": "assd", "mode": "min", "tolerance": 0.0},
        {"metric": "boundary_f1", "mode": "max", "tolerance": 0.0},
    ],
    "min_epoch": 1,
}


def normalize_selection_policy(policy: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a validated policy with defaults filled in."""
    out = deepcopy(DEFAULT_SELECTION_POLICY)
    if policy:
        for key, value in policy.items():
            if key == "primary" and isinstance(value, Mapping):
                out["primary"].update(value)
            elif key == "tie_breakers":
                out["tie_breakers"] = [dict(item) for item in value]
            else:
                out[key] = value

    criteria = [out["primary"], *out.get("tie_breakers", [])]
    for criterion in criteria:
        if not criterion.get("metric"):
            raise ValueError(f"Selection criterion has no metric: {criterion}")
        if criterion.get("mode") not in {"max", "min"}:
            raise ValueError(f"Selection mode must be 'max' or 'min': {criterion}")
        criterion["tolerance"] = float(criterion.get("tolerance", 0.0))
    out["min_epoch"] = int(out.get("min_epoch", 1))
    return out


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compare_values(candidate: Any, incumbent: Any, mode: str, tolerance: float = 0.0) -> int:
    """Compare two scalar values.

    Returns
    -------
    1
        Candidate is better.
    0
        Values are tied within tolerance.
    -1
        Candidate is worse.

    Finite values are always preferred to NaN/inf values.
    """
    c_finite = _is_finite(candidate)
    i_finite = _is_finite(incumbent)
    if c_finite and not i_finite:
        return 1
    if i_finite and not c_finite:
        return -1
    if not c_finite and not i_finite:
        return 0

    c = float(candidate)
    i = float(incumbent)
    tol = max(0.0, float(tolerance))
    delta = c - i
    if abs(delta) <= tol:
        return 0
    if mode == "max":
        return 1 if delta > 0 else -1
    return 1 if delta < 0 else -1


def compare_metric_dicts(
    candidate: Mapping[str, Any],
    incumbent: Optional[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Return whether candidate should replace incumbent under the policy."""
    normalized = normalize_selection_policy(policy)
    if incumbent is None:
        return True, "first eligible candidate"

    criteria: List[Mapping[str, Any]] = [
        normalized["primary"],
        *normalized.get("tie_breakers", []),
    ]
    for criterion in criteria:
        metric = criterion["metric"]
        result = compare_values(
            candidate.get(metric),
            incumbent.get(metric),
            mode=criterion["mode"],
            tolerance=criterion.get("tolerance", 0.0),
        )
        if result > 0:
            return True, f"better validation {metric} ({criterion['mode']})"
        if result < 0:
            return False, f"worse validation {metric} ({criterion['mode']})"

    # Exact/tolerance tie: keep the earlier incumbent to avoid post-hoc drift.
    return False, "tied under the pre-specified policy; retained earlier checkpoint"


def ranking_key(metrics: Mapping[str, Any], policy: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Create a deterministic sorting key (best first when used with sorted)."""
    normalized = normalize_selection_policy(policy)
    key: List[Any] = []
    criteria: Iterable[Mapping[str, Any]] = [
        normalized["primary"],
        *normalized.get("tie_breakers", []),
    ]
    for criterion in criteria:
        value = metrics.get(criterion["metric"])
        if not _is_finite(value):
            key.extend([1, 0.0])
            continue
        v = float(value)
        key.extend([0, -v if criterion["mode"] == "max" else v])
    key.append(str(metrics.get("variant", metrics.get("name", ""))))
    return tuple(key)
