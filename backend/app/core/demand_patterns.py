from typing import List, Dict, Optional, Any
import random
from enum import Enum


class DemandPatternType(str, Enum):
    CLASSIC = "classic"
    RANDOM = "random"
    SEASONAL = "seasonal"
    CONSTANT = "constant"


DEFAULT_CLASSIC_PARAMS = {
    "initial_demand": 4,
    "change_week": 6,
    "final_demand": 8,
}


def _safe_int(value: Any, default: int) -> int:
    """Convert a value to an integer, falling back to the provided default."""
    try:
        if value is None:
            raise ValueError("None")
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_classic_params(params: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Normalize classic demand parameters to the {initial, change_week, final} schema."""
    params = params or {}

    initial = _safe_int(
        params.get("initial_demand", params.get("base_demand")),
        DEFAULT_CLASSIC_PARAMS["initial_demand"],
    )

    if "change_week" in params:
        change_week = _safe_int(params.get("change_week"), DEFAULT_CLASSIC_PARAMS["change_week"])
    else:
        stable_period = params.get("stable_period")
        change_week = (
            _safe_int(stable_period, DEFAULT_CLASSIC_PARAMS["change_week"] - 1) + 1
            if stable_period is not None
            else DEFAULT_CLASSIC_PARAMS["change_week"]
        )

    change_week = max(1, change_week)

    if "final_demand" in params:
        final = _safe_int(params.get("final_demand"), DEFAULT_CLASSIC_PARAMS["final_demand"])
    else:
        step_increase = params.get("step_increase")
        final = (
            initial + _safe_int(step_increase, DEFAULT_CLASSIC_PARAMS["final_demand"] - initial)
            if step_increase is not None
            else DEFAULT_CLASSIC_PARAMS["final_demand"]
        )

    initial = max(0, initial)
    final = max(0, final)

    return {
        "initial_demand": initial,
        "change_week": change_week,
        "final_demand": final,
    }


def normalize_demand_pattern(pattern_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a normalized demand pattern dictionary with sanitized parameters."""
    pattern = dict(pattern_config or {})
    raw_type = pattern.get("type", DemandPatternType.CLASSIC)
    try:
        pattern_type = DemandPatternType(raw_type)
    except ValueError:
        pattern_type = DemandPatternType.CLASSIC

    params = pattern.get("params", {}) if isinstance(pattern.get("params", {}), dict) else {}

    if pattern_type == DemandPatternType.CLASSIC:
        params = normalize_classic_params(params)

    normalized = {
        key: value
        for key, value in pattern.items()
        if key not in {"type", "params"}
    }
    normalized.update({
        "type": pattern_type.value,
        "params": params,
    })
    return normalized


class DemandGenerator:
    """Generates different types of demand patterns for the beer game."""

    @staticmethod
    def generate_classic(
        num_rounds: int = 52,
        initial_demand: Optional[int] = None,
        change_week: Optional[int] = None,
        final_demand: Optional[int] = None,
        stable_period: Optional[int] = None,
        step_increase: Optional[int] = None,
    ) -> List[int]:
        """Generate a classic beer game demand pattern with a single step change."""
        if num_rounds <= 0:
            return []

        normalized = normalize_classic_params(
            {
                "initial_demand": initial_demand,
                "change_week": change_week,
                "final_demand": final_demand,
                "stable_period": stable_period,
                "step_increase": step_increase,
            }
        )

        initial = normalized["initial_demand"]
        final = normalized["final_demand"]
        change_at = normalized["change_week"]

        demand: List[int] = []
        for week in range(1, num_rounds + 1):
            demand.append(final if week >= change_at else initial)

        return demand

    @staticmethod
    def generate_random(num_rounds: int, min_demand: int = 1, max_demand: int = 10) -> List[int]:
        """Generate random demand values within a specified range."""
        return [random.randint(min_demand, max_demand) for _ in range(num_rounds)]

    @staticmethod
    def generate_seasonal(num_rounds: int, base_demand: int = 4, amplitude: int = 2, period: int = 12) -> List[int]:
        """Generate a seasonal demand pattern."""
        import math

        return [
            max(1, int(base_demand + amplitude * math.sin(2 * math.pi * (i % period) / period)))
            for i in range(num_rounds)
        ]

    @staticmethod
    def generate_constant(num_rounds: int, demand: int = 4) -> List[int]:
        """Generate a constant demand pattern."""
        return [demand] * num_rounds

    @classmethod
    def generate(
        cls,
        pattern_type: DemandPatternType,
        num_rounds: int,
        **kwargs,
    ) -> List[int]:
        """Generate demand pattern based on the specified type."""
        if pattern_type == DemandPatternType.CLASSIC:
            return cls.generate_classic(num_rounds, **kwargs)
        if pattern_type == DemandPatternType.RANDOM:
            return cls.generate_random(num_rounds, **kwargs)
        if pattern_type == DemandPatternType.SEASONAL:
            return cls.generate_seasonal(num_rounds, **kwargs)
        if pattern_type == DemandPatternType.CONSTANT:
            return cls.generate_constant(num_rounds, **kwargs)
        raise ValueError(f"Unknown demand pattern type: {pattern_type}")


DEFAULT_DEMAND_PATTERN = {
    "type": DemandPatternType.CLASSIC.value,
    "params": DEFAULT_CLASSIC_PARAMS.copy(),
}


def get_demand_pattern(
    pattern_config: Optional[Dict] = None,
    num_rounds: int = 52,
) -> List[int]:
    """Get a demand pattern based on the provided configuration."""
    normalized = normalize_demand_pattern(pattern_config or DEFAULT_DEMAND_PATTERN)

    try:
        pattern_type = DemandPatternType(normalized.get("type", DemandPatternType.CLASSIC))
    except ValueError:
        pattern_type = DemandPatternType.CLASSIC

    params = normalized.get("params", {})

    return DemandGenerator.generate(pattern_type, num_rounds, **params)
