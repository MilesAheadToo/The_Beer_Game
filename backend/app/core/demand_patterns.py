from typing import List, Dict, Optional
import random
from enum import Enum

class DemandPatternType(str, Enum):
    CLASSIC = "classic"
    RANDOM = "random"
    SEASONAL = "seasonal"
    CONSTANT = "constant"

class DemandGenerator:
    """
    Generates different types of demand patterns for the beer game.
    """
    @staticmethod
    def generate_classic(num_rounds: int = 52, stable_period: int = 5, step_increase: int = 4) -> List[int]:
        """
        Generate a classic beer game demand pattern with a step increase after a stable period.
        
        Args:
            num_rounds: Total number of rounds in the game
            stable_period: Number of initial rounds with stable demand (default: 5)
            step_increase: The amount to increase demand by after the stable period (default: 4)
            
        Returns:
            List of demand values for each round
        """
        if num_rounds <= 0:
            return []
            
        # Initial stable demand (4 units per round)
        base_demand = 4
        demand = [base_demand] * stable_period
        
        # Step increase after stable period
        for _ in range(stable_period, num_rounds):
            demand.append(base_demand + step_increase)
            
        return demand

    @staticmethod
    def generate_random(num_rounds: int, min_demand: int = 1, max_demand: int = 10) -> List[int]:
        """
        Generate random demand values within a specified range.
        
        Args:
            num_rounds: Number of rounds to generate demand for
            min_demand: Minimum possible demand (inclusive)
            max_demand: Maximum possible demand (inclusive)
        """
        return [random.randint(min_demand, max_demand) for _ in range(num_rounds)]

    @staticmethod
    def generate_seasonal(num_rounds: int, base_demand: int = 4, amplitude: int = 2, period: int = 12) -> List[int]:
        """
        Generate seasonal demand pattern with a base demand and seasonal variation.
        
        Args:
            num_rounds: Number of rounds to generate demand for
            base_demand: Average demand level
            amplitude: Maximum variation from base demand
            period: Number of rounds in a complete seasonal cycle
        """
        import math
        return [
            max(1, int(base_demand + amplitude * math.sin(2 * math.pi * (i % period) / period)))
            for i in range(num_rounds)
        ]
    
    @staticmethod
    def generate_constant(num_rounds: int, demand: int = 4) -> List[int]:
        """
        Generate a constant demand pattern.
        
        Args:
            num_rounds: Number of rounds to generate demand for
            demand: Constant demand value for all rounds
        """
        return [demand] * num_rounds

    @classmethod
    def generate(
        cls, 
        pattern_type: DemandPatternType, 
        num_rounds: int, 
        **kwargs
    ) -> List[int]:
        """
        Generate demand pattern based on the specified type.
        
        Args:
            pattern_type: Type of demand pattern to generate
            num_rounds: Number of rounds to generate demand for
            **kwargs: Additional arguments specific to each pattern type
        """
        if pattern_type == DemandPatternType.CLASSIC:
            return cls.generate_classic(num_rounds, **kwargs)
        elif pattern_type == DemandPatternType.RANDOM:
            return cls.generate_random(num_rounds, **kwargs)
        elif pattern_type == DemandPatternType.SEASONAL:
            return cls.generate_seasonal(num_rounds, **kwargs)
        elif pattern_type == DemandPatternType.CONSTANT:
            return cls.generate_constant(num_rounds, **kwargs)
        else:
            raise ValueError(f"Unknown demand pattern type: {pattern_type}")

# Default demand pattern configuration
DEFAULT_DEMAND_PATTERN = {
    "type": DemandPatternType.CLASSIC,
    "params": {
        "stable_period": 5,
        "step_increase": 4
    }
}

def get_demand_pattern(
    pattern_config: Optional[Dict] = None,
    num_rounds: int = 52
) -> List[int]:
    """
    Get a demand pattern based on the provided configuration.
    
    Args:
        pattern_config: Configuration for the demand pattern
        num_rounds: Number of rounds to generate demand for
        
    Returns:
        List of demand values for each round
    """
    if pattern_config is None:
        pattern_config = DEFAULT_DEMAND_PATTERN
        
    pattern_type = pattern_config.get("type", DemandPatternType.CLASSIC)
    params = pattern_config.get("params", {})
    
    return DemandGenerator.generate(pattern_type, num_rounds, **params)
