import math
from typing import List


AMOUNT_OF_Z_TERMS = 11

def theta(t: float) -> float:
    """Asymptotic expansion of the Riemann–Siegel theta function."""
    return (t / 2) * math.log(t / (2 * math.pi)) - t / 2 - math.pi / 8 + 1 / (48 * t) + 7 / (5760 * math.pow(t, 3))


def Z(t: float) -> float:
    """Riemann–Siegel Z-function approximation using all terms up to N ≈ sqrt(t / 2π)."""
    summation: float = 0.0
    N = math.floor(math.sqrt(t / (2 * math.pi)))
    for n in range(1, N + 1):
        angle = theta(t) - t * math.log(n)
        summation += math.cos(angle) / math.sqrt(n)
    return 2 * summation


def Z_ten_first_cos_terms(t: float) -> List[float]:
    """First 10 terms of the Riemann–Siegel series (cos terms) for features."""
    cos_terms: List[float] = []
    for n in range(2, AMOUNT_OF_Z_TERMS):
        angle = theta(t) - t * math.log(n)
        factor = 2 / math.sqrt(n)
        cos_terms.append(factor * math.cos(angle))
    return cos_terms


def Z_ten_first_sin_terms(t: float) -> List[float]:
    """First 10 terms of the Riemann–Siegel series (sin terms) for features."""
    sin_terms: List[float] = []
    for n in range(1, AMOUNT_OF_Z_TERMS):
        angle = theta(t) - t * math.log(n)
        factor = 2 / math.sqrt(n)
        sin_terms.append(factor * math.sin(angle))
    return sin_terms
