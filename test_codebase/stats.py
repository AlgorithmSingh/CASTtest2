"""Statistical analysis module for computing descriptive statistics."""

import math
from typing import List, Optional, Tuple


def compute_mean(values: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        values: A non-empty list of numeric values.

    Returns:
        The arithmetic mean.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def compute_variance(values: List[float], ddof: int = 0) -> float:
    """Calculate the variance of a list of numbers.

    Args:
        values: A non-empty list of numeric values.
        ddof: Delta degrees of freedom. Use 0 for population
              variance and 1 for sample variance.

    Returns:
        The variance of the values.
    """
    mean = compute_mean(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    return sum(squared_diffs) / (len(values) - ddof)


def compute_std(values: List[float], ddof: int = 0) -> float:
    """Calculate the standard deviation."""
    return math.sqrt(compute_variance(values, ddof))


def compute_median(values: List[float]) -> float:
    """Calculate the median of a list of numbers.

    For even-length lists, returns the average of the two middle values.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        mid = n // 2
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


class StatsSummary:
    """Compute and store a full descriptive statistics summary."""

    def __init__(self, data: List[float]):
        self.data = data
        self._mean: Optional[float] = None
        self._variance: Optional[float] = None
        self._std: Optional[float] = None
        self._median: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    @property
    def mean(self) -> float:
        if self._mean is None:
            self._mean = compute_mean(self.data)
        return self._mean

    @property
    def variance(self) -> float:
        if self._variance is None:
            self._variance = compute_variance(self.data, ddof=1)
        return self._variance

    @property
    def std(self) -> float:
        if self._std is None:
            self._std = compute_std(self.data, ddof=1)
        return self._std

    @property
    def median(self) -> float:
        if self._median is None:
            self._median = compute_median(self.data)
        return self._median

    @property
    def minimum(self) -> float:
        if self._min is None:
            self._min = min(self.data)
        return self._min

    @property
    def maximum(self) -> float:
        if self._max is None:
            self._max = max(self.data)
        return self._max

    def range(self) -> float:
        """Return the range (max - min) of the data."""
        return self.maximum - self.minimum

    def z_scores(self) -> List[float]:
        """Compute z-scores for each value in the data."""
        return [(x - self.mean) / self.std for x in self.data]

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile using linear interpolation.

        Args:
            p: Percentile value between 0 and 100.

        Returns:
            The interpolated percentile value.
        """
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        sorted_data = sorted(self.data)
        k = (p / 100.0) * (len(sorted_data) - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def to_dict(self) -> dict:
        """Return all statistics as a dictionary."""
        return {
            "mean": self.mean,
            "median": self.median,
            "variance": self.variance,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "range": self.range(),
            "count": len(self.data),
        }

    def __repr__(self) -> str:
        return (
            f"StatsSummary(mean={self.mean:.4f}, median={self.median:.4f}, "
            f"std={self.std:.4f}, n={len(self.data)})"
        )
