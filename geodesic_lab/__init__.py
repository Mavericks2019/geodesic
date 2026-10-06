"""Geodesic robot animation and differential geometry experiments."""

from .geometry import Metric, Robot, SingularMetricError
from .solver import (
    TrajectoryResult,
    integrate_geodesic,
    integrate_task_geodesic,
    integrate_original,
    original_acceleration,
    original_metric,
    solve_boundary,
)

__all__ = [
    "Robot", "Metric", "SingularMetricError", "TrajectoryResult",
    "integrate_geodesic", "integrate_task_geodesic", "integrate_original", "original_acceleration",
    "original_metric", "solve_boundary",
]
