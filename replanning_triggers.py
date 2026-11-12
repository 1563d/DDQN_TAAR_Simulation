"""Code excerpt: the original occupancy and aerial-deviation triggers.

The caller supplies map snapshots, path points, the checking schedule, and
the call count. This module evaluates the predicates used to request a plan.
"""

from math import dist

import numpy as np


def occupancy_event(
    current: np.ndarray, reference: np.ndarray, *, check_due: bool
) -> bool:
    """Report any map difference when a scheduled check is due.

    The reference snapshot is the map used for the latest successful plan.
    The caller maintains the checking timer and reference snapshot.
    """
    if current.shape != reference.shape:
        raise ValueError("The two occupancy maps must have the same shape.")
    return check_due and not np.array_equal(current, reference)


def aerial_deviation_event(
    position, reference_path, *, airborne: bool, threshold: float
) -> bool:
    """Report aerial deviation using the nearest discrete reference point.

    Position, path points, and threshold must use the same distance units.
    """
    if not airborne:
        return False
    return min(dist(position, point) for point in reference_path) > threshold


def replanning_requested(
    *, occupancy_changed: bool, aerial_deviation: bool, calls: int, call_limit: int
) -> bool:
    """Allow either event to request replanning within the call budget."""
    return (occupancy_changed or aerial_deviation) and calls < call_limit
