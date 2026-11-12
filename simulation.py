"""Run a small example of voxel action masking and replanning triggers."""

import numpy as np

from replanning_triggers import aerial_deviation_event, occupancy_event, replanning_requested
from voxel_action_mask import MOVES, valid_action_mask


def main() -> None:
    # The map is indexed as [z, y, x], while path points use (x, y, z).
    reference_map = np.zeros((3, 4, 4), dtype=np.uint8)
    current_map = reference_map.copy()
    current_map[1, 1, 2] = 1
    position = (1, 1, 1)
    reference_path = [position, (2, 1, 1), (3, 1, 1)]
    forward_action = MOVES.index((1, 0, 0))
    before = valid_action_mask(reference_map, position)
    after = valid_action_mask(current_map, position)

    print(f"Feasible moves before/after the update: {before.sum()}/{after.sum()}")
    print(f"Move to (2, 1, 1) after the update: {bool(after[forward_action])}")

    cases = (
        ("Unchanged map, on path", reference_map, position, 0),
        ("New obstacle", current_map, position, 0),
        ("Aerial deviation", reference_map, (1, 2, 1), 0),
        ("New obstacle, budget exhausted", current_map, position, 1),
    )
    for name, grid, current_position, calls in cases:
        changed = occupancy_event(grid, reference_map, check_due=True)
        deviated = aerial_deviation_event(
            current_position, reference_path, airborne=True, threshold=0.5
        )
        request = replanning_requested(
            occupancy_changed=changed, aerial_deviation=deviated, calls=calls, call_limit=1
        )
        print(f"{name}: map change={changed}, deviation={deviated}, request={request}")


if __name__ == "__main__":
    main()
