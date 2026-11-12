"""Code excerpt: destination-voxel feasibility for 26-neighbor motions.

The caller supplies a voxel map and a current grid position. Action indices
follow the order of MOVES. This module covers only the feasibility mask.
"""

from itertools import product

import numpy as np


MOVES = tuple(move for move in product((-1, 0, 1), repeat=3) if any(move))


def valid_action_mask(grid: np.ndarray, position: tuple[int, int, int]) -> np.ndarray:
    """Return a Boolean mask for destinations inside the map and unoccupied.

    The map is indexed as grid[z, y, x]. A positive entry marks an occupied
    voxel. Positions and displacements are expressed in (x, y, z) order.
    """
    size_z, size_y, size_x = grid.shape
    x, y, z = position
    mask = np.zeros(len(MOVES), dtype=bool)
    for action, (dx, dy, dz) in enumerate(MOVES):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z:
            mask[action] = grid[nz, ny, nx] <= 0
    return mask
