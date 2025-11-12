from pathlib import Path

import laspy
import numpy as np

from pc_rwalker import random_walker_segmentation


if __name__ == '__main__':
    n_neighbors = 25
    bunny_seeds = [
        [240],        # right ear
        [2825],       # left ear
        [164],        # snout
        [152, 6697],  # body front ant back
        [5053],       # right foot
        [9665],       # left foot
        [3445],       # tail
    ]
    bunny = Path(__file__).parent / 'bunny.laz'
    las = laspy.read(bunny)

    las.classification = np.zeros_like(las.classification)
    for i, s in enumerate(bunny_seeds, start=1):
        las.classification[s] = i

    las.write(bunny.with_stem('seed_bunny'))

    idx = random_walker_segmentation(
        las.xyz, bunny_seeds, n_neighbors, n_proc=1
    )

    las.classification = idx
    las.write(bunny.with_stem('seg_bunny'))
