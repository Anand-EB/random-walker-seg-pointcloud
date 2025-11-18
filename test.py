from pathlib import Path

import laspy
import numpy as np

from pc_rwalker import random_walker_segmentation, random_walker_segmentation_gc
from time import perf_counter


def run_benchmark():
    n_neighbors = 25
    # seed_sets = [
    #     [
    #         [240],        # right ear
    #         [2825],       # left ear
    #         [164],        # snout
    #         [152, 6697],  # body front ant back
    #         [5053],       # right foot
    #         [9665],       # left foot
    #         [3445],       # tail
    #     ],
    #     [
    #         [100], [500], [1000], [1500],
    #         [2000], [2500], [3000]
    #     ],
    #     [
    #         [405], [3221, 4500], [1200],
    #         [90], [7500], [8000], [9000]
    #     ],
    # ]

    seed_sets = [
        [
            [240],        # right ear
            [2825],       # left ear
            [164],        # snout
            [152, 6697],  # body front ant back
            [5053],       # right foot
            [9665],       # left foot
            [3445],       # tail
        ],
        [
            [100], [500], [1000], [1500],
            [2000], [2500], [3000]
        ],
        [
            [405], [3221, 4500], [1200],
            [90], [7500], [8000], [9000]
        ],
    ]

    bunny = Path(__file__).parent / 'example/08oc6829_subsample.las'
    las = laspy.read(bunny)



    for run_idx, bunny_seeds in enumerate(seed_sets, start=1):
        print(f"\n=== Benchmark Run {run_idx} ({len(bunny_seeds)} seed groups) ===")

        las.classification = np.zeros_like(las.classification)
        for i, s in enumerate(bunny_seeds, start=1):
            las.classification[s] = i

        las.write(bunny.with_stem(f'seed_bunny_{run_idx}'))

        print("Running Geometry-central algorithm...")
        start_time = perf_counter()
        idx_new = random_walker_segmentation_gc(
            las.xyz, bunny_seeds, n_neighbors
        )
        end_time = perf_counter()
        print(f"Geometry-central algorithm: {end_time - start_time:.6f} seconds")

        print("Running Legacy algorithm...")
        las.classification = idx_new
        las.write(bunny.with_stem(f'seg_bunny_gc_{run_idx}'))

        start_time = perf_counter()
        idx = random_walker_segmentation(
            las.xyz, bunny_seeds, n_neighbors, n_proc=1
        )
        end_time = perf_counter()
        print(f"Legacy algorithm: {end_time - start_time:.6f} seconds")

        las.classification = idx
        las.write(bunny.with_stem(f'seg_bunny_{run_idx}'))

        


if __name__ == '__main__':
    run_benchmark()
