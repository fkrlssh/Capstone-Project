"""
Rebuild presentation union PNG from saved episode grids (offline).

Usage (from 10_test):
  python merge_presentation_grid.py
  python merge_presentation_grid.py --exclude-latest
"""
import argparse

from config import CFG
from belief_map import BeliefMap


def main():
    parser = argparse.ArgumentParser(description="Merge episode_*_episode_grid.csv into one PNG.")
    parser.add_argument(
        "--out-dir",
        default=CFG.OUTPUT_DIR,
        help="outputs directory (default: config OUTPUT_DIR)",
    )
    parser.add_argument(
        "--exclude-latest",
        action="store_true",
        help="exclude the highest-numbered saved episode",
    )
    args = parser.parse_args()

    exclude = None
    if args.exclude_latest:
        import glob
        import os
        import re

        episode_dir = os.path.join(args.out_dir, getattr(CFG, "EPISODE_MAP_DIR", "episodes"))
        paths = glob.glob(os.path.join(episode_dir, "episode_*_episode_grid.csv"))
        nums = []
        for p in paths:
            m = re.search(r"episode_(\d+)_episode_grid", os.path.basename(p))
            if m:
                nums.append(int(m.group(1)))
        if nums:
            exclude = max(nums)

    stats = BeliefMap.save_presentation_union(args.out_dir, exclude_episode=exclude)
    if stats is None:
        print("[MAP] no episode grids found to merge.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
