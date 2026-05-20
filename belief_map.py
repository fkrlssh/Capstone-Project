import os
import csv
import math
import glob
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from config import CFG


class GridValue:
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    TARGET = 3
    VISITED = 4


class BeliefMap:
    """
    Belief map with separated global / episode layers.

    Priority rule:
    - TARGET has the highest priority and is never overwritten by VISITED.
    - VISITED has higher priority than OBSTACLE.
    - VISITED can overwrite OBSTACLE.
    - OBSTACLE can never overwrite VISITED or TARGET.

    Layers:
    - global_grid : cumulative persistent map. It keeps VISITED/TARGET across episodes.
                    OBSTACLE is not accumulated globally to avoid white-map pollution.
    - episode_grid: current episode map. It keeps this episode's VISITED/OBSTACLE/TARGET.
                    This is reset at the start of each episode.
    - grid        : merged view used by planners and coverage.
                    VISITED/TARGET from global layer override episode obstacles.
    """

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, resolution: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution

        self.width = int(math.ceil((x_max - x_min) / resolution)) + 1
        self.height = int(math.ceil((y_max - y_min) / resolution)) + 1

        self.global_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.episode_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

        self.episode_index = 0

        import threading
        self.lock = threading.Lock()

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)

        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return None

        return gx, gy

    def _merged_value_at(self, xx: int, yy: int) -> int:
        g = int(self.global_grid[yy, xx])
        e = int(self.episode_grid[yy, xx])

        # Persistent global VISITED/TARGET has the highest priority.
        if g == GridValue.TARGET:
            return GridValue.TARGET
        if g == GridValue.VISITED:
            return GridValue.VISITED

        # Current episode VISITED/TARGET also override obstacle.
        if e == GridValue.TARGET:
            return GridValue.TARGET
        if e == GridValue.VISITED:
            return GridValue.VISITED
        if e == GridValue.OBSTACLE:
            return GridValue.OBSTACLE
        if e == GridValue.FREE:
            return GridValue.FREE

        # Global FREE is allowed, but global OBSTACLE is intentionally not used.
        if g == GridValue.FREE:
            return GridValue.FREE

        return GridValue.UNKNOWN

    def _refresh_cell_locked(self, xx: int, yy: int):
        self.grid[yy, xx] = self._merged_value_at(xx, yy)

    def _refresh_all_locked(self):
        # Default unknown.
        self.grid[:, :] = GridValue.UNKNOWN

        # Global free, then episode free/obstacle, then visited/target priority.
        self.grid[self.global_grid == GridValue.FREE] = GridValue.FREE
        self.grid[self.episode_grid == GridValue.FREE] = GridValue.FREE
        self.grid[self.episode_grid == GridValue.OBSTACLE] = GridValue.OBSTACLE

        self.grid[self.episode_grid == GridValue.VISITED] = GridValue.VISITED
        self.grid[self.episode_grid == GridValue.TARGET] = GridValue.TARGET
        self.grid[self.global_grid == GridValue.VISITED] = GridValue.VISITED
        self.grid[self.global_grid == GridValue.TARGET] = GridValue.TARGET

    def start_new_episode(self, episode_index: int):
        """
        Reset the current episode layer only.
        Persistent VISITED/TARGET in global_grid are preserved.
        Old episode obstacles are removed here.
        """
        with self.lock:
            self.episode_index = int(episode_index)
            self.episode_grid[:, :] = GridValue.UNKNOWN
            self._refresh_all_locked()

    def clear_episode_obstacles(self):
        """Remove only current episode obstacles. VISITED/TARGET stay."""
        with self.lock:
            self.episode_grid[self.episode_grid == GridValue.OBSTACLE] = GridValue.UNKNOWN
            self._refresh_all_locked()

    def get_value_world(self, x: float, y: float) -> int:
        idx = self.world_to_grid(x, y)

        if idx is None:
            return GridValue.OBSTACLE

        gx, gy = idx

        with self.lock:
            return int(self.grid[gy, gx])

    def mark_radius(self, x: float, y: float, value: int, radius: float):
        idx = self.world_to_grid(x, y)

        if idx is None:
            if value == GridValue.TARGET and getattr(CFG, "TARGET_MAP_MARK_LOG_ENABLED", True):
                print(
                    f"[TARGET MAP MARK] skipped (outside grid): "
                    f"x={x:.2f}, y={y:.2f} bounds "
                    f"x=[{self.x_min},{self.x_max}] y=[{self.y_min},{self.y_max}]"
                )
            return 0

        gx, gy = idx
        r = max(1, int(radius / self.resolution))
        changed = 0

        with self.lock:
            for yy in range(max(0, gy - r), min(self.height, gy + r + 1)):
                for xx in range(max(0, gx - r), min(self.width, gx + r + 1)):
                    before = int(self.grid[yy, xx])

                    if value == GridValue.VISITED:
                        # VISITED can overwrite obstacles, but it can never overwrite TARGET.
                        if self.global_grid[yy, xx] != GridValue.TARGET:
                            self.global_grid[yy, xx] = GridValue.VISITED
                        if self.episode_grid[yy, xx] != GridValue.TARGET:
                            self.episode_grid[yy, xx] = GridValue.VISITED

                    elif value == GridValue.TARGET:
                        self.global_grid[yy, xx] = GridValue.TARGET
                        self.episode_grid[yy, xx] = GridValue.TARGET

                    elif value == GridValue.OBSTACLE:
                        # OBSTACLE is episode-local and cannot overwrite VISITED/TARGET.
                        if self.global_grid[yy, xx] in (GridValue.VISITED, GridValue.TARGET):
                            self._refresh_cell_locked(xx, yy)
                            continue
                        if self.episode_grid[yy, xx] in (GridValue.VISITED, GridValue.TARGET):
                            self._refresh_cell_locked(xx, yy)
                            continue
                        self.episode_grid[yy, xx] = GridValue.OBSTACLE

                    else:
                        # FREE or other non-priority values cannot overwrite VISITED/TARGET.
                        if self.global_grid[yy, xx] in (GridValue.VISITED, GridValue.TARGET):
                            self._refresh_cell_locked(xx, yy)
                            continue
                        if self.episode_grid[yy, xx] in (GridValue.VISITED, GridValue.TARGET):
                            self._refresh_cell_locked(xx, yy)
                            continue
                        self.episode_grid[yy, xx] = value

                    self._refresh_cell_locked(xx, yy)
                    after = int(self.grid[yy, xx])
                    if after != before:
                        changed += 1

        return changed


    def episode_visited_ratio_radius(self, x: float, y: float, radius: float) -> float:
        """
        현재 episode 안에서 이미 VISITED/TARGET으로 찍힌 영역 비율.
        에이전트가 같은 episode에서 같은 공간을 반복 방문하는 것을 피하기 위한 planner penalty에 사용한다.
        """
        idx = self.world_to_grid(x, y)

        if idx is None:
            return 1.0

        gx, gy = idx
        r = max(1, int(radius / self.resolution))

        with self.lock:
            sub = self.episode_grid[
                max(0, gy - r):min(self.height, gy + r + 1),
                max(0, gx - r):min(self.width, gx + r + 1)
            ]

            if sub.size == 0:
                return 1.0

            visited = np.count_nonzero(
                (sub == GridValue.VISITED) | (sub == GridValue.TARGET)
            )

        return float(visited) / float(sub.size)

    def merged_visited_ratio_radius(self, x: float, y: float, radius: float) -> float:
        """
        global + episode가 합쳐진 현재 grid에서 VISITED/TARGET 비율.
        참고용. 실제 episode 회피는 episode_visited_ratio_radius를 우선 사용한다.
        """
        idx = self.world_to_grid(x, y)

        if idx is None:
            return 1.0

        gx, gy = idx
        r = max(1, int(radius / self.resolution))

        with self.lock:
            sub = self.grid[
                max(0, gy - r):min(self.height, gy + r + 1),
                max(0, gx - r):min(self.width, gx + r + 1)
            ]

            if sub.size == 0:
                return 1.0

            visited = np.count_nonzero(
                (sub == GridValue.VISITED) | (sub == GridValue.TARGET)
            )

        return float(visited) / float(sub.size)

    def get_episode_value_world(self, x: float, y: float) -> int:
        idx = self.world_to_grid(x, y)

        if idx is None:
            return GridValue.OBSTACLE

        gx, gy = idx

        with self.lock:
            return int(self.episode_grid[gy, gx])

    def unknown_ratio_radius(self, x: float, y: float, radius: float) -> float:
        idx = self.world_to_grid(x, y)

        if idx is None:
            return 0.0

        gx, gy = idx
        r = max(1, int(radius / self.resolution))

        with self.lock:
            sub = self.grid[
                max(0, gy - r):min(self.height, gy + r + 1),
                max(0, gx - r):min(self.width, gx + r + 1)
            ]

            if sub.size == 0:
                return 0.0

            unknown = np.count_nonzero(sub == GridValue.UNKNOWN)

        return float(unknown) / float(sub.size)

    def _coverage_subgrid(self, ignore_margin: float = 0.0):
        x0 = self.x_min + ignore_margin
        x1 = self.x_max - ignore_margin
        y0 = self.y_min + ignore_margin
        y1 = self.y_max - ignore_margin

        g0 = self.world_to_grid(x0, y0)
        g1 = self.world_to_grid(x1, y1)

        if g0 is None or g1 is None:
            return None

        gx0, gy0 = g0
        gx1, gy1 = g1
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
        gy0, gy1 = min(gy0, gy1), max(gy0, gy1)

        return gx0, gx1, gy0, gy1

    def coverage_ratio(self, ignore_margin: float = 0.0) -> float:
        stats = self.coverage_stats(ignore_margin=ignore_margin)
        return stats["observed"]

    def coverage_stats(self, ignore_margin: float = 0.0) -> Dict[str, float]:
        """
        Coverage stats from the merged current view.

        observed: UNKNOWN이 아닌 전체 비율 = VISITED + OBSTACLE + TARGET + FREE
        visited : 실제 누적 방문/관측 비율 = VISITED + TARGET
        obstacle: 현재 episode obstacle 비율. Previous-episode obstacles are not accumulated.
        unknown : 아직 미관측 비율 = UNKNOWN
        """
        bounds = self._coverage_subgrid(ignore_margin)
        if bounds is None:
            return {
                "observed": 0.0,
                "visited": 0.0,
                "obstacle": 0.0,
                "unknown": 1.0,
                "target": 0.0,
                "free": 0.0,
                "global_visited": 0.0,
                "episode_obstacle": 0.0,
            }

        gx0, gx1, gy0, gy1 = bounds

        with self.lock:
            sub = self.grid[gy0:gy1 + 1, gx0:gx1 + 1]
            gsub = self.global_grid[gy0:gy1 + 1, gx0:gx1 + 1]
            esub = self.episode_grid[gy0:gy1 + 1, gx0:gx1 + 1]
            total = sub.size
            if total <= 0:
                return {
                    "observed": 0.0,
                    "visited": 0.0,
                    "obstacle": 0.0,
                    "unknown": 1.0,
                    "target": 0.0,
                    "free": 0.0,
                    "global_visited": 0.0,
                    "episode_obstacle": 0.0,
                }

            unknown_count = int(np.count_nonzero(sub == GridValue.UNKNOWN))
            visited_count = int(np.count_nonzero(sub == GridValue.VISITED))
            obstacle_count = int(np.count_nonzero(sub == GridValue.OBSTACLE))
            target_count = int(np.count_nonzero(sub == GridValue.TARGET))
            free_count = int(np.count_nonzero(sub == GridValue.FREE))
            observed_count = total - unknown_count
            global_visited_count = int(np.count_nonzero(gsub == GridValue.VISITED)) + int(np.count_nonzero(gsub == GridValue.TARGET))
            episode_obstacle_count = int(np.count_nonzero(esub == GridValue.OBSTACLE))

        return {
            "observed": float(observed_count) / float(total),
            "visited": float(visited_count + target_count) / float(total),
            "obstacle": float(obstacle_count) / float(total),
            "unknown": float(unknown_count) / float(total),
            "target": float(target_count) / float(total),
            "free": float(free_count) / float(total),
            "global_visited": float(global_visited_count) / float(total),
            "episode_obstacle": float(episode_obstacle_count) / float(total),
        }


    def _coverage_subgrid_for_area(self, area, ignore_margin: float = 0.0):
        """Return grid bounds for an arbitrary world-coordinate area."""
        x0 = area["x_min"] + ignore_margin
        x1 = area["x_max"] - ignore_margin
        y0 = area["y_min"] + ignore_margin
        y1 = area["y_max"] - ignore_margin

        g0 = self.world_to_grid(x0, y0)
        g1 = self.world_to_grid(x1, y1)

        if g0 is None or g1 is None:
            return None

        gx0, gy0 = g0
        gx1, gy1 = g1
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
        gy0, gy1 = min(gy0, gy1), max(gy0, gy1)

        return gx0, gx1, gy0, gy1

    def coverage_stats_area(self, area, ignore_margin: float = 0.0) -> Dict[str, float]:
        """
        Coverage stats for a local search window.
        The meaning is the same as coverage_stats(), but bounded by area.
        """
        bounds = self._coverage_subgrid_for_area(area, ignore_margin)
        if bounds is None:
            return {
                "observed": 0.0,
                "visited": 0.0,
                "obstacle": 0.0,
                "unknown": 1.0,
                "target": 0.0,
                "free": 0.0,
                "global_visited": 0.0,
                "episode_obstacle": 0.0,
            }

        gx0, gx1, gy0, gy1 = bounds

        with self.lock:
            sub = self.grid[gy0:gy1 + 1, gx0:gx1 + 1]
            gsub = self.global_grid[gy0:gy1 + 1, gx0:gx1 + 1]
            esub = self.episode_grid[gy0:gy1 + 1, gx0:gx1 + 1]
            total = sub.size
            if total <= 0:
                return {
                    "observed": 0.0,
                    "visited": 0.0,
                    "obstacle": 0.0,
                    "unknown": 1.0,
                    "target": 0.0,
                    "free": 0.0,
                    "global_visited": 0.0,
                    "episode_obstacle": 0.0,
                }

            unknown_count = int(np.count_nonzero(sub == GridValue.UNKNOWN))
            visited_count = int(np.count_nonzero(sub == GridValue.VISITED))
            obstacle_count = int(np.count_nonzero(sub == GridValue.OBSTACLE))
            target_count = int(np.count_nonzero(sub == GridValue.TARGET))
            free_count = int(np.count_nonzero(sub == GridValue.FREE))
            observed_count = total - unknown_count
            global_visited_count = int(np.count_nonzero(gsub == GridValue.VISITED)) + int(np.count_nonzero(gsub == GridValue.TARGET))
            episode_obstacle_count = int(np.count_nonzero(esub == GridValue.OBSTACLE))

        return {
            "observed": float(observed_count) / float(total),
            "visited": float(visited_count + target_count) / float(total),
            "obstacle": float(obstacle_count) / float(total),
            "unknown": float(unknown_count) / float(total),
            "target": float(target_count) / float(total),
            "free": float(free_count) / float(total),
            "global_visited": float(global_visited_count) / float(total),
            "episode_obstacle": float(episode_obstacle_count) / float(total),
        }

    def _visualize_grid(self, grid: np.ndarray) -> np.ndarray:
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        vis[grid == GridValue.UNKNOWN] = (0, 0, 0)
        vis[grid == GridValue.FREE] = (120, 120, 120)
        vis[grid == GridValue.OBSTACLE] = (255, 0, 0)  # OpenCV BGR: blue obstacle dots
        vis[grid == GridValue.TARGET] = (0, 0, 255)
        vis[grid == GridValue.VISITED] = (0, 180, 0)
        return cv2.flip(vis, 0)

    def _write_grid_files(self, grid: np.ndarray, base_path_without_ext: str):
        np.save(base_path_without_ext + ".npy", grid)
        with open(base_path_without_ext + ".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(grid.tolist())
        cv2.imwrite(base_path_without_ext + ".png", self._visualize_grid(grid))

    def save_episode(self, out_dir: str, episode_index: int = None):
        if episode_index is None:
            episode_index = self.episode_index
        episode_dir = os.path.join(out_dir, getattr(CFG, "EPISODE_MAP_DIR", "episodes"))
        os.makedirs(episode_dir, exist_ok=True)
        prefix = f"episode_{int(episode_index):03d}"
        with self.lock:
            merged = self.grid.copy()
            episode = self.episode_grid.copy()
            global_grid = self.global_grid.copy()
        self._write_grid_files(episode, os.path.join(episode_dir, f"{prefix}_episode_grid"))
        self._write_grid_files(merged, os.path.join(episode_dir, f"{prefix}_merged_grid"))
        self._write_grid_files(global_grid, os.path.join(episode_dir, f"{prefix}_global_grid"))
        if getattr(CFG, "MAP_SAVE_LOG_ENABLED", False):
            print(f"[MAP] saved episode maps: {episode_dir}/{prefix}_*.png")

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        with self.lock:
            merged = self.grid.copy()
            global_grid = self.global_grid.copy()
            episode_grid = self.episode_grid.copy()

        self._write_grid_files(merged, os.path.join(out_dir, "belief_grid"))
        self._write_grid_files(global_grid, os.path.join(out_dir, "belief_global_grid"))
        self._write_grid_files(episode_grid, os.path.join(out_dir, "belief_episode_grid"))

        if getattr(CFG, "MAP_SAVE_LOG_ENABLED", False):
            print(f"[MAP] saved: {os.path.join(out_dir, 'belief_grid.npy')}")
            print(f"[MAP] saved: {os.path.join(out_dir, 'belief_grid.csv')}")
            print(f"[MAP] saved: {os.path.join(out_dir, 'belief_grid.png')}")
            print(f"[MAP] saved global/episode split maps in: {out_dir}")

    @staticmethod
    def load_grid_csv(path: str) -> np.ndarray:
        rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                rows.append([int(x) for x in row])
        return np.array(rows, dtype=np.uint8)

    @staticmethod
    def merge_episode_grids_union(
        grids: List[np.ndarray],
        skip_obstacle: bool = True,
    ) -> np.ndarray:
        """Per-cell union for presentation: TARGET > VISITED > FREE > UNKNOWN."""
        if not grids:
            return None
        h, w = grids[0].shape
        if skip_obstacle:
            prio_lut = np.array([0, 2, 0, 4, 3], dtype=np.int8)
        else:
            prio_lut = np.array([0, 2, 1, 4, 3], dtype=np.int8)
        out_lut = np.array(
            [
                GridValue.UNKNOWN,
                GridValue.OBSTACLE,
                GridValue.FREE,
                GridValue.VISITED,
                GridValue.TARGET,
            ],
            dtype=np.uint8,
        )
        union_prio = np.zeros((h, w), dtype=np.int8)
        for grid in grids:
            if grid.shape != (h, w):
                raise ValueError(f"grid shape mismatch: {grid.shape} vs {(h, w)}")
            union_prio = np.maximum(union_prio, prio_lut[grid])
        return out_lut[union_prio]

    @classmethod
    def build_presentation_union_from_episode_dir(
        cls,
        out_dir: str,
        exclude_episode: Optional[int] = None,
        skip_obstacle: bool = True,
    ) -> Tuple[Optional[np.ndarray], int]:
        episode_dir = os.path.join(out_dir, getattr(CFG, "EPISODE_MAP_DIR", "episodes"))
        pattern = os.path.join(episode_dir, "episode_*_episode_grid.csv")
        paths = sorted(glob.glob(pattern))
        if exclude_episode is not None:
            skip_token = f"episode_{int(exclude_episode):03d}_"
            paths = [p for p in paths if skip_token not in os.path.basename(p)]

        expected_h = int(math.ceil((CFG.MAP_Y_MAX - CFG.MAP_Y_MIN) / CFG.GRID_RESOLUTION)) + 1
        expected_w = int(math.ceil((CFG.MAP_X_MAX - CFG.MAP_X_MIN) / CFG.GRID_RESOLUTION)) + 1
        expected_shape = (expected_h, expected_w)

        grids = []
        for path in paths:
            try:
                npy_path = path[:-4] + ".npy" if path.endswith(".csv") else path + ".npy"
                if os.path.exists(npy_path):
                    grid = np.load(npy_path)
                else:
                    grid = cls.load_grid_csv(path)
                if grid.shape != expected_shape:
                    print(
                        f"[MAP] skip episode grid (old map size {grid.shape}, "
                        f"expected {expected_shape}): {path}"
                    )
                    continue
                grids.append(grid)
            except Exception as e:
                print(f"[MAP] skip unreadable episode grid: {path} ({e})")

        if not grids:
            return None, 0
        return cls.merge_episode_grids_union(grids, skip_obstacle=skip_obstacle), len(grids)

    @classmethod
    def save_presentation_union(
        cls,
        out_dir: str,
        exclude_episode: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        if not getattr(CFG, "PRESENTATION_UNION_ENABLED", True):
            return None

        skip_obstacle = getattr(CFG, "PRESENTATION_UNION_SKIP_OBSTACLE", True)
        union, n_episodes = cls.build_presentation_union_from_episode_dir(
            out_dir,
            exclude_episode=exclude_episode,
            skip_obstacle=skip_obstacle,
        )
        if union is None:
            return None

        base = os.path.join(out_dir, getattr(CFG, "PRESENTATION_UNION_BASENAME", "belief_presentation_union"))
        os.makedirs(out_dir, exist_ok=True)
        instance = cls.__new__(cls)
        instance.height, instance.width = union.shape
        instance._write_grid_files(union, base)

        total = float(union.size)
        visited = int(np.count_nonzero(union == GridValue.VISITED))
        target = int(np.count_nonzero(union == GridValue.TARGET))
        free = int(np.count_nonzero(union == GridValue.FREE))
        unknown = int(np.count_nonzero(union == GridValue.UNKNOWN))
        stats = {
            "episodes_merged": float(n_episodes),
            "visited": (visited + target) / total,
            "observed": (visited + target + free) / total,
            "unknown": unknown / total,
        }
        print(
            f"[MAP] presentation union: {base}.png "
            f"(episodes={n_episodes}, visited={stats['visited'] * 100:.2f}%, "
            f"observed={stats['observed'] * 100:.2f}%)"
        )
        return stats
