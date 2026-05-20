import math
from typing import Dict, Tuple

from config import CFG
from belief_map import BeliefMap, GridValue
from utils import normalize2, point_in_strip, inside_geofence, geofence_penalty


class SixGridPlanner:
    def __init__(self, agent_name: str, strip: Dict[str, float], belief_map: BeliefMap):
        self.agent_name = agent_name
        self.strip = strip
        self.belief_map = belief_map
        self.blocked_cells = set()
        self.history = []
        self.preferred_vx = 1.0
        self.preferred_vy = 0.0

    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        size = CFG.PLANNER_CELL_SIZE
        return int(math.floor((x - CFG.MAP_X_MIN) / size)), int(math.floor((y - CFG.MAP_Y_MIN) / size))

    def cell_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        size = CFG.PLANNER_CELL_SIZE
        cx, cy = cell
        return CFG.MAP_X_MIN + (cx + 0.5) * size, CFG.MAP_Y_MIN + (cy + 0.5) * size

    def direction_candidates(self):
        for deg in CFG.SIX_GRID_ANGLES_DEG:
            rad = math.radians(deg)
            yield math.cos(rad), math.sin(rad), deg

    def mark_blocked_cell(self, cell):
        self.blocked_cells.add(cell)

    def mark_front_obstacle(self, x: float, y: float, yaw_deg: float, obstacle_dist: float = None):
        if obstacle_dist is None:
            obstacle_dist = CFG.OBSTACLE_DIST
        max_mark = float(getattr(CFG, "OBSTACLE_MAP_MARK_MAX_DIST", 7.0))
        if float(obstacle_dist) > max_mark:
            return
        rad = math.radians(yaw_deg)
        mark_dist = min(float(obstacle_dist), max_mark)
        ox = x + math.cos(rad) * mark_dist
        oy = y + math.sin(rad) * mark_dist
        cell = self.cell_of(ox, oy)
        self.blocked_cells.add(cell)
        self.belief_map.mark_radius(ox, oy, GridValue.OBSTACLE, radius=CFG.OBSTACLE_MARK_RADIUS)

    def agent_penalty(self, hub, x: float, y: float, z: float) -> float:
        penalty = 0.0
        with hub.lock:
            positions = dict(hub.positions)
        for other, pos in positions.items():
            if other == self.agent_name or other == CFG.COMMAND:
                continue
            ox, oy, oz = pos
            if abs(z - oz) > getattr(CFG, "PROXIMITY_Z_IGNORE_DISTANCE", 1.2):
                continue
            d = math.hypot(x - ox, y - oy)
            if d < CFG.REPULSION_DISTANCE:
                penalty += (CFG.REPULSION_DISTANCE - d) / CFG.REPULSION_DISTANCE
        return penalty

    def strip_center_reward(self, x: float, y: float, active_strip=None) -> float:
        if active_strip is None:
            active_strip = self.strip
        y_center = (active_strip["y_min"] + active_strip["y_max"]) / 2.0
        half_h = max(1e-6, (active_strip["y_max"] - active_strip["y_min"]) / 2.0)
        d = abs(y - y_center) / half_h
        return max(0.0, 1.0 - d)

    def front_obstacle_penalty(self, action_dx: float, action_dy: float, yaw_deg: float, snapshot) -> float:
        if not snapshot.get("obstacle", False):
            return 0.0
        front_clear = float(snapshot.get("front_clear", 100.0))
        max_mark = float(getattr(CFG, "OBSTACLE_MAP_MARK_MAX_DIST", 7.0))
        if front_clear > max_mark + 2.0:
            return 0.0
        yaw_rad = math.radians(yaw_deg)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        ax, ay = normalize2(action_dx, action_dy)
        front_align = ax * fx + ay * fy
        return 1.0 if front_align > 0.65 else 0.0

    def choose_next_cell(self, x: float, y: float, z: float, yaw_deg: float, snapshot: Dict, hub):
        current_cell = self.cell_of(x, y)
        active_strip = hub.get_agent_area(self.agent_name) if hasattr(hub, "get_agent_area") else None
        if active_strip is None:
            active_strip = self.strip

        if not self.history or self.history[-1] != current_cell:
            self.history.append(current_cell)
            recent_mem = max(20, int(getattr(CFG, "RECENT_CELL_MEMORY", 18) * 3))
            if len(self.history) > recent_mem:
                self.history = self.history[-recent_mem:]

        goal = hub.get_global_goal(self.agent_name)
        goal_ux = goal_uy = 0.0
        if goal is not None:
            goal_ux, goal_uy = normalize2(goal[0] - x, goal[1] - y)
            old_goal_dist = math.hypot(goal[0] - x, goal[1] - y)
        else:
            old_goal_dist = 0.0

        candidates = []
        for dx, dy, deg in self.direction_candidates():
            nx = x + dx * CFG.PLANNER_CELL_SIZE
            ny = y + dy * CFG.PLANNER_CELL_SIZE
            cand_cell = self.cell_of(nx, ny)
            cx, cy = self.cell_center(cand_cell)
            if not inside_geofence(cx, cy):
                continue
            if hasattr(hub, "point_inside_dynamic_bounds") and not hub.point_inside_dynamic_bounds(cx, cy, margin=CFG.AREA_MARGIN):
                continue

            out_of_strip = 0.0 if point_in_strip(cx, cy, active_strip) else 1.0
            blocked = 1.0 if cand_cell in self.blocked_cells else 0.0
            cell_value = self.belief_map.get_value_world(cx, cy)
            episode_cell_value = self.belief_map.get_episode_value_world(cx, cy) if hasattr(self.belief_map, "get_episode_value_world") else cell_value
            unknown_radius = float(getattr(CFG, "PLANNER_UNKNOWN_SCORE_RADIUS", CFG.PLANNER_CELL_SIZE * 2.0))
            unknown_ratio = self.belief_map.unknown_ratio_radius(cx, cy, radius=unknown_radius)

            visited_penalty = 1.0 if cell_value == GridValue.VISITED else 0.0
            episode_visited_cell_penalty = 1.0 if episode_cell_value in (GridValue.VISITED, GridValue.TARGET) else 0.0
            episode_visited_area_ratio = 0.0
            if hasattr(self.belief_map, "episode_visited_ratio_radius"):
                episode_visited_area_ratio = self.belief_map.episode_visited_ratio_radius(
                    cx,
                    cy,
                    radius=getattr(CFG, "EPISODE_VISITED_AVOID_RADIUS", CFG.PLANNER_CELL_SIZE),
                )

            recent_mem = int(getattr(CFG, "RECENT_CELL_MEMORY", 18))
            recent_cells = self.history[-recent_mem:] if recent_mem > 0 else []
            recent_penalty = 1.0 if cand_cell in recent_cells else 0.0

            obstacle_penalty = 1.0 if cell_value == GridValue.OBSTACLE else 0.0

            pdx, pdy = normalize2(self.preferred_vx, self.preferred_vy)
            direction_reward = max(0.0, dx * pdx + dy * pdy)
            goal_align = max(0.0, dx * goal_ux + dy * goal_uy) if goal is not None else 0.0
            goal_progress = 0.0
            if goal is not None:
                new_goal_dist = math.hypot(goal[0] - cx, goal[1] - cy)
                goal_progress = max(-1.0, min(1.0, (old_goal_dist - new_goal_dist) / max(CFG.PLANNER_CELL_SIZE, 1e-6)))

            score = 0.0
            score += CFG.R_UNKNOWN * unknown_ratio
            score += CFG.R_DIRECTION * direction_reward
            score += CFG.R_STRIP_CENTER * self.strip_center_reward(cx, cy, active_strip)
            score += CFG.R_GLOBAL_GOAL * goal_align
            score += CFG.R_GOAL_DISTANCE * goal_progress
            score -= CFG.P_VISITED * visited_penalty
            score -= getattr(CFG, "P_EPISODE_VISITED_CELL", 8.0) * episode_visited_cell_penalty
            score -= getattr(CFG, "P_EPISODE_VISITED_AREA", 12.0) * episode_visited_area_ratio
            score -= getattr(CFG, "P_RECENT_CELL", 5.0) * recent_penalty
            score -= CFG.P_OBSTACLE * obstacle_penalty
            score -= CFG.P_GEOFENCE * geofence_penalty(cx, cy)
            score -= CFG.P_AGENT * self.agent_penalty(hub, cx, cy, z)
            score -= CFG.P_BLOCKED * blocked
            score -= CFG.P_FRONT_OBSTACLE * self.front_obstacle_penalty(dx, dy, yaw_deg, snapshot)
            score -= CFG.P_OUT_OF_STRIP * out_of_strip

            candidates.append({"cell": cand_cell, "xy": (cx, cy), "score": score, "dir": (dx, dy), "deg": deg})

        if candidates:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            best = candidates[0]
            if best["score"] >= CFG.BACKTRACK_SCORE_THRESHOLD:
                self.preferred_vx, self.preferred_vy = best["dir"]
                return best["cell"], best["xy"], f"goal_greedy deg={best['deg']:.0f} score={best['score']:.2f}"

        # Backtracking은 최후 수단이다. 가능하면 episode 내에서 덜 방문한 이전 셀을 선택한다.
        backtrack_candidates = []
        while len(self.history) > 1:
            prev_cell = self.history.pop()
            if prev_cell == current_cell:
                continue
            px, py = self.cell_center(prev_cell)
            if not inside_geofence(px, py):
                continue
            if hasattr(hub, "point_inside_dynamic_bounds") and not hub.point_inside_dynamic_bounds(px, py, margin=CFG.AREA_MARGIN):
                continue
            ratio = 0.0
            if hasattr(self.belief_map, "episode_visited_ratio_radius"):
                ratio = self.belief_map.episode_visited_ratio_radius(
                    px,
                    py,
                    radius=getattr(CFG, "EPISODE_VISITED_AVOID_RADIUS", CFG.PLANNER_CELL_SIZE),
                )
            backtrack_candidates.append((ratio, prev_cell, px, py))
            if len(backtrack_candidates) >= 8:
                break

        if backtrack_candidates:
            backtrack_candidates.sort(key=lambda item: item[0])
            _, prev_cell, px, py = backtrack_candidates[0]
            dx, dy = px - x, py - y
            self.preferred_vx, self.preferred_vy = normalize2(dx, dy)
            return prev_cell, (px, py), "backtracking_less_visited"

        if goal is not None:
            fallback_cell = self.cell_of(goal[0], goal[1])
            return fallback_cell, goal, "fallback_global_goal"

        sx = (active_strip["x_min"] + active_strip["x_max"]) / 2.0
        sy = (active_strip["y_min"] + active_strip["y_max"]) / 2.0
        fallback_cell = self.cell_of(sx, sy)
        return fallback_cell, (sx, sy), "fallback_strip_center"
