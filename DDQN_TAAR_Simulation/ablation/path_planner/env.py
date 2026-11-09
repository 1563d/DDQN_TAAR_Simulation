# path_planner/env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class GridWorld3DEnv(gym.Env):
    """
    # 风机巡检场景（与你提供的实现一致，支持 cube1/cube2 参数）
    - grid indexed as grid[z, y, x]
    - cube1/cube2: dict or None, shape {"min":(x0,y0,z0), "max":(x1,y1,z1), "value":int}
    """
    metadata = {}

    def __init__(self, size_x=60, size_y=80, size_z=27,
                 start=None, goal=None, max_steps=None,
                 scene: str = "wt_onshore_cabinets",
                 seed=None,
                #  cube1={"min": (12,60,0), "max": (15,66,6), "value": 1},
                 cube1=None,
                 cube2=None):
        super().__init__()
        self.size_x, self.size_y, self.size_z = int(size_x), int(size_y), int(size_z)
        self.scene = scene
        self.rng = np.random.RandomState(0 if seed is None else seed)

        # 体素网格：grid[z, y, x]
        self.grid = np.zeros((self.size_z, self.size_y, self.size_x), dtype=np.int8)

        # 构建场景
        p = self._build_scene()
        self.scene_params = p

        # 起点/终点/阶段点
        self.start = tuple(start) if start else (p["start_xy"][0], p["start_xy"][1], 0)
        self.goal = tuple(goal) if goal else tuple(p["inspect_point"])
        self.waypoint = tuple(p["waypoint"])
        self.stage_points = [self.waypoint]

        # 关键几何信息
        self.tower_base = p["tower_base"]
        self.tower_height = p["h_tower"]
        self.wall_inner = p["wall_inner"]

        # 邻域观测缓冲
        self._refresh_padded_grid()

        # 动作空间（26 邻域）
        self.moves = [(dx, dy, dz)
                      for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1)
                      for dz in (-1, 0, 1)
                      if not (dx == 0 and dy == 0 and dz == 0)]
        self.action_space = spaces.Discrete(len(self.moves))

        # 观测空间：46 维（20 + 26）
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(46,), dtype=np.float32)

        # 步数限制
        self.max_steps = max_steps if max_steps else 5 * (self.size_x + self.size_y + self.size_z)

        # 固定两个立方体配置（可为 None）
        self.cube1 = cube1
        self.cube2 = cube2 if cube2 is not None else None

        # 基线占据（用于外部基线快照可参考）
        self._baseline_grid = self.grid.copy()

        # 可视化（z,y,x）：新增障碍掩码
        self.added_obstacles_mask = None

        # 初始化环境（reset）
        self.reset()

    # ===================== 场景构建 =====================
    def _build_scene(self):
        """构建风机场景（尽量与原实现保持一致）"""
        Z, Y, X = self.size_z, self.size_y, self.size_x

        # 塔筒参数（3x3 正方柱）
        tower_cx = int(X * 0.5)
        tower_cy = int(Y * 0.5)
        h_tower = 20

        x0 = max(0, tower_cx - 1)
        x1 = min(X, tower_cx + 2)
        y0 = max(0, tower_cy - 1)
        y1 = min(Y, tower_cy + 2)
        # 注意：这些 y 索引含 -30 偏移（仿你原实现）
        self.grid[0:h_tower, max(0,y0-30):max(0,y1-30), x0:x1] = 1

        # 轮毂与三叶片（薄片近似）
        hub_x = tower_cx
        hub_z = h_tower - 2
        hub_y = min(Y - 2, tower_cy + 2)
        self._create_blade(hub_x, max(0,hub_y-30), hub_z, angle=50,  fill_val=2)
        self._create_blade(hub_x, max(0,hub_y-30), hub_z, angle=155, fill_val=3)
        self._create_blade(hub_x, max(0,hub_y-30), hub_z, angle=295, fill_val=4)

        # 围墙（正方形，高6，厚1，与塔筒间隔4格）
        wall_gap = 4
        wall_h = min(6, Z)
        wall_th = 1

        ix0 = max(0, x0 - wall_gap)
        ix1 = min(X - 1, x1 - 1 + wall_gap)
        iy0 = max(0, y0 - wall_gap)
        iy1 = min(Y - 1, y1 - 1 + wall_gap)

        ox0 = max(0, ix0 - wall_th)
        ox1 = min(X - 1, ix1 + wall_th)
        oy0 = max(0, iy0 - wall_th)
        oy1 = min(Y - 1, iy1 + wall_th)

        # 四条墙（注意 y 有偏移 -30，保留你的原实现偏移行为以兼容现有数据）
        self.grid[0:wall_h, max(0,oy0-30):max(0,oy0+1-30), ox0:ox1+1] = 1
        self.grid[0:wall_h, max(0,oy1-30):max(0,oy1+1-30), ox0:ox1+1] = 1
        self.grid[0:wall_h, max(0,oy0-30):max(0,oy1+1-30), ox0:ox0+1] = 1
        self.grid[0:wall_h, max(0,oy0-30):max(0,oy1+1-30), ox1:ox1+1] = 1

        # 设备柜（围墙外左下）
        cab_size = 3
        cab_h = 6
        cab_x0 = 3
        cab_y0 = 56
        cab_x1 = cab_x0 + cab_size
        cab_y1 = cab_y0 + cab_size

        if cab_x1 <= X and cab_y1 <= Y:
            self.grid[0:cab_h, cab_y0:cab_y1, cab_x0:cab_x1] = 1
            waypoint = (7, 58, 0)
        else:
            waypoint = (max(0, ox0 - 3), oy0 + 2, 0)

        # 障碍物（固定）
        # 保留原有写法（含 y 偏移），以与你先前数据兼容
        self.grid[0:6, max(0,64):min(self.size_y,67), min(self.size_x,36):min(self.size_x,50)] = 1
        # 上面做了安全裁剪以避免越界（原代码使用切片常量，可能越界）
        # 其他块按原样近似写入（确保索引合法）
        if self.size_y > 76 and self.size_x > 27:
            self.grid[0:6, 70:76, 24:27] = 1
        if self.size_y > 51 and self.size_x > 24:
            self.grid[0:6, 48:51, 12:24] = 1
        if self.size_y > 40 and self.size_x > 42:
            self.grid[0:6, 34:40, 36:42] = 1
        if self.size_y > 35 and self.size_x > 12:
            self.grid[0:6, 32:35, 0:12] = 1

        # 起点（柜子附近）与终点
        start_xy = (54, 74)
        inspect_point = (30, 40-30, 21)  # 保持你的原值（可能含负 y 偏移）

        return {
            "tower_base": (tower_cx, tower_cy, 0),
            "h_tower": h_tower,
            "hub_position": (hub_x, hub_y, hub_z),
            "waypoint": waypoint,
            "start_xy": start_xy,
            "inspect_point": inspect_point,
            "wall_inner": (ix0, ix1, iy0, iy1)
        }

    def _create_blade(self, hub_x, hub_y, hub_z, angle, fill_val,
                      length=6, width=1, thickness=1):
        import math
        L, W, T = float(length), float(width), float(thickness)
        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)

        x_end = hub_x + L * c
        z_end = hub_z + L * s
        margin = W * 0.5 + 2.0

        xx0 = max(0, int(np.floor(min(hub_x, x_end) - margin)))
        xx1 = min(self.size_x - 1, int(np.ceil(max(hub_x, x_end) + margin)))
        zz0 = max(0, int(np.floor(min(hub_z, z_end) - margin)))
        zz1 = min(self.size_z - 1, int(np.ceil(max(hub_z, z_end) + margin)))
        yy0 = max(0, int(hub_y))
        yy1 = min(self.size_y - 1, int(hub_y + T - 1))

        half_w = W * 0.5 - 1e-6
        eps_u = 1e-6

        for y in range(yy0, yy1 + 1):
            cy = (y + 0.5) - hub_y
            if not (0.0 <= cy < T):
                continue
            for z in range(zz0, zz1 + 1):
                cz = (z + 0.5) - hub_z
                for x in range(xx0, xx1 + 1):
                    cx = (x + 0.5) - hub_x
                    u = c * cx + s * cz
                    v = -s * cx + c * cz
                    if (-half_w <= v <= half_w) and (-eps_u <= u < L - eps_u):
                        if self.grid[z, y, x] == 0:
                            self.grid[z, y, x] = int(fill_val)

    # ===================== 固定障碍工具 =====================
    def _write_cube(self, grid, cube_cfg):
        if not cube_cfg:
            return
        x0, y0, z0 = map(int, cube_cfg["min"])
        x1, y1, z1 = map(int, cube_cfg["max"])
        val = int(cube_cfg.get("value", 1))

        # 边界裁剪（注意：切片的结束索引不包含）
        x0 = max(0, min(self.size_x, x0)); x1 = max(0, min(self.size_x, x1))
        y0 = max(0, min(self.size_y, y0)); y1 = max(0, min(self.size_y, y1))
        z0 = max(0, min(self.size_z, z0)); z1 = max(0, min(self.size_z, z1))
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return

        grid[z0:z1, y0:y1, x0:x1] = val

    def apply_fixed_obstacles(self, clear_before_apply=True):
        """
        将 cube1、cube2 写入当前 grid。
        clear_before_apply=True 表示先清空再写入（仅保留基础场景+两个方块）。
        """
        prev_occ = (self.grid > 0)

        if clear_before_apply:
            # 先恢复基础场景轮廓，再写方块
            self.grid.fill(0)
            p = self._build_scene()
            self.scene_params = p

        self._write_cube(self.grid, self.cube1)
        self._write_cube(self.grid, self.cube2)

        cur_occ = (self.grid > 0)
        diff = np.logical_and(cur_occ, ~prev_occ)
        self.added_obstacles_mask = diff.astype(np.bool_)
        self._refresh_padded_grid()

    # ===================== 工具函数 =====================
    def get_static_occupancy_snapshot(self):
        """返回当前占据布尔体素（供行为层作为基线）"""
        return (self.grid.astype(np.uint8) > 0).copy()

    def _refresh_padded_grid(self):
        self.grid_pad = np.pad(self.grid, ((1, 1), (1, 1), (1, 1)),
                               mode='constant', constant_values=1)

    def _norm_state(self, pos):
        """归一化观测（46 维）"""
        x, y, z = pos

        if not getattr(self, "wp_reached", False):
            gx, gy, gz = self.waypoint
        else:
            gx, gy, gz = self.goal

        dx = gx - x
        dy = gy - y
        dz = gz - z
        euclidean_dist = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
        manhattan_dist = abs(dx) + abs(dy) + abs(dz)

        dir_x = dx / euclidean_dist
        dir_y = dy / euclidean_dist
        dir_z = dz / euclidean_dist

        x_aligned = float(x == gx)
        y_aligned = float(y == gy)
        z_aligned = float(z == gz)

        should_move_x = float(abs(dx) > abs(dy)) if z == gz else 0.0
        should_move_y = float(abs(dy) >= abs(dx)) if z == gz else 0.0
        should_move_z = float(z != gz)

        on_ground = float(z == 0)
        ground_blocked = 0.0
        if z == 0:
            blocked_count = 0
            for check_dx, check_dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + check_dx, y + check_dy
                if (0 <= nx < self.size_x and 0 <= ny < self.size_y):
                    if self.grid[0, ny, nx] > 0:
                        blocked_count += 1
                else:
                    blocked_count += 1
            ground_blocked = blocked_count / 4.0

        pos_goal_dir = np.array([
            x / (self.size_x - 1 + 1e-9),
            y / (self.size_y - 1 + 1e-9),
            z / (self.size_z - 1 + 1e-9),
            gx / (self.size_x - 1 + 1e-9),
            gy / (self.size_y - 1 + 1e-9),
            gz / (self.size_z - 1 + 1e-9),
            dir_x, dir_y, dir_z,
            min(euclidean_dist / 50.0, 1.0),
            min(manhattan_dist / 100.0, 1.0),
            float(getattr(self, "wp_reached", False)),
            on_ground,
            ground_blocked,
            x_aligned, y_aligned, z_aligned,
            should_move_x, should_move_y, should_move_z,
        ], dtype=np.float32)

        # 3x3x3 邻域（去中心）
        cz, cy, cx = int(z) + 1, int(y) + 1, int(x) + 1
        nb = self.grid_pad[cz-1:cz+2, cy-1:cy+2, cx-1:cx+2].reshape(-1)
        local = np.delete(nb, 13)
        local = (local > 0).astype(np.float32)

        return np.concatenate([pos_goal_dir, local]).astype(np.float32)

    def _distance_to_target(self, pos, target):
        dx = pos[0] - target[0]
        dy = pos[1] - target[1]
        dz = pos[2] - target[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if np.isinf(dist) or np.isnan(dist):
            return 1e6
        return float(dist)

    def _is_inside_wall(self, x, y):
        ix0, ix1, iy0, iy1 = self.wall_inner
        return ix0 <= x <= ix1 and iy0 <= y <= iy1

    def _distance_to_wall_edge(self, x, y):
        ix0, ix1, iy0, iy1 = self.wall_inner
        ox0 = max(0, ix0 - 1)
        ox1 = min(self.size_x - 1, ix1 + 1)
        oy0 = max(0, iy0 - 1)
        oy1 = min(self.size_y - 1, iy1 + 1)

        if ix0 <= x <= ix1 and iy0 <= y <= iy1:
            dx = min(x - ix0, ix1 - x)
            dy = min(y - iy0, iy1 - y)
            return -max(dx, dy)

        if x < ox0:
            dx = ox0 - x
        elif x > ox1:
            dx = x - ox1
        else:
            dx = 0

        if y < oy0:
            dy = oy0 - y
        elif y > oy1:
            dy = y - oy1
        else:
            dy = 0

        return max(dx, dy)

    def _has_ground_path(self, start_xy, target_xy, max_expansions=4000):
        """z=0 平面 4-邻域 BFS：判断是否存在地面可达路径"""
        sx, sy = int(start_xy[0]), int(start_xy[1])
        gx, gy = int(target_xy[0]), int(target_xy[1])

        if not (0 <= sx < self.size_x and 0 <= sy < self.size_y):
            return False
        if not (0 <= gx < self.size_x and 0 <= gy < self.size_y):
            return False

        g0 = self.grid[0]
        if g0[sy, sx] > 0 or g0[gy, gx] > 0:
            return False

        q = deque([(sx, sy)])
        vis = np.zeros((self.size_y, self.size_x), dtype=np.uint8)
        vis[sy, sx] = 1
        expand = 0

        while q and expand < max_expansions:
            x, y = q.popleft()
            if x == gx and y == gy:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    if not vis[ny, nx] and g0[ny, nx] == 0:
                        vis[ny, nx] = 1
                        q.append((nx, ny))
            expand += 1
        return False

    # ===================== 主逻辑 =====================
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # 恢复到“基础场景”，然后写入两个固定立方体障碍（若存在）
        self.grid.fill(0)
        _ = self._build_scene()
        prev_occ = (self.grid > 0)

        # 写入两个立方体（如果为 None 则跳过）
        self._write_cube(self.grid, self.cube1)
        self._write_cube(self.grid, self.cube2)

        cur_occ = (self.grid > 0)
        self.added_obstacles_mask = np.logical_and(cur_occ, ~prev_occ)

        self.pos = tuple(self.start)
        self.steps = 0
        self.wp_reached = False
        self.prev_action = None
        self.last_move_dir = None

        self.best_dist_to_wp = self._distance_to_target(self.start, self.waypoint)
        self.best_dist_to_goal = float('inf')
        self.steps_since_wp = 0

        self._refresh_padded_grid()
        return self._norm_state(self.pos), {}

    def set_cubes(self, cube1=None, cube2=None, clear_before_apply=True):
        if cube1 is not None:
            self.cube1 = cube1
        if cube2 is not None:
            self.cube2 = cube2
        self.apply_fixed_obstacles(clear_before_apply=clear_before_apply)

    def step(self, action):
        """Advance the environment by one action using the paper-ablation reward.

        Reward protocol v2 removes the two sources of reward exploitation present
        in the original implementation:
        1) a blocked action receives collision penalty only (no progress/proximity
           or motion reward is accumulated on the same step); and
        2) absolute near-waypoint/near-goal bonuses are removed. Progress shaping
           depends only on the *change* in distance, so staying near the target or
           repeatedly hitting an obstacle cannot generate positive return.

        The original motion-cost coefficients and one-time waypoint/goal bonuses
        are retained to keep the revised ablation as close as possible to the
        manuscript implementation.
        """
        R_COLLISION = -5.0
        R_WP_REACH = 100.0
        R_GOAL_REACH = 300.0

        COST_GROUND_PER_M = 1.00
        COST_AIR_PER_M = 1.60
        COST_ASCEND_PER_M = 2.00
        COST_DESCEND_PER_M = 0.40
        PENALTY_DIAG_GROUND = -1.50
        PENALTY_TURN = 1.20
        PENALTY_FLY_WHEN_GROUND_EXISTS = -6.0
        BONUS_TAKEOFF_WHEN_NEEDED = +2.0
        PENALTY_HIGH_ALT_WHEN_GROUND = -0.60
        K_PROG_S1 = 2.5
        K_PROG_S2 = 3.5
        GOAL_RADIUS = 0.5
        WP_RADIUS = 1.0

        self.steps += 1
        dx, dy, dz = self.moves[action]
        px, py, pz = self.pos
        nx, ny, nz = px + dx, py + dy, pz + dz

        out_of_bounds = not (0 <= nx < self.size_x and 0 <= ny < self.size_y and 0 <= nz < self.size_z)
        hit_obstacle = False if out_of_bounds else (self.grid[nz, ny, nx] > 0)
        collision = bool(out_of_bounds or hit_obstacle)

        # IMPORTANT: collision is an exclusive reward branch.  The position does
        # not change and no distance/motion/proximity term is evaluated.
        if collision:
            reward = R_COLLISION
            if pz == 0 and hit_obstacle:
                reward -= 1.0
            done = bool(self.steps >= self.max_steps)
            info = {
                "collision": True,
                "reward_protocol": "paper_v2_progress_only",
                "wp_reached": bool(self.wp_reached),
            }
            return self._norm_state(self.pos), float(reward), done, False, info

        reward = 0.0
        done = False
        prev_pos = (px, py, pz)
        self.pos = (nx, ny, nz)
        current_move_dir = (dx, dy, dz)

        move_len = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        if nz == 0:
            reward -= COST_GROUND_PER_M * move_len
        else:
            reward -= COST_AIR_PER_M * move_len

        if dz > 0:
            reward -= COST_ASCEND_PER_M * abs(dz)
        elif dz < 0:
            reward -= COST_DESCEND_PER_M * abs(dz)

        if pz == 0 and nz == 0 and (abs(dx) == 1 and abs(dy) == 1 and dz == 0):
            reward += PENALTY_DIAG_GROUND

        if self.last_move_dir is not None:
            prev_vec = np.array(self.last_move_dir, dtype=float)
            curr_vec = np.array(current_move_dir, dtype=float)
            if np.linalg.norm(prev_vec) > 0 and np.linalg.norm(curr_vec) > 0:
                cosang = float(np.dot(prev_vec, curr_vec) / (np.linalg.norm(prev_vec) * np.linalg.norm(curr_vec)))
                cosang = np.clip(cosang, -1.0, 1.0)
                angle = float(np.arccos(cosang))
                reward -= PENALTY_TURN * (angle / np.pi)

        if not self.wp_reached:
            ground_path_exists = self._has_ground_path((px, py), (self.waypoint[0], self.waypoint[1]))
            if ground_path_exists:
                if nz > 0 or dz > 0:
                    reward += PENALTY_FLY_WHEN_GROUND_EXISTS * (1.0 if dz > 0 else 0.5)
                reward += PENALTY_HIGH_ALT_WHEN_GROUND * max(nz, 0)
            elif pz == 0 and dz > 0:
                reward += BONUS_TAKEOFF_WHEN_NEEDED

        if not self.wp_reached:
            d_prev = self._distance_to_target(prev_pos, self.waypoint)
            d_now = self._distance_to_target(self.pos, self.waypoint)
            reward += K_PROG_S1 * (d_prev - d_now)

            if d_now <= WP_RADIUS:
                self.wp_reached = True
                reward += R_WP_REACH
                if nz == 0:
                    reward += 15.0
                self.last_move_dir = None
        else:
            d_prev = self._distance_to_target(prev_pos, self.goal)
            d_now = self._distance_to_target(self.pos, self.goal)
            reward += K_PROG_S2 * (d_prev - d_now)

            if d_now <= GOAL_RADIUS:
                reward += R_GOAL_REACH
                done = True

        self.last_move_dir = current_move_dir

        # No timeout-distance bonus: an unfinished episode remains a failure,
        # even if the robot is close to the goal at the step budget.
        if self.steps >= self.max_steps:
            done = True

        reward = float(np.clip(reward, -100.0, 500.0))
        info = {
            "collision": False,
            "reward_protocol": "paper_v2_progress_only",
            "wp_reached": bool(self.wp_reached),
        }
        return self._norm_state(self.pos), reward, done, False, info

