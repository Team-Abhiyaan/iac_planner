from dataclasses import dataclass
from typing import TypeVar, Callable, List

import numpy as np

from iac_planner.path_sampling._core import polyeval
from iac_planner.path_sampling.global_path_handler import GlobalPathHandler
from iac_planner.path_sampling.types import RoadLinePolynom


@dataclass
class Weights:
    cte: float = 1
    vel: float = 1
    slope: float = 1


@dataclass
class VelParams:
    m: float = 630.0
    mu: float = 0.85
    rolling_friction: float = 100.0
    d: float = 4
    tempVel: float = 0.0
    tmp: float = 0.0
    power: float = 3140
    downforceCoeff: float = 0.965
    alphar: float = 0.0
    Calpha: float = 867
    dragCoeff: float = 0.445
    maxPower: float = 314000
    FintoV: float = 314000
    back: float = 10


@dataclass
class CollisionParams:
    circle_offset = 0
    circle_radii = 0.75
    growth_factor_a = 0
    growth_factor_b = 0


@dataclass
class PathGenerationParams:
    n_long: int = 10
    n_pts_long: int = 100


state_t = TypeVar('state_t')  # np.ndarray[[3], float]
path_t = TypeVar('path_t')  # np.ndarray[[-1, 2], float]


@dataclass
class Env:
    # nh: Node
    # m_pub: Publisher = None

    weights: Weights = Weights()
    vel_params: VelParams = VelParams()
    collision_params: CollisionParams = CollisionParams()
    path_generation_params: PathGenerationParams = PathGenerationParams()

    info: Callable[[str], None] = None

    state: state_t = np.zeros(4)  # [x, y, theta, v] # Global Frame
    gear = None  # IDK datatype
    path: path_t = None  # [ [x, y], ... ]

    obstacles: np.ndarray = np.zeros((0, 2))
    other_vehicle_states: List[state_t] = ()  # Local Frame

    left_poly: RoadLinePolynom = None  # Local Frame
    right_poly: RoadLinePolynom = None  # Local Frame

    global_path_handler: GlobalPathHandler = GlobalPathHandler()

    plot_paths: bool = True

    def shift_to_ego(self, pts):  # pts: (n x 2)
        yaw = self.state[2]
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]]
        )
        return ((pts - self.state[:2]).reshape((-1, 2)) @ rot_matrix).reshape(pts.shape)

    def shift_to_global(self, pts):  # pts: (n x 2)
        yaw = self.state[2]
        rot_matrix = np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw), np.cos(-yaw)]]
        )
        return ((pts.reshape((-1, 2)) @ rot_matrix) + self.state[:2].reshape((1, 2))).reshape(pts.shape)

    def lane_to_points(self):
        if self.left_poly is not None and self.right_poly is not None:
            cl = [self.left_poly.c3, self.left_poly.c2, self.left_poly.c1, self.left_poly.c0][::-1]
            cr = [self.right_poly.c3, self.right_poly.c2, self.right_poly.c1, self.right_poly.c0][::-1]

            # Generate points on the lane boundaries
            # Lane boundary is in local frame
            x0, y0 = 0, 0  # env.state[:2]
            x_l = np.linspace(x0 - 20, x0 + 150, 40)
            x_r = np.linspace(x0 - 20, x0 + 150, 40)

            y_l = np.array([polyeval(c, cl) for c in x_l])
            y_r = np.array([polyeval(c, cr) for c in x_r])

            left = np.array([x_l, y_l]).T
            right = np.array([x_r, y_r]).T
            pts = np.vstack([left, right])

            return self.shift_to_global(pts)
        else:
            return []
