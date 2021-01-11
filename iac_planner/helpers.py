from dataclasses import dataclass
from typing import TypeVar, Callable, List
import numpy as np

from iac_planner.path_sampling.global_path_handler import GlobalPathHandler
from iac_planner.path_sampling.types import RoadLinePolynom


@dataclass
class Weights:
    cte: float = 1
    vel: float = 1
    slope: float = 1


@dataclass
class VelParams:
    m: float = 730.0
    mu: float = 0.6
    rolling_friction: float = 65.0


@dataclass
class CollisionParams:
    circle_offset = 0
    circle_radii = 1
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

    state: state_t = np.zeros(4)  # [x, y, theta, v]
    path: path_t = None  # [ [x, y], ... ]

    obstacles: np.ndarray = np.zeros((0, 2))
    other_vehicle_states: List[state_t] = ()

    left_poly: RoadLinePolynom = None
    right_poly: RoadLinePolynom = None

    global_path_handler: GlobalPathHandler = GlobalPathHandler()