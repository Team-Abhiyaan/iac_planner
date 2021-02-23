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
    m: float = 630.0
    mu: float = 0.85
    rolling_friction: float = 100.0
    d: float =4
    tempVel: float= 0.0
    tmp: float = 0.0
    power: float = 3140
    downforceCoeff: float= 0.965
    alphar: float = 0.0
    Calpha: float = 867
    dragCoeff: float = 0.445
    maxPower: float= 314000
    FintoV: float = 314000
    back: float = 10



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
    gear = None # IDK datatype
    path: path_t = None  # [ [x, y], ... ]

    obstacles: np.ndarray = np.zeros((0, 2))
    other_vehicle_states: List[state_t] = ()

    left_poly: RoadLinePolynom = None
    right_poly: RoadLinePolynom = None

    global_path_handler: GlobalPathHandler = GlobalPathHandler()

    plot_paths: bool = True
