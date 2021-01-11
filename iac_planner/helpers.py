from dataclasses import dataclass
from typing import TypeVar, Callable, List
import numpy as np



@dataclass
class Weights:
    cte: float = 1
    vel: float = 1
    slope: float = 1


@dataclass
class VelParams:
    m: float = 730.0
    mu: float = 0.6
    rolling_friction: float = 100.0
    d: float =4
    tempVel: float= 0.0
    tmp: float = 0.0
    power: float = 3140
    downforceCoeff: float= 0.965
    alphar: float = 0.0
    Calpha: float = 867
    dragCoeff: float = 0.5
    maxPower: float= 294000
    FintoV: float = 294000
    back: float = 10



@dataclass
class CollisionParams:
    circle_offset = 0
    circle_radii = 1
    growth_factor_a = 0
    growth_factor_b = 0


state_t = TypeVar('state_t')  # np.ndarray[[3], float]
path_t = TypeVar('path_t')  # np.ndarray[[-1, 2], float]


@dataclass
class Env:
    # nh: Node
    # m_pub: Publisher = None

    weights: Weights = Weights()
    vel_params: VelParams = VelParams()
    collision_params: CollisionParams = CollisionParams()
    info: Callable[[str], None] = None

    state: state_t = np.zeros(4)  # [x, y, theta, v]
    path: path_t = None  # [ [x, y], ... ]

    obstacles: np.ndarray = np.zeros((0, 2))
    other_vehicle_states: List[state_t] = ()
