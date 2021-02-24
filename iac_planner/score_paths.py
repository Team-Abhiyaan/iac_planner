from math import inf
from typing import Iterable, Tuple, Callable, Optional

import numpy as np

from iac_planner.collision_check import CollisionChecker
# from iac_planner.generate_markers import visualize
from iac_planner.generate_velocity_profile import generate_velocity_profile
from iac_planner.helpers import Env, path_t


def slope(path: path_t, i: int) -> float:
    return np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])


def score_paths(env: Env, paths: Iterable[path_t], max_path_len: Optional[int] = 50
                ) -> Tuple[Optional[Tuple[path_t, np.ndarray]], float]:
    best_trajectory = None
    best_cost = +inf

    collision_checker = CollisionChecker(env, max_path_len, time_step=0.5)
    for index, path in enumerate(paths):
        path = path[:]
        cost = 0.0
        costs = np.zeros(len(path))  # store individual costs for visualization

        vel_profile = generate_velocity_profile(env, path)

        if not collision_checker.check_collisions(path):
            continue

        # Assume the path is a set of line segments
        global_path_index = 0  # The closes segment to the current point on the candidate path
        distance_func = get_distance_func(env.path, global_path_index)  # To get distance of point from nearest segment

        for j in range(len(path)):
            pt = path[j]

            # Shift to the 'closest' segment of the path
            # `get_normal(env.path, global_path_index)(*pt) > 0`
            # checks if `pt` is in the forward direction relative to the normal
            while global_path_index != len(env.path) - 1 and get_normal(env.path, global_path_index)(*pt) > 0:
                global_path_index += 1
                distance_func = get_distance_func(env.path, global_path_index)

            # Cross Track Error
            costs[j] += env.weights.cte * np.abs(distance_func(*pt))

            # Velocity Term
            if j != len(path) - 1:
                costs[j] += env.weights.vel * 1.0 / (vel_profile[j] + 1.0)

            # Slope term:
            if j != len(path) - 1:
                costs[j] += env.weights.slope * np.abs(slope(path, j) - slope(env.path, global_path_index))

            cost += costs[j]

        # Visualize
        # visualize(env.m_pub, env.nh.get_clock(), index, path, weights=costs, weight_max=3)

        if cost < best_cost:
            best_trajectory = path, vel_profile
            best_cost = cost

    return best_trajectory, best_cost


def get_normal(path: path_t, i: int) -> Callable[[float, float], float]:
    """
    Generates Normal to path at vertex i
    :param path: Path
    :param i: index of vertex
    :return: function which evaluates Ax + By + C, positive values indicates point is towards vertex i + 1
    """
    vertex = path[i]
    # Slope of normal to the required curve
    theta = np.arctan2(path[i + 1][1] - path[i - 1][1], path[i + 1][0] - path[i - 1][0])

    def ret(x: float, y: float) -> float:
        return (x - vertex[0]) * np.cos(theta) + (y - vertex[1]) * np.sin(theta)

    return ret


def get_distance_func(path: path_t, i: int) -> Callable[[float, float], float]:
    """
    Generates line between vertex i and vertex i + 1
    :param path: Path
    :param i: index of vertex
    :return: function which evaluates Ax + By + C, here sign has no meaning
    """

    if i >= len(path) - 1:
        # TODO: Choose a better constant
        return lambda x, y: 4

    vertex = path[i]
    # Slope of segment
    theta = slope(path, i)

    def ret(x: float, y: float) -> float:
        return (x - vertex[0]) * np.sin(theta) - (y - vertex[1]) * np.cos(theta)

    return ret


"""
MapGridCostFunction This cost function class rates a trajectory based on how closely the trajectory follows a 
global path or approaches a goal point. This attempts to optimize calculation speed by using the same precomputed map 
of distances to a path or goal point for all trajectories. 

In the dwa_local_planner, this cost function is instantiated multiple times, for different purposes. To keep the 
trajectory close to the path, to make the robot advance towards a local goal, and also to make the robot front (
"nose") point towards a local goal. This cost function is a heuristics that can give bad results or even fail with 
unsuitable parameters. 
"""

"""
~<name>/pdist_scale (double, default: 0.6) The weighting for how much the controller should stay close to the path 
it was given, maximal possible value is 5.0 
~<name>/gdist_scale (double, default: 0.8) The weighting for how much the controller should attempt to reach its local
goal, also controls speed, maximal possible value is 5.0 
~<name>/occdist_scale (double, default: 0.01) The weighting for how much the controller should attempt to avoid 
obstacles
"""
