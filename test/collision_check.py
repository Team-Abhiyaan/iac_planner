from dataclasses import dataclass
from typing import List, Generator

import numpy as np
import matplotlib.pyplot as plt

from iac_planner.collision_check import CollisionChecker
from iac_planner.helpers import state_t, CollisionParams, path_t, VelParams
from iac_planner.path_sampling.types import RoadLinePolynom


@dataclass
class Env:
    info = print

    state: state_t = np.array([2, 2, np.pi / 6, 226])  # [x, y, theta, v]
    obstacles: np.ndarray = np.zeros((0, 2))
    other_vehicle_states: List[state_t] = (np.array(
        [5, 0, 0, 0]),)

    # https://www.desmos.com/calculator/mf0yccchqn
    left_poly: RoadLinePolynom = RoadLinePolynom(8, 0.05, -0.003, -0.00001)
    right_poly: RoadLinePolynom = RoadLinePolynom(-10, 0.1, -0.004, -0.00001)

    collision_params: CollisionParams = CollisionParams()
    vel_params: VelParams = VelParams()

    def shift_to_ego(self, pts):  # pts: (n x 2)
        yaw = env.state[2]
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]]
        )
        return ((pts - self.state[:2]).reshape((-1, 2)) @ rot_matrix).reshape(pts.shape)

    def shift_to_global(self, pts):  # pts: (n x 2)
        yaw = env.state[2]
        rot_matrix = np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw), np.cos(-yaw)]]
        )
        return ((pts.reshape((-1, 2)) @ rot_matrix) + self.state[:2].reshape((1, 2))).reshape(pts.shape)


def get_paths(state: state_t, n_paths: int = 12, n_pts: int = 32, alpha_max: float = 0.01) \
        -> Generator[path_t, None, None]:
    x0, y0, theta = state[:3]
    x = np.linspace(x0, x0 + 50, n_pts)
    for alpha in np.linspace(-alpha_max, alpha_max, n_paths):
        y = y0 + (x - x0) * np.tan(theta) + (x - x0) ** 2 * alpha
        yield np.vstack([x, y]).T


if __name__ == '__main__':
    env = Env()

    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim((env.state[0] - 10, env.state[0] + 75))
    plt.ylim((env.state[1] - 30, env.state[1] + 75))

    cc = CollisionChecker(env, path_length=20, time_step=0.5)

    # Ego
    plt.arrow(env.state[0], env.state[1],
              10 * np.cos(env.state[2]), 10 * np.sin(env.state[2]), head_width=2, label='vehicle', zorder=100)

    # Lanes
    if len(cc.obstacles) != 0:
        plt.scatter(*zip(*cc.obstacles), label='Lane Boundaries', s=5)

    for path in get_paths(env.state, 10):
        is_valid = cc.check_collisions(path)
        plt.plot(*path.T, linewidth=(2 if is_valid else 0.5), c=('green' if is_valid else 'red'), zorder=0)

    for i, (state, path) in enumerate(zip(env.other_vehicle_states,
                                          np.swapaxes(cc._other_vehicle_paths[:, :2, :], 2, 1))):
        plt.arrow(*(env.shift_to_global(state[:2])),
                  10 * np.cos(state[2] + env.state[2]), 10 * np.sin(state[2] + env.state[2]),
                  head_width=2,
                  label=f"other {i + 1}", color='black', zorder=50)
        plt.plot(*env.shift_to_global(path).T, linewidth=8, label=f"path {i + 1}", color='orange', zorder=20)

    plt.show()
