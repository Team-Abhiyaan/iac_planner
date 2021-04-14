from dataclasses import dataclass
from typing import List, Generator

import numpy as np
import matplotlib.pyplot as plt

from iac_planner.collision_check import CollisionChecker
from iac_planner.generate_velocity_profile import generate_velocity_profile
from iac_planner.helpers import state_t, CollisionParams, path_t, VelParams
from iac_planner.path_sampling._core import polyeval
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


def get_paths(state: state_t, n_paths: int = 12, n_pts: int = 32, alpha_max: float = 0.01) \
        -> Generator[path_t, None, None]:
    x0, y0, theta = state[:3]
    x = np.linspace(x0, x0 + 50, n_pts)
    for alpha in np.linspace(-alpha_max, alpha_max, n_paths):
        y = y0 + (x - x0) * np.tan(theta) + (x - x0) ** 2 * alpha
        yield np.vstack([x, y]).T


def main():
    env = Env()

    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim((env.state[0] - 10, env.state[0] + 75))
    plt.ylim((env.state[1] - 30, env.state[1] + 75))

    cc = CollisionChecker(env, path_length=20)

    # Ego
    # Overall Length 192 inches/4876 mm
    # Overall Width 76 inches/1930 mm

    def draw_rect(center, theta=0, L=4.876, W=1.930, **kwargs):
        c, s = np.cos(theta), np.sin(theta)
        plt.gca().add_artist(
            plt.Rectangle(center + np.array([-c * L / 2 + s * W / 2, -s * L / 2 - c * W / 2]), width=L, height=W,
                          angle=theta * 180 / np.pi, **kwargs))

    # plt.arrow(env.state[0], env.state[1],
    #           10 * np.cos(env.state[2]), 10 * np.sin(env.state[2]), head_width=2, label='vehicle', zorder=100)
    draw_rect(env.state[:2], env.state[2], color="blue")
    plt.gca().add_artist(
        plt.Circle(env.state[:2], env.collision_params.circle_radii, color='red')
    )

    # Lanes
    if len(lane_boundry_pts := env.lane_to_points()) != 0:
        plt.scatter(*zip(*lane_boundry_pts), label='Lane Boundaries', s=5)

    for path in get_paths(env.state, 10):
        vel_profile = generate_velocity_profile(env, path)
        if np.any(np.isnan(vel_profile)):
            print("ERROR: NAN in velocity_profile")
            continue
        is_valid = cc.check_collisions(path, vel_profile)
        plt.plot(*path.T, linewidth=(2 if is_valid else 0.5), c=('green' if is_valid else 'red'), zorder=0)

    for i, (state, path) in enumerate(zip(env.other_vehicle_states,
                                          np.swapaxes(cc._other_vehicle_paths[:, :2, :], 2, 1))):
        plt.arrow(*(env.shift_to_global(state[:2])),
                  10 * np.cos(state[2] + env.state[2]), 10 * np.sin(state[2] + env.state[2]),
                  head_width=2,
                  label=f"other {i + 1}", color='black', zorder=50)
        draw_rect(env.shift_to_global(state[:2]), state[2] + env.state[2], color="black")

        plt.plot(*env.shift_to_global(path).T, linewidth=8, label=f"path {i + 1}", color='orange', zorder=20)

    plt.show()


def for_timing(env: Env):
    cc = CollisionChecker(env, path_length=20)
    for path in get_paths(env.state, n_paths=32, n_pts=100):
        vel_profile = generate_velocity_profile(env, path)
        if np.any(np.isnan(vel_profile)):
            continue
        cc.check_collisions(path, vel_profile)
        # print(cc.check_collisions(path))


TIME_IT = False
if __name__ == '__main__':
    if TIME_IT:
        from time import time

        env = Env()

        start = time()
        for i in range(10):
            for_timing(env)
        end = time()

        print(f"Elapsed time {(end - start) / 10}")

    else:
        main()
