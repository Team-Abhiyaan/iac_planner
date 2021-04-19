import numpy as np
import scipy.spatial

import math

from iac_planner.helpers import Env, path_t, CollisionParams
from iac_planner.generate_velocity_profile import generate_velocity_profile

# TODO: Fix docstrings
from iac_planner.path_sampling._core import polyeval


# # For Timing
# import time
# times = {}
#
#
# def print_time(func):
#     def wrapper(*arg, **kw):
#         global times
#         t1 = time.time()
#         res = func(*arg, **kw)
#         t2 = time.time()
#         times[func.__name__] = times.get(func.__name__, 0) + t2 - t1
#         return res
#     return wrapper
#


class CollisionChecker:
    def __init__(self, env: Env, path_length: int):
        # # For Timing
        # global times
        # times.clear()
        # self.print_times = True

        self._env = env
        self._params: CollisionParams = env.collision_params
        self._path_length = path_length

        self.other_vehicle_states = np.zeros((len(env.other_vehicle_states), 3))
        if len(env.other_vehicle_states) != 0:
            self.other_vehicle_states[:, :2] = env.shift_to_global(np.stack(env.other_vehicle_states, axis=0)[:, :2])
            self.other_vehicle_states[:, 2] += env.state[2]

        self._other_vehicle_paths = np.zeros((len(self.other_vehicle_states), 3, path_length), dtype=float)
        self._time_steps = np.zeros((path_length,))

        # # TODO: Why using only 1st other_vehicle
        # if len(self.other_vehicle_states) > 0:
        #     self.other_vech_current_vel = self.other_vehicle_states[0][3]
        #     # self.other_vech_prev_vel = 0

    # # For Timing
    # def __del__(self):
    #     if self.print_times:
    #         global times
    #         for k, v in times.items():
    #             print(f"{k}: {v:.3f}")

    def _lanes_collision_check(self, path: path_t):
        if self._env.left_poly is None or self._env.right_poly is None:
            print("NO POLY!!!")
            return True

        if len(path) == 0:
            assert (False, "Empty Path")

            # Transform path to local frame
        env = self._env
        yaw = env.state[2]
        pos_cur = np.array([env.state[0], env.state[1]]).reshape((1, 2))
        rot_matrix = np.array([
            [np.cos(-yaw), np.sin(-yaw)],
            [-np.sin(-yaw), np.cos(-yaw)]]
        )
        p_trans = (path - pos_cur) @ rot_matrix

        # Check for outside lane
        # TODO: Add a buffer
        for pt in p_trans:
            if (polyeval(pt[0], self._env.left_poly) - pt[1]) * (polyeval(pt[0], self._env.right_poly) - pt[1]) > 0:
                return False

        return True

    @staticmethod
    def generate_time_steps(path, velocity_profile):
        time_steps = np.zeros_like(velocity_profile)

        for i in range(len(velocity_profile) - 1):
            s = np.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
            time_steps[i + 1] = (2 * s) / (velocity_profile[i + 1] + velocity_profile[i])

        return time_steps

    @staticmethod
    def generate_other_vehicle_paths(time_step, other_vehicle_states):
        other_vehicle_paths = np.zeros((len(other_vehicle_states), 3, len(time_step)), dtype=float)

        for i in range(len(other_vehicle_paths)):
            state = other_vehicle_states[i]
            path = np.zeros((3, len(time_step)), dtype=float)

            path[0][0] = state[0]
            path[1][0] = state[1]
            path[2][0] = state[2]

            for j in range(1, len(time_step)):
                path[0][j] = path[0][j - 1] + time_step[j] * state[3] * math.cos(state[2])
                path[1][j] = path[1][j - 1] + time_step[j] * state[3] * math.sin(state[2])
                path[2][j] = state[2]

            other_vehicle_paths[i] = path

        return other_vehicle_paths

    # @print_time
    def init_other_paths(self, path, vel_profile=None):
        if vel_profile is None:
            vel_profile = generate_velocity_profile(self._env, path)
        # if np.any(np.isnan(vel_profile)):
        #     print("Invalid Velocity profile")
        #     return False

        self._time_steps = self.generate_time_steps(path, vel_profile)
        self._other_vehicle_paths = self.generate_other_vehicle_paths(self._time_steps, self.other_vehicle_states)

    # @print_time
    def _dynamic_collision_check(self, path: path_t, vel_profile=None) -> bool:
        """ Returns a bool array on whether each path is collision free.
        Args:
                path: a path_t in global frame
                vel_profile: a 1D np.array of length (len(path) - 1)
        Returns:
                bool: whether path is safe
        """

        if len(self._other_vehicle_paths) == 0:
            return True

        if len(path) == 0:
            assert (False, "Empty Path")

        self.init_other_paths(path, vel_profile)

        angle = self._env.state[2]
        for j in range(len(path) - 1):
            # generating ego vehicle's circle location from the given offset
            circle_offsets = self._params.circle_offsets
            ego_circle_locations = np.zeros((len(circle_offsets), 2))

            if j != 0:
                angle = np.arctan2(*(path[j, :2] - path[j - 1, :2]))

            ego_circle_locations[:, 0] = path[j, 0] + circle_offsets * math.cos(angle)
            ego_circle_locations[:, 1] = path[j, 1] + circle_offsets * math.sin(angle)

            for vel, opath in zip(self.other_vehicle_states[:, 3], self._other_vehicle_paths):
                if np.sum(np.abs(path[j, :2] - opath[:2, j])) > 10:
                    # len(circle_offsets) * 2 * self._params.circle_radii:
                    continue

                # generating other vehicles' circle locations based on circle offset
                other_circle_locations = np.zeros_like(ego_circle_locations)
                other_circle_locations[:, 0] = opath[0][j] + circle_offsets * math.cos(opath[2][j])
                other_circle_locations[:, 1] = opath[1][j] + circle_offsets * math.sin(opath[2][j])
                # print(other_circle_locations, ego_circle_locations)

                # calculating if any collisions occur
                # growth_factor = self._params.growth_factor_b + self._params.growth_factor_a * (
                #         1 - self.other_vech_prev_vel / vel
                # )
                growth_factor = 0

                collision_dists = scipy.spatial.distance.cdist(other_circle_locations, ego_circle_locations).flatten()
                min_dist = self._params.circle_radii * (2 + growth_factor * np.sum(self._time_steps[:j]))
                collision_dists -= min_dist

                if np.any(collision_dists < 0):
                    return False

        # self.other_vech_prev_vel = self.other_vehicle_states[:, 3]
        return True

    def check_collisions(self, path: path_t, vel_profile=None) -> bool:
        """

        Args:
                path: a path_t in global frame
                vel_profile: a 1D np.array of length (len(path) - 1)
        Returns:
            is path safe
        """
        return self._lanes_collision_check(path) and self._dynamic_collision_check(path, vel_profile)
