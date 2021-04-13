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
    def __init__(self, env: Env, path_length: int, time_step: float):
        # # For Timing
        # global times
        # times.clear()
        # self.print_times = True

        self._env = env

        self._params: CollisionParams = env.collision_params

        self._time_step = time_step
        self._path_length = path_length

        self.other_vehicle_states = np.array(env.other_vehicle_states)
        self._other_vehicle_paths = np.zeros((len(self.other_vehicle_states), 3, path_length), dtype=float)

        # TODO: Why using only 1st other_vehicle
        if len(self.other_vehicle_states) > 0:
            self.other_vech_current_vel = self.other_vehicle_states[0][3]
            # self.other_vech_prev_vel = 0

        # for i in range(len(self._other_vehicle_paths)):
        #     vehicle_state: state_t = other_vehicle_states[i]
        #     vehicle_path = np.zeros((3, path_length), dtype=float)

        #     time = np.arange(1, path_length + 1)  # , 1)
        #     vehicle_path[0] = time_step * vehicle_state[3] * math.cos(vehicle_state[2]) * time + vehicle_state[0]
        #     vehicle_path[1] = time_step * vehicle_state[3] * math.sin(vehicle_state[2]) * time + vehicle_state[1]
        #     vehicle_path[2] = vehicle_state[2]
        #     visualize(env.m_pub, env.nh.get_clock(), 75 + i, vehicle_path.T[:, :2],
        #               color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))

        #     self._other_vehicle_paths[i] = vehicle_path

        # Takes in a set of obstacle borders and path waypoints and returns
        # a boolean collision check array that tells if a path has an obstacle
        # or not

        # Find line perpendicular to the ego vehicles current heading
        # Find line with same slope but 5 meters ahead

        # TODO: Only used by visualization
        self.obstacles = []
        if env.left_poly is not None and env.right_poly is not None:

            cl = [env.left_poly.c3, env.left_poly.c2, env.left_poly.c1, env.left_poly.c0][::-1]
            cr = [env.right_poly.c3, env.right_poly.c2, env.right_poly.c1, env.right_poly.c0][::-1]

            # Generate points on the lane boundaries
            # Lane boundary is in local frame
            x0, y0 = 0, 0  # env.state[:2]
            x_l = np.linspace(x0 - 20, x0 + 150, 40)
            x_r = np.linspace(x0 - 20, x0 + 150, 40)

            y_l = np.array([polyeval(c, cl) for c in x_l])
            y_r = np.array([polyeval(c, cr) for c in x_r])
            yaw = env.state[2]
            x_ll, y_ll = x_l * np.cos(yaw) - y_l * np.sin(yaw), x_l * np.sin(yaw) + y_l * np.cos(yaw)
            x_rr, y_rr = x_r * np.cos(yaw) - y_r * np.sin(yaw), x_r * np.sin(yaw) + y_r * np.cos(yaw)
            x_ll += env.state[0]
            x_rr += env.state[0]
            y_ll += env.state[1]
            y_rr += env.state[1]
            self.obstacles = list(zip(x_ll, y_ll)) + list(zip(x_rr, y_rr))
            # print(f"OBSTACLE CHECKING: {len(self.obstacles)=}")
            self._obstacles = self.obstacles

        else:
            self._obstacles = []

    # # For Timing
    # def __del__(self):
    #     if self.print_times:
    #         global times
    #         for k, v in times.items():
    #             print(f"{k}: {v:.3f}")

    @staticmethod
    def generate_time_step(path, velocity_profile):
        time_step = np.zeros((len(velocity_profile),), dtype=float)

        for i in range(len(velocity_profile) - 1):
            s = np.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
            time_step[i + 1] = (2 * s) / (velocity_profile[i + 1] + velocity_profile[i])

        return time_step

    @staticmethod
    def generate_other_vehicle_paths(time_step, other_vehicle_states):
        other_vehicle_paths = np.zeros((len(other_vehicle_states), 3, len(time_step)), dtype=float)

        for i in range(len(other_vehicle_paths)):
            vehicle_state = other_vehicle_states[i]
            vehicle_path = np.zeros((3, len(time_step)), dtype=float)

            vehicle_path[0][0] = vehicle_state[0]
            vehicle_path[1][0] = vehicle_state[1]
            vehicle_path[2][0] = vehicle_state[2]

            for j in range(1, len(time_step)):
                vehicle_path[0][j] = vehicle_path[0][j - 1] + time_step[j] * vehicle_state[3] * math.cos(
                    vehicle_state[2])
                vehicle_path[1][j] = vehicle_path[1][j - 1] + time_step[j] * vehicle_state[3] * math.sin(
                    vehicle_state[2])
                vehicle_path[2][j] = vehicle_state[2]

            other_vehicle_paths[i] = vehicle_path

        return other_vehicle_paths

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

    # @print_time
    def _static_collision_check(self, path: path_t):
        """Returns a bool array on whether each path is collision free.
        args:
            paths: A list of paths in the global frame.
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]
        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
        if self._env.left_poly is None or self._env.right_poly is None:
            assert (False, "No poly lol")

        if len(path) == 0:
            assert (False, "Empty Path")

        if not self._lanes_collision_check(path):
            return False
        return True

    # @print_time
    def _dynamic_collision_check(self, path: path_t):
        """ Returns a bool array on whether each path is collision free.
        args:
                paths: A list of paths in the global frame.
                    A path is a list of points of the following format:
                        [x_points, y_points, t_points]:
                            x_points: List of x values (m)
                            y_points: List of y values (m)
                            t_points: List of yaw values (rad)
                        Example of accessing the ith path, jth point's t value:
                            paths[i][2][j]
                ego_state: ego state vector for the vehicle. (global frame)
                    format: [ego_x, ego_y, ego_yaw, ego_speed]
                        ego_x and ego_y     : position (m)
                        ego_yaw             : top-down orientation [-pi to pi] (ground frame)
                        ego_speed : speed (m/s)
                other_vehicle_states: other vehicles' state vectors
                    Each state vector is of the format (global frame):
                        [pos_x, pos_y, yaw, speed]
                            pos_x and pos_y	: position (m)
                            yaw 			: top-down orientation [-pi to pi] (ground frame)
                            speed 			: speed (m/s)
                        Example of accessing the ith car's speed would be:
                            other_vehicle_states[i][3]
                look_ahead_time: The look ahead time to which the paths have been generated (s)
            returns:
                collision_check_array: A list of boolean values which classifies
                    whether the path is collision-free (true), or not (false). The
                    ith index in the collision_check_array list corresponds to the
                    ith path in the paths list.
        """
        self.init_other_paths(path)

        if len(self._other_vehicle_paths) == 0:
            return True

        if len(path) == 0:
            assert (False, "Empty Path")

        # time step between each point in path
        time_step = self._time_step
        print(time_step)

        for j in range(len(path[0])):

            # generating ego vehicle's circle location from the given offset
            ego_circle_locations = np.zeros((1, 2))

            circle_offset = self._params.circle_offset
            ego_circle_locations[:, 0] = path[0][j] + circle_offset * math.cos(path[2][j])
            ego_circle_locations[:, 1] = path[1][j] + circle_offset * math.sin(path[2][j])

            for vel, path in zip(self.other_vehicle_states[:, 3], self._other_vehicle_paths):
                # generating other vehicles' circle locations based on circle offset
                other_circle_locations = np.zeros((1, 2))

                other_circle_locations[:, 0] = path[0][j] + circle_offset * math.cos(path[2][j])
                other_circle_locations[:, 1] = path[1][j] + circle_offset * math.sin(path[2][j])

                # calculating if any collisions occur
                # growth_factor = self._params.growth_factor_b + self._params.growth_factor_a * (
                #         1 - self.other_vech_prev_vel / vel
                # )
                growth_factor = 0

                collision_dists = scipy.spatial.distance.cdist(other_circle_locations, ego_circle_locations)
                collision_dists = np.subtract(collision_dists,
                                              self._params.circle_radii * (2 + growth_factor * np.sum(time_step[:j])))

                if np.any(collision_dists < 0):
                    return False

        # self.other_vech_prev_vel = self.other_vehicle_states[:, 3]
        return True

    # @print_time
    def init_other_paths(self, path):
        velocity_profile = generate_velocity_profile(self._env, path)
        self._time_step = self.generate_time_step(path, velocity_profile)
        self._other_vehicle_paths = self.generate_other_vehicle_paths(self._time_step, self.other_vehicle_states)

    def check_collisions(self, path: path_t):
        return self._static_collision_check(path) and self._dynamic_collision_check(path)
