from typing import Generator, List

import numpy as np

from iac_planner.helpers import path_t, PathGenerationParams
from iac_planner.path_sampling.spline import SplineGenerator
from iac_planner.path_sampling.types import Pose


def generate_paths(env) -> Generator[path_t, None, None]:
    """
    Generates straight lines around the direction the vehicle faces
    TODO: Replace with an actual path generator

    :param env: Environment
    :param n: Number of paths
    :param n_pts: Number of points in each path
    :return:
    """
    s = env.state
    p = Pose(s[0], s[1], yaw=s[2])
    p_obs: Pose = None
    if len(env.other_vehicle_states) != 0:
        state = env.other_vehicle_states[0]
        p_obs = Pose(state[0], state[1], yaw=state[2])
        print(p_obs)
        # p_obs.yaw = state[2]

    gen = SplineGenerator(env.global_path_handler, p, p_obs, env.left_poly, env.right_poly)
    params: PathGenerationParams = env.path_generation_params
    for path in gen.generate_long(params.n_long, params.n_pts_long):
        yield np.stack(path, axis=1)

    if p_obs is not None and env.left_poly is not None and env.right_poly is not None:
        for path in gen.generate_lat(params.n_long, params.n_pts_long):
            yield np.stack(path, axis=1)
