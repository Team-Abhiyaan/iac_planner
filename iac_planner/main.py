#!/usr/bin/env python3
"""
Visualizes the CTE of some simple paths using rviz2
"""
import sys
from typing import Optional, Iterable, Callable
import logging

import numpy as np
from rticonnextdds_connector import Input
import rticonnextdds_connector as rti

from iac_planner.generate_paths import generate_paths
from iac_planner.generate_velocity_profile import generate_velocity_profile
from iac_planner.helpers import Env, state_t
from iac_planner.path_sampling.types import RoadLinePolynom
from iac_planner.score_paths import score_paths

GLOBAL_PATH_CSV_FILE = "./resources/velocityProfileMu85.csv"

logging.basicConfig(level=logging.NOTSET)
_logger = logging.getLogger(__name__)

def main(args: Optional[Iterable[str]] = None):
    env: Env = Env()
    env.global_path_handler.load_from_csv(GLOBAL_PATH_CSV_FILE)

    np_path = env.global_path_handler.global_path.to_numpy()[:, :2]
    env.path = np.vstack([np_path] * 6 + [np_path[:20, :]])
    info: Callable[[str], None] = _logger.info
    env.info = info
    info("Starting up...")

    try:
        with rti.open_connector(config_name="SCADE_DS_Controller::Controller",
                                url="resources/RtiSCADE_DS_Controller_ego1.xml") as connector:
            # Readers
            class Inputs:
                sim_wait: Input = connector.get_input("simWaitSub::simWaitReader"),
                vehicle_state: Input = connector.get_input("vehicleStateOutSub::vehicleStateOutReader")

                def list(self) -> Iterable[Input]:
                    return [self.sim_wait, self.vehicle_state]

            inputs = Inputs()

            # Writers
            vehicle_correct = connector.getOutput("toVehicleModelCorrectivePub::toVehicleModelCorrectiveWriter")
            vehicle_steer = connector.getOutput("toVehicleSteeringPub::toVehicleSteeringWriter")
            sim_done = connector.getOutput("toSimDonePub::toSimDoneWriter")
            sim_done.write()

            while True:
                for reader in inputs.list():
                    reader.wait()
                    reader.take()

                # read values to env
                state_in = inputs.vehicle_state.samples[-1]
                env.state[0] = state_in["cdgPos_x"]
                env.state[1] = state_in["cdgPos_y"]
                env.state[1] = state_in["cdgSpeed_heading"]
                env.state[2] = np.sqrt(state_in["cdgSpeed_x"] ** 2 + state_in["cdgSpeed_y"] ** 2)

                trajectory = None
                try:
                    # TODO: Load track Boundaries as Obstacles
                    track_polys = connector.get_input("camRoadLinesF1Sub::camRoadLinesF1Reader")
                    track_polys.wait()
                    track_polys.take()

                    for track_poly in track_polys.samples.valid_data_iter:
                        roadlinepolyarray = track_poly['roadLinesPolynomsArray']
                        left_array = roadlinepolyarray[0]
                        right_array = roadlinepolyarray[1]
                        env.left_poly = RoadLinePolynom(left_array['c0'], left_array['c1'], left_array['c2'],
                                                        left_array['c3'])
                        env.right_poly = RoadLinePolynom(right_array['c0'], right_array['c1'], right_array['c2'],
                                                         right_array['c3'])

                    # TODO: Load dynamic vehicles
                    other_vehicle_states = connector.get_input("radarFSub::radarFReader")
                    other_vehicle_states.wait()
                    other_vehicle_states.take()

                    env.other_vehicle_states = []
                    for other_vehicle_state in other_vehicle_states.samples.valid_data_iter:
                        targetsArray = other_vehicle_state['targetsArray']
                        x = targetsArray['posXInChosenRef']
                        y = targetsArray['posYInChosenRef']
                        yaw = targetsArray['posHeadingInChosenRef']
                        v = targetsArray['absoluteSpeedX']

                        env.other_vehicle_states[0] = np.array([x, y, yaw, v])

                    trajectory = run(env)
                    if trajectory is None:
                        trajectory = env.path[:18, :], generate_velocity_profile(env, env.path[:19, :])
                except Exception:
                    trajectory = env.path[:18, :], generate_velocity_profile(env, env.path[:19, :])

                # TODO: Run controller
                def controller(trajectory):
                    return "outputs"

                controller(trajectory)

                # Publish to the RTI connext Writers
                # vehicle_correct, vehicle_steer

                sim_done.write()


    except KeyboardInterrupt:
        info("Keyboard interrupt")


def run(env: Env):
    info = env.info
    # Remove passed points
    # Note fails if robot and path have very different orientations, check for that

    update_global_path(env)

    # # Publish Global Path and Current Position
    # visualize(env.m_pub, env.nh.get_clock(), 50, [env.state[:2]], scale=0.5,
    #           color=ColorRGBA(r=1.0, b=1.0, a=1.0))
    #
    # visualize(env.m_pub, env.nh.get_clock(), 51, env.path)

    # for i, state in enumerate(env.other_vehicle_states):
    #     visualize(env.m_pub, env.nh.get_clock(), 52 + 1 + i, [state[:2]], scale=0.5,
    #               color=ColorRGBA(r=1.0, g=1.0, a=1.0))

    paths = generate_paths(env)

    best_trajectory, cost = score_paths(env, paths, max_path_len=env.path_generation_params.n_pts_long)

    if best_trajectory is not None:
        info(f"Lowest {cost=:.2f}: {best_trajectory[1][:4]}")
    else:
        info("No trajectory found.")

    return best_trajectory


def update_global_path(env: Env):
    def line_behind_vehicle(x: float, y: float) -> float:
        p: state_t = env.state
        return (x - p[0]) * np.cos(p[2]) + (y - p[1]) * np.sin(p[2])

    def is_behind(x: float, y: float) -> bool:
        return line_behind_vehicle(x, y) < 0

    while is_behind(*env.path[0]):
        env.path = env.path[1:, :]


if __name__ == '__main__':
    main(sys.argv)
