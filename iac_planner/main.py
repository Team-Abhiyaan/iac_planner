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

from iac_planner.controller import Controller
from iac_planner.generate_paths import generate_paths
from iac_planner.generate_velocity_profile import generate_velocity_profile
from iac_planner.helpers import Env, state_t
from iac_planner.path_sampling.types import RoadLinePolynom
from iac_planner.score_paths import score_paths

# TODO: update this
GLOBAL_PATH_CSV_FILE = "./resources/velocityProfileMu85.csv"

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(args: Optional[Iterable[str]] = None):
    env: Env = Env()
    info: Callable[[str], None] = _logger.info
    env.info = info

    info("Starting up...")

    env.global_path_handler.load_from_csv(GLOBAL_PATH_CSV_FILE)
    np_path = env.global_path_handler.global_path.to_numpy()[:, :2]  # Take only (x, y) from path.
    env.path = np.vstack([np_path] * 6 + [np_path[:20, :]])  # Repeat for 6 laps

    info(f"Loaded path from {GLOBAL_PATH_CSV_FILE} of lenght {len(env.path)}")

    try:
        with rti.open_connector(config_name="SCADE_DS_Controller::Controller",
                                url="resources/RtiSCADE_DS_Controller_ego1.xml") as connector:
            info('Opened RTI Connector')

            # Readers
            class Inputs:
                sim_wait: Input = connector.get_input("simWaitSub::simWaitReader")
                vehicle_state: Input = connector.get_input("vehicleStateOutSub::vehicleStateOutReader")
                track_polys: Input = connector.get_input("camRoadLinesF1Sub::camRoadLinesF1Reader")
                other_vehicle_states: Input = connector.get_input("radarFSub::radarFReader")

                def list(self) -> Iterable[Input]:
                    return [self.sim_wait, self.vehicle_state, self.track_polys, self.other_vehicle_states]

            inputs = Inputs()

            # Writers
            vehicle_correct = connector.getOutput("toVehicleModelCorrectivePub::toVehicleModelCorrectiveWriter")
            vehicle_steer = connector.getOutput("toVehicleSteeringPub::toVehicleSteeringWriter")
            sim_done = connector.getOutput("toSimDonePub::toSimDoneWriter")

            sim_done.write()

            controller = Controller()

            while True:
                load_rti(env, inputs)
                info('Got RTI inputs')

                # Remove passed points
                # Note fails if robot and path have very different orientations, check for that
                update_global_path(env)
                use_global = False  # for plotting only
                trajectory = None
                trajectory = run(env)
                if trajectory is None:
                    info("Warning: Could not get trajectory, falling back to global path.")
                    trajectory = env.path[:31, :], generate_velocity_profile(env, env.path[:32, :])
                    use_global = True

                throttle, steer = controller.run_controller_timestep(env, trajectory)

                if env.plot_paths:
                    import matplotlib.pyplot as plt
                    plt.clf()
                    plt.gca().set_aspect('equal', adjustable='box')
                    from iac_planner.collision_check import CollisionChecker
                    obs = CollisionChecker(env, 20, time_step=0.5).obstacles
                    plt.scatter(*zip(*obs), label='obstacles', s=5)
                    plt.scatter(*env.path[:40].T, label='global path', s=5)
                    if trajectory is not None:
                        plt.scatter(*trajectory[0].T, label=('local path' if not use_global else 'fake local'), s=5)
                    plt.arrow(env.state[0], env.state[1], 20 * np.cos(env.state[2]), 20 * np.sin(env.state[2]),
                              head_width=5, label='vehicle')
                    for state in env.other_vehicle_states:
                        plt.arrow(state[0], state[1], 20 * np.cos(state[2]), 20 * np.sin(state[2]), head_width=5,
                                  label='other')

                    plt.xlim((env.state[0] - 75, env.state[0] + 75))
                    plt.ylim((env.state[1] - 150, env.state[1] + 75))
                    plt.legend()
                    plt.pause(0.01)

                vehicle_steer.instance.setNumber("AdditiveSteeringWheelAngle", steer)
                vehicle_correct.instance.setNumber("AcceleratorAdditive", throttle)
                vehicle_correct.write()
                vehicle_steer.write()

                sim_done.write()
                info("=" * 20)

    except KeyboardInterrupt:
        info("Keyboard interrupt")


def load_rti(env, inputs):
    # for reader in inputs.list():
    #     reader.wait()
    #     reader.take()
    inputs.sim_wait.wait()
    inputs.sim_wait.take()
    env.info('Got sim wait')
    inputs.vehicle_state.wait()
    inputs.vehicle_state.take()
    env.info('Got state')

    # read values to env
    state_in = None
    for state_in in inputs.vehicle_state.samples.valid_data_iter:
        pass
    env.state[0] = state_in["cdgPos_x"]
    env.state[1] = state_in["cdgPos_y"]
    env.state[2] = state_in["cdgPos_heading"]
    env.state[3] = np.sqrt(state_in["cdgSpeed_x"] ** 2 + state_in["cdgSpeed_y"] ** 2)
    env.gear = state_in["GearEngaged"]

    # TODO: Load track Boundaries as Obstacles
    track_polys = inputs.track_polys
    track_polys.wait()
    track_polys.take()
    track_poly = None
    for track_poly in track_polys.samples.valid_data_iter:
        pass
    roadlinepolyarray = track_poly['roadLinesPolynomsArray']
    left_array = roadlinepolyarray[0]
    right_array = roadlinepolyarray[1]
    env.left_poly = RoadLinePolynom(left_array['c0'], left_array['c1'], left_array['c2'],
                                    left_array['c3'])
    env.right_poly = RoadLinePolynom(right_array['c0'], right_array['c1'], right_array['c2'],
                                     right_array['c3'])

    # TODO: Load dynamic vehicles
    other_vehicle_states = inputs.other_vehicle_states
    other_vehicle_states.wait()
    other_vehicle_states.take()
    env.other_vehicle_states = []
    other_vehicles = None
    for other_vehicles in other_vehicle_states.samples.valid_data_iter:
        pass
    targetsArray = other_vehicles['targetsArray']
    print(len(targetsArray))
    for other_vehicle in targetsArray:
        x = other_vehicle['posXInChosenRef']
        y = other_vehicle['posYInChosenRef']
        yaw = other_vehicle['posHeadingInChosenRef']
        v = other_vehicle['absoluteSpeedX']
        env.other_vehicle_states.append(np.array([x, y, yaw, v]))


def run(env: Env):
    info = env.info

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
        info(f"Lowest {cost=:.2f}")
    else:
        info("ERROR: No trajectory found.")

    return best_trajectory


def update_global_path(env: Env):
    def line_behind_vehicle(x: float, y: float) -> float:
        p: state_t = env.state
        # yaw = p[2]
        p0, p1 = env.path[0:2]
        yaw = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        return (x - p[0]) * np.cos(yaw) + (y - p[1]) * np.sin(yaw)

    def is_behind(x: float, y: float) -> bool:
        return line_behind_vehicle(x, y) < 0

    while is_behind(*env.path[0]):
        print('Passed a point in global path.')
        env.path = env.path[1:, :]


if __name__ == '__main__':
    main(sys.argv)
