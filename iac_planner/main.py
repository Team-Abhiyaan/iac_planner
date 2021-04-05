#!/usr/bin/env python3

import logging
import sys
import time
from typing import Optional, Iterable, Callable, List

import numpy as np
import rticonnextdds_connector as rti
from rticonnextdds_connector import Input

from iac_planner.controller import Controller
from iac_planner.generate_paths import generate_paths
from iac_planner.generate_velocity_profile import generate_velocity_profile
from iac_planner.helpers import Env, state_t
from iac_planner.path_sampling.types import RoadLinePolynom
from iac_planner.score_paths import score_paths

EGO = 1


def get_xml_url(ego: int) -> str:
    if ego == 2:
        return "./resources/RtiSCADE_DS_Controller_ego2.xml"
    elif ego == 1:
        return "./resources/RtiSCADE_DS_Controller_ego1.xml"
    else:
        raise Exception("Invalid EGO number")


# DS_CONTROLLER_EGO_XML = "./resources/RtiSCADE_DS_Controller_ego1.xml"
GLOBAL_PATH_CSV_FILE = "./resources/velocityProfileMu85.csv"

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(args: Optional[List[str]] = None):
    env: Env = Env()
    info: Callable[[str], None] = _logger.info
    env.info = info

    global EGO
    EGO = 1
    if args is not None and len(args) >= 2 and args[1].strip() == '2':
        EGO = 2
    print(f"Using ego {EGO}")
    if args is not None and len(args) >= 3 and args[2].strip() == '--no-plot':
        env.plot_paths = False

    info("Starting up...")

    env.global_path_handler.load_from_csv(GLOBAL_PATH_CSV_FILE)
    np_path = env.global_path_handler.global_path.to_numpy()[:, :2]  # Take only (x, y) from path.
    env.path = np.vstack([np_path] * 6 + [np_path[:20, :]])  # Repeat for 6 laps

    info(f"Loaded path from {GLOBAL_PATH_CSV_FILE} of length {len(env.path)}")

    try:
        with rti.open_connector(config_name="SCADE_DS_Controller::Controller",
                                url=get_xml_url(EGO)) as connector:
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
                # info('Got RTI inputs')
                t_start = time.time()
                # Remove passed points
                # Note fails if robot and path have very different orientations, check for that
                update_global_path(env)
                use_global = False  # for plotting only

                # ~ 3300 ms
                trajectory = run(env)
                if trajectory is None:
                    info("Warning: Could not get trajectory, falling back to global path.")
                    trajectory = env.path[:31, :], generate_velocity_profile(env, env.path[:32, :])
                    use_global = True

                # ~ 30 ms
                throttle, steer = controller.run_controller_timestep(env, trajectory)

                t_logic = time.time()
                print(f"time: {(t_logic - t_start):.3f}")

                if env.plot_paths:  # ~ 200ms
                    import matplotlib.pyplot as plt
                    plt.clf()
                    plt.title(f"EGO {EGO}")
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlim((env.state[0] - 75, env.state[0] + 75))
                    plt.ylim((env.state[1] - 75, env.state[1] + 75))

                    from iac_planner.collision_check import CollisionChecker
                    cc = CollisionChecker(env, 20, time_step=0.5)
                    cc.init_other_paths(trajectory[0])

                    if len(cc.obstacles) != 0:
                        plt.scatter(*zip(*cc.obstacles), label='obstacles', s=5)
                    plt.scatter(*env.path[:40].T, label='global path', s=5)
                    if trajectory is not None:
                        plt.scatter(*trajectory[0].T, label=('local path' if not use_global else 'fake local'), s=5)
                    plt.arrow(env.state[0], env.state[1], 20 * np.cos(env.state[2]), 20 * np.sin(env.state[2]),
                              head_width=5, label='vehicle')
                    for i, state in enumerate(env.other_vehicle_states):
                        plt.arrow(state[0], state[1],
                                  20 * np.cos(state[2]), 20 * np.sin(state[2]),
                                  head_width=5,
                                  label=f"other {i + 1}", color='red')

                    for i, path in enumerate(cc._other_vehicle_paths):
                        plt.scatter(*path[:2], s=5, label=f"path {i + 1}", color='red')

                    plt.legend()
                    plt.pause(0.005)

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
    # env.info('Got state')

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
    road_line_polys = track_poly['roadLinesPolynomsArray']
    if len(road_line_polys) >= 2:
        left_array = road_line_polys[0]

        right_array = road_line_polys[1]
        env.left_poly = RoadLinePolynom(left_array['c0'], left_array['c1'], left_array['c2'],
                                        left_array['c3'])
        env.right_poly = RoadLinePolynom(right_array['c0'], right_array['c1'], right_array['c2'],
                                         right_array['c3'])
    else:
        print("ERROR: didn't get data")

    other_vehicle_states = inputs.other_vehicle_states
    other_vehicle_states.wait()
    other_vehicle_states.take()
    env.other_vehicle_states = []
    other_vehicles = None
    for other_vehicles in other_vehicle_states.samples.valid_data_iter:
        pass
    targets_array = other_vehicles['targetsArray']
    print(f"{len(targets_array)} other vehicles")
    for other_vehicle in targets_array:
        x = other_vehicle['posXInChosenRef']
        y = other_vehicle['posYInChosenRef']
        yaw = other_vehicle['posHeadingInChosenRef']
        v = other_vehicle['absoluteSpeedX']
        ego_yaw = env.state[2]
        xx, yy = \
            x * np.cos(ego_yaw) - y * np.sin(ego_yaw), \
            x * np.sin(ego_yaw) + y * np.cos(ego_yaw)
        xx += env.state[0]
        yy += env.state[1]
        yaw += ego_yaw
        env.other_vehicle_states.append(np.array([xx, yy, yaw, v]))


def run(env: Env):
    info = env.info

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

    if line_behind_vehicle(*env.path[0]) < -4:
        update_global_path_by_dist(env)

    while is_behind(*env.path[0]):
        print('Passed a point in global path.')
        env.path = env.path[1:, :]


def update_global_path_by_dist(env: Env):
    def is_close(x: float, y: float) -> bool:
        xe, ye = env.state[:2]
        return (x - xe) ** 2 + (y - ye) ** 2 <= 6 ** 2

    i = 0
    while not is_close(*env.path[i]):
        # print('Passed a point in global path.')
        i += 1
        if i + 10 > len(env.path):
            print("ERROR: No point in global path is close enough")
            return
    print(f"Skipped {i} points in global path, remaining {len(env.path) - i}")
    env.path = env.path[i:, :]


if __name__ == '__main__':
    main(sys.argv)
