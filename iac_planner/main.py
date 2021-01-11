#!/usr/bin/env python3
"""
Visualizes the CTE of some simple paths using rviz2
"""
import sys
from math import sqrt
from typing import Optional, Iterable, Callable
import logging

import numpy as np
# from std_msgs.msg import ColorRGBA

# from iac_planner.generate_markers import visualize
from rticonnextdds_connector import Input

from iac_planner.generate_paths import generate_paths
from iac_planner.helpers import Env, state_t
from iac_planner.score_paths import score_paths

import rticonnextdds_connector as rti


def main(args: Optional[Iterable[str]] = None):
    env: Env = Env()
    env.global_path_handler.load_from_csv('./resources/velocityProfileMu85.csv')
    np_path = env.global_path_handler.global_path.to_numpy()[:, :2]
    env.path = np.vstack([np_path] * 6)
    logging.basicConfig(level=logging.NOTSET)
    info: Callable[[str], None] = logging.getLogger(__name__).info
    env.info = info
    info("Starting up...")

    # visualize(env.m_pub, env.nh.get_clock(), 52, env.obstacles, color=ColorRGBA(g=1.0, a=1.0))

    # TODO: Load the global path

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
                env.state[2] = sqrt(state_in["cdgSpeed_x"] ** 2 + state_in["cdgSpeed_y"] ** 2)

                # TODO: Load track Boundaries as Obstacles
                env.obstacles

                # TODO: Load dynamic vehicles
                env.other_vehicle_states

                # TODO: Load track boundaries
                # env.track_boundry_poly

                # Start stuff
                if len(env.path) <= 10:
                    break

                trajectory = run(env)

                # TODO: Run controller

                # TODO: Publish values from controller
                """
                <struct name="CabToModelCorrective">
                    <member name="AcceleratorAdditive" type="float64"/>
                    <member name="AcceleratorMultiplicative" type="float64"/>
                    <member name="BrakeAdditive" type="float64"/>
                    <member name="BrakeMultiplicative" type="float64"/>
                    <member name="ClutchAdditive" type="float64"/>
                    <member name="ClutchMultiplicative" type="float64"/>
                    <member name="GearboxAutoMode" type="int16"/>
                    <member name="GearboxTakeOver" type="octet"/>
                    <member name="IsRatioLimit" type="octet"/>
                    <member name="MaxRatio" type="int16"/>
                    <member name="MinRatio" type="int16"/>
                    <member name="ParkingBrakeAdditive" type="float64"/>
                    <member name="ParkingBrakeMultiplicative" type="float64"/>
                    <member name="ShiftDown" type="octet"/>
                    <member name="ShiftUp" type="octet"/>
                    <member name="TimeOfUpdate" type="float64"/>
                    <member name="WantedGear" type="int16"/>
                </struct>
                """
                """
                <struct name="CabToSteeringCorrective">
                    <member name="AdditiveSteeringWheelAccel" type="float64"/>
                    <member name="AdditiveSteeringWheelAngle" type="float64"/>
                    <member name="AdditiveSteeringWheelSpeed" type="float64"/>
                    <member name="AdditiveSteeringWheelTorque" type="float64"/>
                    <member name="MultiplicativeSteeringWheelAccel" type="float64"/>
                    <member name="MultiplicativeSteeringWheelAngle" type="float64"/>
                    <member name="MultiplicativeSteeringWheelSpeed" type="float64"/>
                    <member name="MultiplicativeSteeringWheelTorque" type="float64"/>
                    <member name="TimeOfUpdate" type="float64"/>
                </struct>
                """

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
