# from __future__ import print_function
import rticonnextdds_connector as rti
import csv
import numpy as np
import math

from iac_planner.helpers import Env


class Controller:
    def __init__(self):
        self.distance = 0.0
        self.y_difference = 0.0
        self.x_difference = 0.0
        self.required_angle = 0.0
        self.nozero = 0
        self.noone = 0
        self.notwo = 0
        self.nothree = 0
        self.nofour = 0
        self.nofive = 0
        self.lap = 0

        self.i = 3
        self.globalpathi = 3
        # self.localpathi = 0
        self.j = 0
        self.k = 0
        self.fanglea = 0.0
        self.fangleb = 0.0
        self.fanglec = 0.0
        self.fangled = 0.0
        self.diff_1 = 0.0
        self.diff_2 = 0.0
        self.diff_3 = 0.0
        self.diff_4 = 0.0
        self.k_1 = 0
        self.k_2 = 0
        self.k_3 = 0.0
        self.k_4 = 0.0
        self.fangle = 0.0
        self.diffangle = 0.0
        self.steer_put = 0.0
        self.acceleration = 0.0
        self.acceleration_previous = 0.0
        self.globalWaypointsLength = 5102
        self.x2 = [[0.0 for xxx in range(2)] for yyy in range(self.globalWaypointsLength)]
        self.y2 = [[0.0 for xxx in range(2)] for yyy in range(self.globalWaypointsLength)]
        self.theta = [0.0] * self.globalWaypointsLength
        self.L = 3.5
        self.w = 4
        self.h = self.globalWaypointsLength
        self.globalWaypoints = [[0.0 for xxx in range(self.w)] for yyy in range(self.h)]
        self.dataGetter = [[0.0, 0.0, 0.0, 0.0, 0.0]]
        self.loopHelper = 1
        self.throttle_output = 0.0
        self.steer_output = 0.0
        self.brake_output = 0.0
        self.v = 0.0
        self.v_desired = 0.0
        self.y = 0.0
        self.x = 0.0
        self.yaw = 0.0
        self.gear = 1
        self.throttle_output_previous = np.array([0.0, 0.0, 0.0, 0.0])  # np.zeros(4)
        self.v_previous_array = np.array([0.0, 0.0, 0.0, 0.0])
        self.v_array = np.array([0.0, 0.0, 0.0, 0.0])
        self.X2 = np.array([0.0, 0.0, 0.0, 0.0])
        self.A = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.B = np.array([0.0, 0.0, 0.0, 0.0])

        # values to be stored somewhere
        self.runCount = 0
        self.v_req_previous = 0.0
        self.steer_previous = 0.0
        self.acceleration_previous = 0.0
        self.v_previous = 42.66
        self.y_previous = -1835.977
        self.x_previous = -298.153
        self.throttleoutput_previous = 0.0
        self.steeroutput_previous = 0.0
        self.frequency = 20

        self.prev_diffangle = 0.0
        self.v_previous = 41.6666
        self.v_pre_previous = 41.6666
        self.throttle_previous = 1.0
        self.loopHelper = 1

        # Vehicle dynamics constants
        self.alphar = 0.0
        self.Calpha = 867
        self.a, self.b, self.c = 0.0, 0.0, 0.0
        self.back = 10

        self.d = 4
        self.downforceCoeff = 0.965
        # dragCoeff= 0.514
        self.dragCoeff = 0.51
        self.m = 630.0
        self.mu = 0.90
        self.rollingFriction = 100.0
        self.maxPower = 294000
        self.FintoV = self.maxPower

    def read_global_path_csv(self, filename="5LapFinal85.csv"):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            temp = 0
            for row in reader:
                self.globalWaypoints[temp][0], self.globalWaypoints[temp][1], self.globalWaypoints[temp][2], \
                self.globalWaypoints[temp][3] = row[0], row[1], (row[2]), row[3]
                self.globalWaypoints[temp][0], self.globalWaypoints[temp][1], self.globalWaypoints[temp][2], \
                self.globalWaypoints[temp][3] = float(self.globalWaypoints[temp][0]), float(
                    self.globalWaypoints[temp][1]), float(self.globalWaypoints[temp][2]), float(
                    self.globalWaypoints[temp][3])
                temp += 1
                print(temp)
        return self.globalWaypoints

    def getRadius(self, index, waypoints=0, filename="5LapFinal85.csv"):

        if waypoints == 0:
            waypoints = [[0.0 for xx in range(self.w)] for yy in range(self.h)]
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                t = 0
                for row in reader:
                    self.x2[t], self.y2[t] = row[0], row[1]
                    t += 1

            for counted in range(self.h - 1):  # in range(1, self.h - 1):
                if counted > 0:
                    self.x2[counted] = float(self.x2[counted])
                    self.y2[counted] = float(self.y2[counted])
                    waypoints[counted][0] = float(self.x2[counted])
                    waypoints[counted][1] = float(self.y2[counted])

        self.a, self.b, self.c, dd, e, f = waypoints[index][0], waypoints[index][1], waypoints[index - 1][0], \
                                           waypoints[index - 1][1], waypoints[index + 1][0], waypoints[index + 1][1]

        A = np.array([[2 * self.a - 2 * self.c, 2 * self.b - 2 * dd], [2 * self.a - 2 * e, 2 * self.b - 2 * f]])
        B = np.array([self.c * self.c + dd * dd - self.a * self.a - self.b * self.b,
                      e * e + f * f - self.a * self.a - self.b * self.b])

        if (self.a - self.c) * (self.b - f) == (self.a - e) * (dd - self.b):
            return 100000000000000000.0
        else:
            solution = np.linalg.solve(A, B)
            return np.sqrt(np.square(solution[0] - self.a) + np.square(solution[1] - self.b))

    # # For testing
    # def rti_connext(self):
    #     connector = rti.Connector("SCADE_DS_Controller::Controller", "RtiSCADE_DS_Controller.xml")
    #     self.vehicle_State = connector.get_input("vehicleStateOutSub::vehicleStateOutReader")
    #     self.simWait = connector.get_input("simWaitSub::simWaitReader")
    #     self.vehicleCorrect = connector.getOutput("toVehicleModelCorrectivePub::toVehicleModelCorrectiveWriter")
    #     self.vehicleSteer = connector.getOutput("toVehicleSteeringPub::toVehicleSteeringWriter")
    #     self.simDone = connector.getOutput("toSimDonePub::toSimDoneWriter")
    #     self.simDone.write()

    def run_controller_timestep(self, env, trajectory=None):
        # if trajectory is None:
        # print("INFO: GEtting default path")
        # i = self.globalpathi
        # waypoints = self.globalWaypoints
        # print((len(waypoints), 1))
        # else:
        waypoints = []
        for pt, vel in zip(*trajectory):
            print([pt[0], pt[1], vel])
            waypoints.append([pt[0], pt[1], vel])
        # waypoints = [(pt + [vel]) for pt, vel in zip(*trajectory)]
        i = 1
        # print((waypoints[0], 2))

        self.x, self.y, self.yaw, self.v = env.state
        self.gear = env.gear

        self.y_difference = self.y - waypoints[i + 1][1]
        self.x_difference = self.x - waypoints[i + 1][0]
        self.distance = np.sqrt(np.square(self.y_difference) + np.square(self.x_difference))

        ###########################################
        gear = self.gear
        if gear == 1:
            gearEfficiency = 0.91

        if gear == 2:
            gearEfficiency = 0.91

        if gear == 3:
            gearEfficiency = 0.91

        if gear == 4:
            gearEfficiency = 0.96

        if gear == 5:
            gearEfficiency = 0.96

        if gear == 6:
            gearEfficiency = 0.96

        N = self.m * 9.81 * np.cos(self.theta[i]) + self.downforceCoeff * np.square(self.v)
        self.theta[i] = 0.0
        self.theta[i + 1] = 0.0

        tempVelo = ((self.v_previous) + self.v_desired) / 2

        if tempVelo == 0:
            tempVelo = 0.000000000001

        R = self.getRadius(i, waypoints)

        self.alphar = (630 / 1.9) * tempVelo / (9.81 * R * self.Calpha)
        # introduce the longitudinal load transfer later

        # sqrt(np.square(self.m*np.square(v1)/R-self.m*9.81*sin(self.theta[self.x])) + np.square(deltav[self.x]*(self.v[self.x]+self.v[self.x-1])/(2*self.d) + self.rollingFriction + 0.445*np.square(self.v[self.x-1]))) = self.mu*N #gives delta

        # limiting factor: friction
        tmp1 = np.square(self.mu * N) - np.square(self.m * tempVelo * tempVelo * math.cos(
            (self.theta[i] + self.theta[i + 1]) / 2) / R - self.m * 9.81 * math.sin(
            (self.theta[i] + self.theta[i + 1]) / 2))

        # limiting factor: Powertrain
        tmp2 = np.square(self.FintoV / tempVelo) / math.cos(self.alphar)

        if tmp1 < 0 or tmp2 < 0:
            self.throttle_output = 0

        else:
            deltaUpper1 = np.sqrt((np.sqrt(
                tmp1) - self.dragCoeff * tempVelo * tempVelo - self.rollingFriction) * 2 * self.distance / self.m + self.v * self.v) - self.v

            deltaLower1 = np.sqrt((-np.sqrt(
                tmp1) - self.dragCoeff * tempVelo * tempVelo - self.rollingFriction) * 2 * self.distance / self.m + self.v * self.v) - self.v

            deltaUpper2 = np.sqrt((np.sqrt(
                tmp2) - self.dragCoeff * tempVelo * tempVelo - self.rollingFriction) * 2 * self.distance / self.m + self.v * self.v) - self.v

            deltaUpper = min(deltaUpper1, deltaUpper2)
            deltaLower = deltaLower1
            
            self.v_desired = min(waypoints[i + 1][2], self.v + deltaUpper)

            accel_desired = (waypoints[i + 2][2] * waypoints[i + 2][2] - waypoints[i + 1][2] * waypoints[i + 1][
                2]) / 2 * self.d

            self.throttle_output = np.minimum(np.maximum(tempVelo * (self.m * (
                    (self.v_desired * self.v_desired - self.v * self.v) * 0.85 + accel_desired * 0.15) / (
                                                                             2 * self.distance) + self.dragCoeff * tempVelo * tempVelo + self.rollingFriction) / (
                                                                 gearEfficiency * self.maxPower), 0), 1)

        ###########################################
        if (
                self.v_desired * self.v_desired < -2 * self.distance * self.dragCoeff * tempVelo * tempVelo / 630 + self.v * self.v):
            if (waypoints[i + 1][2] * waypoints[i + 1][2] < -2 * (
                    self.distance + 4) * self.dragCoeff * tempVelo * tempVelo / 630 + self.v * self.v):
                self.throttle_output = 0
                self.brake_output = 40 * np.minimum(np.maximum(0.72 * (-0.1 * (
                        (self.v_desired - self.v) * 0.9 - (self.v - self.v_previous) * 5 + (
                        self.v_desired - self.v_req_previous) * 6 + self.distance * 0.1)), 0), 1)

        if (self.x - self.x_previous > 0):
            self.fangle = np.arctan((self.y - self.y_previous) / (self.x - self.x_previous))
        elif (self.x - self.x_previous == 0):
            if (self.y > self.y_previous):
                self.fangle = np.pi / 2
            elif (self.y < self.y_previous):
                self.fangle = -np.pi / 2
            else:
                self.fangle = 0
        else:
            self.fangle = np.arctan((self.y - self.y_previous) / (self.x - self.x_previous)) + np.pi

        if (waypoints[i + 2][0] - self.x > 0):
            self.fanglea = np.arctan((waypoints[i + 2][1] - self.y) / (waypoints[i + 2][0] - self.x))
        elif (waypoints[i + 2][0] - self.x == 0):
            if (waypoints[i + 2][1] - self.y > 0):
                self.fanglea = np.pi / 2
            elif (waypoints[i + 2][1] - self.y < 0):
                self.fanglea = -np.pi / 2
            else:
                self.fanglea = 0
        else:
            self.fanglea = np.arctan((waypoints[i + 2][1] - self.y) / (waypoints[i + 2][0] - self.x)) + np.pi

        if (waypoints[i + 4][0] - self.x > 0):
            self.fangleb = np.arctan((waypoints[i + 4][1] - self.y) / (waypoints[i + 4][0] - self.x))
        elif (waypoints[i + 4][0] - self.x == 0):
            if (waypoints[i + 4][1] - self.y > 0):
                self.fangleb = np.pi / 2
            elif (waypoints[i + 4][1] - self.y < 0):
                self.fangleb = -np.pi / 2
            else:
                self.fangleb = 0
        else:
            self.fangleb = np.arctan((waypoints[i + 4][1] - self.y) / (waypoints[i + 4][0] - self.x)) + np.pi

        if (waypoints[i + 5][0] - waypoints[i + 4][0] > 0):
            self.fanglec = np.arctan(
                (waypoints[i + 5][1] - waypoints[i + 4][1]) / (waypoints[i + 5][0] - waypoints[i + 4][0]))
        elif (waypoints[i + 5][0] - waypoints[i + 4][0] == 0):
            if (waypoints[i + 5][1] - waypoints[i + 4][1] > 0):
                self.fanglec = np.pi / 2
            elif (waypoints[i + 5][1] - waypoints[i + 4][1] < 0):
                self.fanglec = -np.pi / 2
            else:
                self.fanglec = 0
        else:
            self.fanglec = np.arctan((waypoints[i + 5][1] - waypoints[i + 4][1]) / (
                    waypoints[i + 5][0] - waypoints[i + 4][0])) + np.pi

        if (waypoints[i + 1][0] - waypoints[i][0] > 0):
            self.fangled = np.arctan(
                (waypoints[i + 1][1] - waypoints[i][1]) / (waypoints[i + 1][0] - waypoints[i][0]))
        elif (waypoints[i + 1][0] - waypoints[i][0] == 0):
            if (waypoints[i + 1][1] - waypoints[i][1] > 0):
                self.fangled = np.pi / 2
            elif (waypoints[i + 1][1] - waypoints[i][1] < 0):
                self.fangled = -np.pi / 2
            else:
                self.fangled = 0
        else:
            self.fangled = np.arctan(
                (waypoints[i + 1][1] - waypoints[i][1]) / (waypoints[i + 1][0] - waypoints[i][0])) + np.pi

        self.k_1 = 0.008 * -5
        self.k_2 = 0.07 * -3
        self.k_3 = 0.3 * -2.5
        self.k_4 = 0.60 * -1.5

        self.diff_1 = self.fangle - self.fanglea
        self.diff_2 = self.fangle - self.fangleb
        self.diff_3 = self.fangle - self.fanglec
        self.diff_4 = self.fangle - self.fangled

        if (self.fangle - self.fanglea > np.pi):
            self.diff_1 = -(2 * np.pi - self.fangle + self.fanglea)
        elif (self.fangle - self.fanglea < -np.pi):
            self.diff_1 = self.fangle - self.fanglea + 2 * np.pi

        if (self.fangle - self.fangleb > np.pi):
            self.diff_2 = -(2 * np.pi - self.fangle + self.fangleb)
        elif (self.fangle - self.fangleb < -np.pi):
            self.diff_2 = self.fangle - self.fangleb + 2 * np.pi

        if (self.fangle - self.fanglec > np.pi):
            self.diff_3 = -(2 * np.pi - self.fangle + self.fanglec)
        elif (self.fangle - self.fanglec < -np.pi):
            self.diff_3 = self.fangle - self.fanglec + 2 * np.pi

        if (self.fangle - self.fangled > np.pi):
            self.diff_4 = -(2 * np.pi - self.fangle + self.fangled)
        elif (self.fangle - self.fangled < -np.pi):
            self.diff_4 = self.fangle - self.fangled + 2 * np.pi

        self.steer_put = (self.k_1 * (self.diff_1) + self.k_2 * (self.diff_2) + self.k_3 * (
            self.diff_3) + self.k_4 * (self.diff_4))
        self.diffangle = self.fangle - self.steer_previous
        self.steer_output = self.steer_put + (self.prev_diffangle - self.diffangle) * 1.5

        minR = self.m * tempVelo * tempVelo * math.cos((self.theta[i] + self.theta[i + 1]) / 2) / (
                self.m * 9.81 * math.sin((self.theta[i] + self.theta[i + 1]) / 2) + self.mu * N)

        if self.steer_output > 0:
            self.steer_output = np.minimum(
                minR * minR * 1.9 * 9.81 * self.Calpha / (self.L * 630 * tempVelo) + 630 * tempVelo / (
                        1.9 * 9.81 * minR * self.Calpha), self.steer_output)

        if self.steer_output < 0:
            self.steer_output = np.maximum(
                -minR * minR * 1.9 * 9.81 * self.Calpha / (self.L * 630 * tempVelo) - 630 * tempVelo / (
                        1.9 * 9.81 * minR * self.Calpha), self.steer_output)

        # self.alphar= (630/1.9)*tempVelo/(9.81*R*self.Calpha)
        if (self.steer_output >= 1):
            self.steer_output = 1
        elif (self.steer_output <= -1):
            self.steer_output = -1

        # if ( self.v > 44.2222):
        # self.throttle_output = 1
        if (self.v > 86.77 and self.v < 86.9444):
            self.throttle_output = 0.7

        # steer = self.steer_output - steeroutput_previous
        # throttle = self.throttle_output - throttleoutput_previous

        # newVariableGetter= [self.steer_output,self.throttle_output,self.v,self.x,self.y]
        # dataGetter.append(newVariableGetter)

        # if loopHelper==500:
        #    with open('dataGet.csv','w',newline='') as csvfile:
        #        fieldnames = ['Steer','Throttle','self.v','self.x','self.y']
        #        thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
        #        thewriter.writeheader()
        #        for t in range(loopHelper-5):
        #            thewriter.writerow({'Steer':dataGetter[t+2][0],'Throttle':dataGetter[t+2][1],'self.v':dataGetter[t+2][2],'self.x':dataGetter[t+2][3],'self.y':dataGetter[t+2][4]})
        #            loopHelper+=1
        # else:
        #    loopHelper+=1
        print('Control State Values')
        print('--------Error-----------')
        print(self.v - self.v_desired, self.x_difference, self.y_difference)
        print('--------Throttle-----------')
        print(self.throttle_output)
        print('---------Brake----------')
        print(self.brake_output)
        print('--------Steer-----------')
        print(self.steer_output)
        print('-------------------')

        self.v_pre_previous = self.v_previous
        self.v_previous = self.v
        self.x_previous = self.x
        self.y_previous = self.y
        self.v_req_previous = self.v_desired
        k = i
        self.steeroutput_previous = self.steer_output
        self.throttleoutput_previous = self.throttle_output
        # self.acceleration_previous = acceleration
        dist = float("inf")

        '''
        for j in range(len(waypoints)):
            if dist >= np.sqrt(np.square(self.y-waypoints[j][1])+np.square(self.x-waypoints[j][0])):
                if j>1000 and i<100:
                    print('Waypoint Denied')
                else:
                    dist = np.sqrt(np.square(self.y-waypoints[j][1])+np.square(self.x-waypoints[j][0]))
                    if j >= 1054:
                        j = j-1054+39
                    i = j + 2
        '''
        #a_indy = 0
        # self.lap = waypoints[i][3]
        # self.nozero = 0
        # if self.lap == 0 and self.nozero == 0:
        #     for j in range(0, 5):
        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             print(i, waypoints[i][3])
        #             if self.lap == waypoints[j][3]:
        #                 # if waypoints[i][3] == waypoints[j][3]:
        #                 dist = np.sqrt(
        #                     np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                 a_indy = j + 2
        #             # if j == 4:
        #             # 	self.lap += 1
        # if self.lap == 1 and self.noone == 0:
        #     self.nozero = 1
        #     for j in range(5, 1020):

        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             if i < 1015 and j < 1015:
        #                 print(i, waypoints[i][3])
        #                 if self.lap == waypoints[j][3]:
        #                     # if waypoints[i][3] == waypoints[j][3]:
        #                     dist = np.sqrt(
        #                         np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                     a_indy = j + 2
        #                 # if j == 1019:
        #                 # 	self.lap += 1
        #             elif i > 1015 and j > 1015:
        #                 self.lap += 1
        # if self.lap == 2 and self.notwo == 0:
        #     self.noone = 1
        #     for j in range(1020, 2035):
        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             if i < 2020 and j < 2020:
        #                 print(i, waypoints[i][3])
        #                 if self.lap == waypoints[j][3]:
        #                     # if waypoints[i][3] == waypoints[j][3]:
        #                     dist = np.sqrt(
        #                         np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                     a_indy = j + 2
        #                 # if j == 2034:
        #                 # 	self.lap += 1
        #             elif i > 2020 and j > 2020:
        #                 self.lap += 1
        # if self.lap == 3 and self.nothree == 0:
        #     self.notwo = 1
        #     for j in range(2035, 3050):
        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             if i < 3035 and j < 3035:
        #                 print(i, waypoints[i][3])
        #                 if self.lap == waypoints[j][3]:
        #                     # if waypoints[i][3] == waypoints[j][3]:
        #                     dist = np.sqrt(
        #                         np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                     a_indy = j + 2
        #                 # if j == 3049:
        #                 # 	self.lap += 1
        #             elif i > 3035 and j > 3035:
        #                 self.lap += 1
        # if self.lap == 4 and self.nofour == 0:
        #     self.nothree = 1
        #     for j in range(3050, 4065):
        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             if i < 4050 and j < 4050:
        #                 print(i, waypoints[i][3])
        #                 if self.lap == waypoints[j][3]:
        #                     # if waypoints[i][3] == waypoints[j][3]:
        #                     dist = np.sqrt(
        #                         np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                     a_indy = j + 2
        #                 # if j == 4064:
        #                 # 	self.lap += 1
        #             elif i > 4050 and j > 4050:
        #                 self.lap += 1
        # if self.lap == 5 and self.nofive == 0:
        #     self.nofour = 1
        #     for j in range(4065, 5080):
        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             if i < 5065 and j < 5065:
        #                 print(i, waypoints[i][3])
        #                 if self.lap == waypoints[j][3]:
        #                     # if waypoints[i][3] == waypoints[j][3]:
        #                     dist = np.sqrt(
        #                         np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                     a_indy = j + 2
        #                 # if j == 5079:
        #                 # 	self.lap += 1
        #             elif i > 5065 and j > 5065:
        #                 self.lap += 1
        # if self.lap == 6:
        #     self.nofive = 1
        #     for j in range(5080, len(waypoints)):
        #         if dist >= np.sqrt(np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0])):
        #             if i < len(waypoints) and j < len(waypoints):
        #                 print(i, waypoints[i][3])
        #                 if self.lap == waypoints[j][3]:
        #                     # if waypoints[i][3] == waypoints[j][3]:
        #                     dist = np.sqrt(
        #                         np.square(self.y - waypoints[j][1]) + np.square(self.x - waypoints[j][0]))
        #                     a_indy = j + 2

        # for j in range(i-2,i+30):
        #     if dist >= np.sqrt(np.square(self.y-waypoints[j][1])+np.square(self.x-waypoints[j][0])):
        #         dist = np.sqrt(np.square(self.y-waypoints[j][1])+np.square(self.x-waypoints[j][0]))
        #         a_indy = j

        # for j in range(self.globalpathi-2,self.globalpathi+30):
        #     if dist >= np.sqrt(np.square(self.y-waypoints[j][1])+np.square(self.x-waypoints[j][0])):
        #         dist = np.sqrt(np.square(self.y-waypoints[j][1])+np.square(self.x-waypoints[j][0]))
        #         a_indy = j

        # i = a_indy
        # i = i+1
        #self.globalpathi = a_indy
        #self.globalpathi = self.globalpathi + 1
        #self.lap = waypoints[self.globalpathi][3]

        #print("Index I")
        #print(self.globalpathi)
        self.steer_previous = self.fangle
        self.prev_diffangle = self.diffangle
        self.runCount += 1

        print(self.steer_output)

        return self.throttle_output, self.steer_output

# if __name__ == '__main__':
#     controllerObject = Controller()
#     controllerObject.rti_connext()
#     controllerObject.read_global_path_csv()
#     while True:
#         controllerObject.run_controller_timestep(0)
