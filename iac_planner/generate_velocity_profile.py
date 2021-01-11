import numpy as np
import math
from iac_planner.helpers import Env, path_t

def getRadius(index, path: path_t):
    waypoint= [[0.0 for xx in range(4)] for yy in range(len(path)-1)] 
    x, y = path[:, 0], path[:, 1]

    for counted in range(len(path)-2):
        if counted > 0:
            x[counted-1] = float(x[counted])
            y[counted-1] = float(y[counted]) 
            waypoint[counted-1][0] = float(x[counted])
            waypoint[counted-1][1] = float(y[counted])
    
    a,b,c,dd,e,f= waypoint[index][0],waypoint[index][1],waypoint[index-1][0],waypoint[index-1][1],waypoint[index+1][0],waypoint[index+1][1]
    
    A= np.array([[2*a-2*c,2*b-2*dd],[2*a-2*e,2*b-2*f]])
    B= np.array([c*c+dd*dd-a*a-b*b,e*e+f*f-a*a-b*b])
    
    if (a-c)*(b-f) == (a-e)*(dd-b):
        return 100000000000000000.0
    else:
        solution= np.linalg.solve(A,B)
        return np.sqrt(np.square(solution[0]-a)+np.square(solution[1]-b))

def BetterPredict(index, waypoints,N ,v,theta,env: Env, path: path_t):
    #waypoints[index][3]= np.sqrt((-np.sqrt(tmp1) - dragCoeff*tempVel*tempVel - rollingFriction)*2*d/m + v[xx-1]*v[xx-1])
    #tmp1= np.square(mu*N) - np.square(m*tempVel*tempVel*math.cos((theta[index-1]+theta[index-2])/2)/R - m*9.81*math.sin((theta[index-1]+theta[index-2])/2))
    param = env.vel_params
    tempVel = param.tempVel
    m = param.m
    mu = param.mu
    rollingFriction = param.rolling_friction
    d = param.d
    dragCoeff = param.dragCoeff

    tempVel= waypoints[index][3]
    R= getRadius(index,path)
    
    tmp1= np.square(mu*N) - np.square(m*tempVel*tempVel*math.cos((theta[index-1]+theta[index])/2)/R - m*9.81*math.sin((theta[index-1]+theta[index])/2))

    c= waypoints[index][3]*waypoints[index][3] + (np.sqrt(tmp1) + dragCoeff*tempVel*tempVel + rollingFriction)*2*d/m
    
    waypoints[index-1][3] = np.sqrt(c)
    
    tempVel= (waypoints[index-2][3]) + (waypoints[index-2][3]-waypoints[index-3][3])/2
    tmp1= np.square(mu*N) - np.square(m*tempVel*tempVel*math.cos((theta[index-1]+theta[index-2])/2)/R - m*9.81*math.sin((theta[index-1]+theta[index-2])/2))
    LowerLimit= np.sqrt((-np.sqrt(tmp1) - dragCoeff*tempVel*tempVel - rollingFriction)*2*d/m + v[index-2]*v[index-2])
    
    print(waypoints[index-1][3], LowerLimit)
    if waypoints[index-1][3]< LowerLimit:
        print("4")
        return BetterPredict(index-1, waypoints,N ,v ,theta,env,path)
    else:
        print("5")
        return waypoints

def stepBack(index, tempVelocity, N ,v ,theta, waypoints,env: Env, path: path_t):
    param = env.vel_params
    mu = param.mu
    dragCoeff = param.dragCoeff
    rollingFriction = param.rolling_friction
    d = param.delta
    m = param.m

    w, h= 4, len(path) - 1
    waypoint= [[0.0 for xx in range(w)] for yy in range(h)] 
    secondaryWaypoints= [[0.0 for xx in range(w)] for yy in range(h)]
    
    for k in range(h-1):
        for kk in range(4):
            secondaryWaypoints[k][kk]= waypoints[k][kk]
    

    x, y = path[:, 0], path[:, 1]
    

    for counted in range(h-1):
        if counted > 0:
            x[counted-1] = float(x[counted])
            y[counted-1] = float(y[counted]) 
            waypoint[counted-1][0] = float(x[counted])
            waypoint[counted-1][1] = float(y[counted])

    #tempVelocity= (waypoints[xx-1][3]) + (waypoints[xx-1][3]-waypoints[xx-2][3])/2
    secondaryWaypoints[index-1][3]= tempVelocity
    R= getRadius(index-1,path)
    
    tempVel= (waypoints[index-2][3]) + (waypoints[index-2][3]-waypoints[index-3][3])/2
    
    tmp1= np.square(mu*N) - np.square(m*tempVel*tempVel*math.cos((theta[index-1]+theta[index-2])/2)/R - m*9.81*math.sin((theta[index-1]+theta[index-2])/2))
    LowerLimit= np.sqrt((-np.sqrt(tmp1) - dragCoeff*tempVel*tempVel - rollingFriction)*2*d/m + v[index-2]*v[index-2])
    print("0000",tmp1)
    print(secondaryWaypoints[index-1][3], LowerLimit)
    
    
    if secondaryWaypoints[index-1][3]< LowerLimit:
        print("2")
        return BetterPredict(index-1, secondaryWaypoints,N ,v ,theta,env, path)
    else:
        print("3")
        return secondaryWaypoints




def generate_velocity_profile(env: Env, path: path_t) -> np.ndarray:
    """

    :param env: Env
    :param path: [n x 2] float
    :return: [n-1] float
    """
    # TODO: Many repeated calculations, store value
    # e.g.
    # if a > foo(a): a = foo(a)
    # if a > (a_max := foo(a)): a = a_max)

    v = np.zeros(len(path) - 1)
    deltav = np.zeros(len(path) - 1)
    theta = np.zeros(len(path) - 1)
    deltav[0] = 0.0
    tempVel= 0.0
    tmp= 0.0
    v[0] = env.state[3]
    waypoint= [[0.0 for xx in range(4)] for yy in range(len(path) - 1)] 
    waypoints= [[0.0 for xx in range(4)] for yy in range(len(path) - 1)]

    x, y = path[:, 0], path[:, 1]

    for counted in range(len(path) - 2):
        if counted > 0:
            x[counted-1] = float(x[counted])
            y[counted-1] = float(y[counted]) 
            waypoint[counted-1][0] = float(x[counted])
            waypoint[counted-1][1] = float(y[counted])
                
    waypoints[0][3] = v[0]

    for xx in range(1, len(path) - 2):
        params = env.vel_params
        m = params.m
        mu = params.mu
        rollingFriction = params.rolling_friction
        downforceCoeff = params.downforceCoeff
        Calpha = params.Calpha
        FintoV = params.FintoV
        dragCoeff = params.dragCoeff
        N = m*9.81*np.cos(theta[xx]) + downforceCoeff*np.square(v[xx])

        # TODO: What are these magic numbers? g and __ Cd ?
        d = 10
        u = v[xx - 1]
        theta[xx] = 0.0
        theta[xx+1] = 0.0

        if xx> 1:
            tempVelo= (waypoints[xx-1][3]) + (waypoints[xx-1][3]-waypoints[xx-2][3])/2
        else:
            tempVelo= waypoints[xx-1][3]-waypoints[xx-2][3]

        if tempVelo == 0:
            tempVelo= 0.000000000001

        R = getRadius(xx, path)

        alphar= (630/1.9)*tempVelo/(9.81*R*Calpha)
        #introduce the longitudinal load transfer later
    
        #sqrt(np.square(m*np.square(v1)/R-m*9.81*sin(theta[x])) + np.square(deltav[x]*(v[x]+v[x-1])/(2*d) + rollingFriction + 0.445*np.square(v[x-1]))) = mu*N #gives delta
    
        #limiting factor: friction
        tmp1= np.square(mu*N) - np.square(m*tempVelo*tempVelo*math.cos((theta[xx]+theta[xx+1])/2)/R - m*9.81*math.sin((theta[xx]+theta[xx+1])/2))

        #limiting factor: Powertrain
        tmp2= np.square(FintoV/tempVelo)/math.cos(alphar)

        if tmp1 < 0:
            tempVelocity= np.sqrt((mu*N + m*9.81*math.sin((theta[xx]+theta[xx+1])/2))*R/(m*math.cos((theta[xx]+theta[xx+1])/2)))

            waypoints= stepBack(xx, tempVelocity, N ,v ,theta, waypoints, path)
            if xx> 1:
                tempVelo= (waypoints[xx-1][3]) + (waypoints[xx-1][3]-waypoints[xx-2][3])/2 
            else: 
                tempVelo= waypoints[xx-1][3]-waypoints[xx-2][3]

            if tempVelo == 0:
                tempVelo= 0.000000000001

            alphar= (630/1.9)*tempVel/(9.81*R*Calpha)
            tmp1= np.square(mu*N) - np.square(m*tempVelo*tempVelo*math.cos((theta[xx]+theta[xx+1])/2)/R - m*9.81*math.sin((theta[xx]+theta[xx+1])/2))
            tmp2= np.square(FintoV/tempVelo)/math.cos(alphar)

        deltaUpper1= np.sqrt((np.sqrt(tmp1) - dragCoeff*tempVelo*tempVelo - rollingFriction)*2*d/m + v[xx-1]*v[xx-1]) - v[xx-1]
    
        deltaLower1= np.sqrt((-np.sqrt(tmp1) - dragCoeff*tempVelo*tempVelo - rollingFriction)*2*d/m + v[xx-1]*v[xx-1]) - v[xx-1]
    
        deltaUpper2= np.sqrt((np.sqrt(tmp2) - dragCoeff*tempVelo*tempVelo - rollingFriction)*2*d/m + v[xx-1]*v[xx-1]) - v[xx-1]  
        deltaUpper= min(deltaUpper1, deltaUpper2)
        deltaLower= deltaLower1
        deltav[xx]= deltaUpper

        v[xx]= waypoints[xx-1][3] + deltav[xx]

    return v
